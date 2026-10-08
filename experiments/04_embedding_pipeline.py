import os #funciones para interactuar con el sistema operativo
import json #trabajar con datos JSON
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import time
import re
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_community.llms import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class PolicyEmbedder:
    def __init__(self, policy_text):
        """
        Initialise l'intégrateur de politique avec des embeddings et un stockage vectoriel
        
        Args:
            policy_text (str): Texte complet de la politique de bagages
        """
        # Initialiser le modèle d'embedding
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Créer un répertoire pour les embeddings s'il n'existe pas
        embeddings_dir = "./policy_embeddings"
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Créer un client Chroma avec la nouvelle configuration
        self.chroma_client = chromadb.PersistentClient(path=embeddings_dir)
        
        # Créer ou récupérer la collection
        self.collection = self.chroma_client.get_or_create_collection(name="luggage_policy")
        
        # Créer des embeddings pour la politique
        self._create_policy_embeddings(policy_text)
    
    def _create_policy_embeddings(self, policy_text):
        """
        Divise le texte de la politique en chunks et crée des embeddings
        
        Args:
            policy_text (str): Texte complet de la politique
        """
        # Diviser la politique en sections
        policy_chunks = policy_text.split('\n\n')
        
        # Filtrer les chunks vides ou trop courts
        policy_chunks = [chunk for chunk in policy_chunks if len(chunk) > 20]
        
        # Générer des embeddings
        embeddings = self.embedding_model.encode(policy_chunks)
        
        # Ajouter les chunks à la collection
        for i, (chunk, embedding) in enumerate(zip(policy_chunks, embeddings)):
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[chunk],
                ids=[f"policy_chunk_{i}"]
            )
    
    def retrieve_relevant_context(self, query, top_k=3):
        """
        Récupère les contextes de politique les plus pertinents
        
        Args:
            query (str): Requête pour laquelle on cherche du contexte
            top_k (int): Nombre de chunks à retourner
        
        Returns:
            list: Chunks de politique les plus pertinents
        """
        # Encoder la requête
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Rechercher les chunks les plus similaires
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0]


def calculate_metrics(results, segment_name="total"):
    """Calcule toutes les métriques à partir des résultats."""
    total = len(results)
    if total == 0:
        return None
    
    # Métriques de base
    exploitable_count = sum(1 for r in results if r['predicted']['compliance_message'] != "Échec de l'analyse")
    exploitable_rate = exploitable_count / total * 100
    
    # Métriques de conformité
    compliance_correct = sum(1 for r in results if r['predicted']['compliance_result'] == r['actual']['compliance_result'])
    compliance_accuracy = compliance_correct / total * 100
    
    # Métriques de frais
    fees_correct = sum(1 for r in results if abs(r['predicted']['fees'] - r['actual']['fees']) < 1)
    fees_accuracy = fees_correct / total * 100
    
    # Erreur moyenne des frais
    fee_errors = [abs(r['predicted']['fees'] - r['actual']['fees']) for r in results]
    mean_fee_error = sum(fee_errors) / total
    
    # Métriques F1
    actual_compliance = [bool(r['actual']['compliance_result']) for r in results]
    predicted_compliance = [bool(r['predicted']['compliance_result']) for r in results]
    f1 = f1_score(actual_compliance, predicted_compliance, average='binary')
    
    # Efficacité globale (moyenne des précisions)
    global_efficiency = (compliance_accuracy + fees_accuracy) / 2
    
    # Analyse des cas problématiques
    problem_cases = []
    fee_error_counts = {}
    
    for i, r in enumerate(results):
        problems = []
        
        # Vérifier les erreurs de conformité
        if r['predicted']['compliance_result'] != r['actual']['compliance_result']:
            problems.append("conformité")
        
        # Vérifier les erreurs de frais
        fee_error = abs(r['predicted']['fees'] - r['actual']['fees'])
        if fee_error >= 1:
            problems.append(f"frais ({fee_error:.2f})")
            
            # Arrondir l'erreur pour le comptage
            rounded_error = round(fee_error)
            if rounded_error in fee_error_counts:
                fee_error_counts[rounded_error] += 1
            else:
                fee_error_counts[rounded_error] = 1
        
        if problems:
            problem_cases.append((i, problems))
    
    # Métriques ajustées (si nécessaire)
    compliance_accuracy_adjusted = compliance_accuracy
    fees_accuracy_adjusted = fees_accuracy
    
    # Types d'erreurs
    error_types = []
    
    # Erreurs de conformité
    conformity_errors = sum(1 for r in results if r['predicted']['compliance_result'] != r['actual']['compliance_result'])
    if conformity_errors > 0:
        error_types.append({
            "type": "Erreur de conformité",
            "count": conformity_errors,
            "percentage": conformity_errors / total * 100
        })
    
    # Erreurs de frais par montant
    for error_amount, count in sorted(fee_error_counts.items(), key=lambda x: x[1], reverse=True):
        error_types.append({
            "type": f"Erreur de frais ({error_amount:.2f})",
            "count": count,
            "percentage": count / total * 100
        })
    
    return {
        "segment": segment_name,
        "total_cases": total,
        "exploitable_rate": exploitable_rate,
        "compliance_accuracy": compliance_accuracy,
        "fees_accuracy": fees_accuracy,
        "f1_score": f1,
        "mean_fee_error": mean_fee_error,
        "compliance_accuracy_adjusted": compliance_accuracy_adjusted,
        "fees_accuracy_adjusted": fees_accuracy_adjusted, 
        "global_efficiency": global_efficiency,
        "problem_cases_count": len(problem_cases),
        "error_types": error_types
    }

def print_metrics_report(metrics):
    """Affiche un rapport de métriques formaté."""
    if not metrics:
        print("Aucune donnée disponible pour générer un rapport.")
        return
    
    print(f"\n=== Analyse des résultats ({metrics['segment']}) ===")
    print("Métriques de précision et d'exploitabilité:")
    print(f"Taux d'exploitabilité: {metrics['exploitable_rate']:.2f}%")
    print(f"Précision de conformité: {metrics['compliance_accuracy']:.2f}%")
    print(f"Précision des frais: {metrics['fees_accuracy']:.2f}%")
    print(f"Score F1 (conformité): {metrics['f1_score']:.4f}")
    print(f"Erreur moyenne des frais: {metrics['mean_fee_error']:.2f} €")
    
    print("Métriques ajustées (tenant compte des résultats inexploitables):")
    print(f"Précision de conformité ajustée: {metrics['compliance_accuracy_adjusted']:.2f}%")
    print(f"Précision des frais ajustée: {metrics['fees_accuracy_adjusted']:.2f}%")
    print(f"Efficacité globale: {metrics['global_efficiency']:.2f}%")
    
    print(f"Cas problématiques identifiés: {metrics['problem_cases_count']}/{metrics['total_cases']}")
    
    if metrics['error_types']:
        print("Types de problèmes rencontrés:")
        for error in metrics['error_types']:
            print(f"•⁠  ⁠{error['type']}: {error['count']} cas ({error['percentage']:.1f}%)")
            
class RAGEnhancedLuggageAnalysisPipeline:
    def __init__(self, policy, model, project_id=None):
        """
        Initialise le pipeline d'analyse avec RAG
        
        Args:
            policy (str): Texte de la politique de bagages
            model: Instance du modèle WatsonxLLM
            project_id (str, optional): ID du projet Watson
        """
        self.policy = policy
        self.model = model
        self.project_id = project_id
        
        # Initialiser l'embedder de politique
        self.policy_embedder = PolicyEmbedder(policy)
        
        # Initialiser les différents prompts pour chaque étape
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialise les différents prompts pour chaque étape du pipeline."""
        
        # 1. Prompt pour l'analyse des bagages à main avec contexte RAG
        self.carry_on_prompt = PromptTemplate(
            input_variables=["policy", "policy_context", "travel_class", "age_category", "luggages"],
            template="""You are a luggage compliance expert focusing ONLY on carry-on baggage. 

CONTEXTE DE POLITIQUE SPÉCIFIQUE:
{policy_context}

Politique générale:
{policy}

Détails du cas:
• Travel Class: {travel_class}
• Age Category: {age_category}
• Luggage Details: {luggages}

INSTRUCTIONS:
1. Utilisez le contexte spécifique pour analyser précisément les bagages à main
2. Identifier tous les articles de cabine et personnels
3. Vérifier la conformité selon la classe et la catégorie d'âge
4. Calculer les frais supplémentaires

Répondez UNIQUEMENT avec un objet JSON valide :
{{
"items_identified": ["liste des articles de cabine"],
"compliance_result": true/false,
"compliance_message": "Explication brève du statut",
"excess_fees": 0
}}"""
        )
        
        # 2. Prompt pour l'analyse des bagages enregistrés
        self.checked_baggage_prompt = PromptTemplate(
            input_variables=["policy", "travel_class", "age_category", "luggages", "carry_on_result"],
            template="""You are a luggage compliance expert focusing ONLY on checked baggage. 
Review the following airline luggage policy about checked baggage allowances:

{policy}

Now, analyze ONLY the checked baggage items in the following case:
• Travel Class: {travel_class}
• Age Category: {age_category}
• Luggage Details: {luggages}

We already analyzed the carry-on items with this result:
{carry_on_result}

IMPORTANT: Focus ONLY on checked baggage, not carry-on items.

1. Identify all checked baggage items from the luggage details
2. Analyze if they comply with the allowed number, weight, and dimensions for the travel class and age category
3. Calculate any fees for extra pieces, overweight, or oversized checked baggage
4. Provide a clear compliance status

You must respond ONLY with a valid JSON object in the following format without any additional text:
{{
"items_identified": ["list of all checked baggage items identified"],
"compliance_result": true/false,
"compliance_message": "Brief explanation of checked baggage compliance status",
"excess_fees": 0
}}"""
        )
        
        # 3. Prompt pour l'identification des articles cargo
        self.cargo_items_prompt = PromptTemplate(
            input_variables=["policy", "travel_class", "age_category", "luggages", "carry_on_result", "checked_result"],
            template="""You are a luggage compliance expert focusing ONLY on identifying cargo or special handling items.
Review the following airline luggage policy:

{policy}

Now, identify ONLY items requiring cargo handling or special treatment in the following case:
• Travel Class: {travel_class}
• Age Category: {age_category}
• Luggage Details: {luggages}

We already analyzed:
- Carry-on items: {carry_on_result}
- Checked baggage: {checked_result}

IMPORTANT: Focus ONLY on identifying items that:
- Cannot be transported as carry-on or checked baggage
- Require special handling or cargo processing
- Are mentioned in the special items or prohibited items sections of the policy

You must respond ONLY with a valid JSON object in the following format without any additional text:
{{
"cargo_items": ["list of items requiring cargo service or special handling"]
}}"""
        )
        
        # 4. Prompt pour le calcul final des frais et la conformité globale
        self.final_analysis_prompt = PromptTemplate(
            input_variables=["policy", "travel_class", "age_category", "luggages", 
                           "carry_on_result", "checked_result", "cargo_result"],
            template="""You are a luggage compliance expert providing the FINAL analysis.
Review the airline luggage policy:

{policy}

Case details:
• Travel Class: {travel_class}
• Age Category: {age_category}
• Luggage Details: {luggages}

We have analyzed each component:
- Carry-on analysis: {carry_on_result}
- Checked baggage analysis: {checked_result}
- Cargo items identified: {cargo_result}

IMPORTANT: Your task is to:
1. Combine all the previous analyses to determine overall compliance
2. Calculate the TOTAL fees by adding carry-on and checked baggage fees
3. Provide a concise final compliance message that covers all issues

You must respond ONLY with a valid JSON object in the following format without any additional text:
{{
"compliance_result": true/false,
"compliance_message": "Brief explanation of overall compliance status",
"cargo_items": ["list from cargo analysis"],
"fees": 0
}}"""
        )
    
    def _call_model(self, prompt, input_data):
        """
        Appelle le modèle avec un prompt spécifique et récupère des contextes RAG
        
        Args:
            prompt: Le prompt à utiliser
            input_data: Les données d'entrée pour le prompt
            
        Returns:
            dict: Le résultat JSON parsé ou None en cas d'échec
        """
        # Récupérer le contexte pertinent avec RAG
        context = self.policy_embedder.retrieve_relevant_context(
            input_data.get('luggages', ''), 
            top_k=2
        )
        
        # Ajouter le contexte aux données d'entrée
        input_data['policy_context'] = "\n".join(context)
        
        chain = prompt | self.model
        
        for attempt in range(3):  # Essayer jusqu'à 3 fois
            try:
                result = chain.invoke(input_data)
                
                # Essayer de parser le JSON
                try:
                    parsed_result = json.loads(result)
                    return parsed_result
                except json.JSONDecodeError:
                    # Si échec, essayer d'extraire le JSON du texte
                    parsed_result = self._extract_json_from_text(result)
                    if parsed_result:
                        return parsed_result
                    
                    print(f"Échec de l'extraction JSON (tentative {attempt + 1})")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"Erreur lors de l'appel du modèle (tentative {attempt + 1}): {str(e)}")
                time.sleep(2)
        
        return None
    
    def _extract_json_from_text(self, text):
        """
        Extrait le JSON d'une réponse textuelle
        
        Args:
            text (str): Texte contenant potentiellement un JSON
            
        Returns:
            dict or None: Le JSON extrait
        """
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]\}))\}))*\}'
        json_match = re.search(json_pattern, text)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None
    
    def analyze_case(self, travel_class, age_category, luggages):
        """
        Analyse un cas complet en passant par toutes les étapes du pipeline.
        
        Args:
            travel_class (str): Classe de voyage (Economy, Business, etc.)
            age_category (str): Catégorie d'âge (Adult, Child, Infant)
            luggages (str): Description détaillée des bagages
            
        Returns:
            dict: Résultat complet de l'analyse de conformité ou None en cas d'échec
        """
        print("\n=== ÉTAPE 1: Analyse des bagages à main ===")
        carry_on_input = {
            "policy": self.policy,
            "travel_class": travel_class,
            "age_category": age_category,
            "luggages": luggages
        }
        
        # Étape 1: Analyser les bagages à main
        carry_on_result = self._call_model(self.carry_on_prompt, carry_on_input)
        if not carry_on_result:
            print("Échec de l'analyse des bagages à main")
            return None
        
        print(f"Résultat bagages à main: {json.dumps(carry_on_result, indent=2)}")
        
        print("\n=== ÉTAPE 2: Analyse des bagages enregistrés ===")
        checked_input = {
            "policy": self.policy,
            "travel_class": travel_class,
            "age_category": age_category,
            "luggages": luggages,
            "carry_on_result": json.dumps(carry_on_result)
        }
        
        # Étape 2: Analyser les bagages enregistrés
        checked_result = self._call_model(self.checked_baggage_prompt, checked_input)
        if not checked_result:
            print("Échec de l'analyse des bagages enregistrés")
            return None
        
        print(f"Résultat bagages enregistrés: {json.dumps(checked_result, indent=2)}")
        
        print("\n=== ÉTAPE 3: Identification des articles cargo ===")
        cargo_input = {
            "policy": self.policy,
            "travel_class": travel_class,
            "age_category": age_category,
            "luggages": luggages,
            "carry_on_result": json.dumps(carry_on_result),
            "checked_result": json.dumps(checked_result)
        }
        
        # Étape 3: Identifier les articles cargo
        cargo_result = self._call_model(self.cargo_items_prompt, cargo_input)
        if not cargo_result:
            print("Échec de l'identification des articles cargo")
            # On peut continuer avec une liste vide d'articles cargo
            cargo_result = {"cargo_items": []}
        
        print(f"Résultat articles cargo: {json.dumps(cargo_result, indent=2)}")
        
        print("\n=== ÉTAPE 4: Analyse finale et calcul des frais ===")
        final_input = {
            "policy": self.policy,
            "travel_class": travel_class,
            "age_category": age_category,
            "luggages": luggages,
            "carry_on_result": json.dumps(carry_on_result),
            "checked_result": json.dumps(checked_result),
            "cargo_result": json.dumps(cargo_result)
        }
        
        # Étape 4: Effectuer l'analyse finale
        final_result = self._call_model(self.final_analysis_prompt, final_input)
        if not final_result:
            print("Échec de l'analyse finale")
            return None
        
        print(f"Résultat final: {json.dumps(final_result, indent=2)}")
        
        # Assurer que tous les champs nécessaires sont présents
        if "compliance_result" not in final_result:
            final_result["compliance_result"] = False
        if "compliance_message" not in final_result:
            final_result["compliance_message"] = "Analyse incomplète"
        if "cargo_items" not in final_result:
            final_result["cargo_items"] = cargo_result.get("cargo_items", [])
        if "fees" not in final_result:
            final_result["fees"] = 0
        
        # Convertir les types si nécessaire
        final_result["compliance_result"] = bool(final_result["compliance_result"])
        if isinstance(final_result["fees"], str):
            try:
                final_result["fees"] = float(final_result["fees"].replace(',', ''))
            except ValueError:
                final_result["fees"] = 0
        
        return final_result

# Fonction principale pour exécuter l'analyse du dataset
def main():
    # Charger la politique de bagages
    try:
        with open('policy-corpus/luggage/luggage_policy.txt', 'r') as file:
            luggage_policy = file.read()
    except FileNotFoundError:
        print("Erreur: Fichier de politique de bagages non trouvé.")
        exit(1)

    # Charger le dataset de test
    try:
        df = pd.read_csv('policy-corpus/luggage/luggage_compliance/luggage_policy_test_dataset_100.csv')
    except FileNotFoundError:
        print("Erreur: Dataset de test non trouvé.")
        exit(1)
    
    df = df.sample(n=100, random_state=42)
    
    # Initialiser le modèle
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": "YOUR_WATSONX_API_KEY"
    }

    client = APIClient(credentials)
    project_id = "a31eb1db-e559-4a08-b0fa-638fdc608777"

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.1,
        GenParams.TOP_K: 10,
        GenParams.TOP_P: 0.9
    }

    llm = WatsonxLLM(
        model_id="mistralai/mistral-large",
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id,
        params=parameters
    )

    # Initialiser le pipeline d'analyse
    pipeline = RAGEnhancedLuggageAnalysisPipeline(luggage_policy, llm, project_id)

    # Créer un répertoire pour stocker les résultats intermédiaires
    if not os.path.exists("pipeline_results"):
        os.makedirs("pipeline_results")

    # Traiter chaque entrée du dataset
    results = []

    for index, row in df.iterrows():
        print(f"\n\n{'=' * 80}")
        print(f"TRAITEMENT DU CAS {index + 1}/{len(df)}")
        print(f"{'=' * 80}")
        print(f"• Travel Class: {row['travel_class']}")
        print(f"• Age Category: {row['age_category']}")
        print(f"• Luggages: {row['luggages']}")
        
        # Analyse complète du cas avec le pipeline
        try:
            predicted_result = pipeline.analyze_case(
                travel_class=row['travel_class'],
                age_category=row['age_category'],
                luggages=row['luggages']
            )
            
            # Si l'analyse réussit, sauvegarder le résultat et l'ajouter aux résultats
            if predicted_result:
                # Sauvegarder le résultat intermédiaire
                with open(f"pipeline_results/result_{index + 1}.json", "w") as f:
                    json.dump(predicted_result, f, indent=2)
                
                # Ajouter aux résultats
                results.append({
                    'predicted': predicted_result,
                    'actual': {
                        'compliance_result': bool(row['compliance_result']),
                        'compliance_message': row['compliance_message'],
                        'cargo_items': row['cargo_items'] if pd.notna(row['cargo_items']) else [],
                        'fees': row['fees']
                    }
                })
            else:
                print(f"L'analyse du cas {index + 1} n'a pas produit de résultat valide, ce cas sera ignoré.")
            
        except Exception as e:
            print(f"Erreur lors de l'analyse du cas {index + 1}: {str(e)}")
            # On ne fait rien de plus, on passe simplement au cas suivant

    # Calculer et afficher les métriques finales seulement si on a des résultats
    if results:
        final_metrics = calculate_metrics(results, "ensemble du dataset")

        print("\n" + "="*80)
        print(" RÉSULTATS FINAUX ".center(80, "="))
        print("="*80)
        print_metrics_report(final_metrics)

        # Sauvegarder les résultats détaillés et les métriques dans un fichier
        output_data = {
            "detailed_results": results,
            "metrics": {
                "final": final_metrics
            }
        }

        with open('luggage_analysis_pipeline_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nRésultats détaillés et métriques sauvegardés dans 'luggage_analysis_pipeline_results.json'")

        # Générer un graphique simple des résultats finaux
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mtick
            
            # Création du graphique des métriques
            plt.figure(figsize=(10, 6))
            
            # Données pour le graphique
            metrics_names = ['Conformité', 'Frais', 'Global', 'F1 (x100)']
            metrics_values = [
                final_metrics['compliance_accuracy'],
                final_metrics['fees_accuracy'],
                final_metrics['global_efficiency'],
                final_metrics['f1_score'] * 100  # Multiplier par 100 pour l'affichage
            ]
            
            # Créer le graphique
            bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.ylabel('Pourcentage (%)')
            plt.title('Métriques de performance sur l\'ensemble du dataset (Approche Pipeline)')
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.ylim(0, 110)  # Limiter l'axe Y à 110% pour la lisibilité
            
            # Sauvegarder le graphique
            plt.tight_layout()
            plt.savefig('luggage_metrics_pipeline_summary.png', dpi=300, bbox_inches='tight')
            print("\nGraphique récapitulatif des métriques sauvegardé dans 'luggage_metrics_pipeline_summary.png'")
            
        except ImportError:
            print("Note: matplotlib n'est pas installé. Aucun graphique n'a été généré.")
        except Exception as e:
            print(f"Erreur lors de la génération des graphiques: {str(e)}")
    else:
        print("\nAucun résultat valide n'a été obtenu. Impossible de calculer les métriques.")
        
if __name__ == "__main__":
    main()