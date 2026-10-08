import os
import json
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
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

class CustomerEmbedder:
    def __init__(self):
        # Usamos el modelo de embeddings
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Configurar un cliente ChromaDB para almacenar los embeddings de clientes
        self.chroma_client = chromadb.PersistentClient(path="./customer_embeddings")
        self.collection = self.chroma_client.get_or_create_collection(name="customer_data")
    
    def add_customer(self, customer_data):
        """
        Añadir un cliente a la base de datos de embeddings.
        
        Args:
            customer_data (dict): Datos del cliente, como clase de viaje, categoría de edad, detalles de equipaje, etc.
        """
        # Convertir los datos del cliente a una cadena representativa
        customer_str = json.dumps(customer_data)
        
        # Obtener el embedding del cliente
        embedding = self.embedding_model.encode([customer_str])[0]
        
        # Almacenar el cliente y su embedding
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[customer_str],
            ids=[f"customer_{len(self.collection.get()['ids']) + 1}"]
        )

    def find_similar_customers(self, customer_data, top_k=3):
        """
        Encontrar clientes similares utilizando embeddings.
        
        Args:
            customer_data (dict): Datos del nuevo cliente para comparar.
            top_k (int): Número de clientes más similares a retornar.
            
        Returns:
            list: Los documentos (clientes) más similares.
        """
        # Convertir los datos del cliente en una cadena representativa y calcular su embedding
        customer_str = json.dumps(customer_data)
        customer_embedding = self.embedding_model.encode([customer_str]).tolist()[0]
        
        # Buscar los clientes más similares en la base de datos
        results = self.collection.query(
            query_embeddings=[customer_embedding],
            n_results=top_k
        )
        
        return results['documents']

def extract_json_from_text(text):
    """Extrait le JSON d'une réponse textuelle qui pourrait contenir d'autres éléments."""
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]\}))\}))*\}'
    json_match = re.search(json_pattern, text)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

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

# Charger la politique de bagages
try:
    with open('policy-corpus/luggage/luggage_policy.txt', 'r') as file:
        luggage_policy = file.read()
except FileNotFoundError:
    try:
        # Essayer avec le chemin fourni dans le code original
        with open('mon_projet_ibm/policy-corpus/luggage/luggage_policy.txt', 'r') as file:
            luggage_policy = file.read()
    except FileNotFoundError:
        print("Erreur: Fichier de politique de bagages non trouvé.")
        exit(1)

# Charger le dataset de test
try:
    df = pd.read_csv('policy-corpus/luggage/luggage_compliance/luggage_policy_test_dataset_100.csv')
    
except FileNotFoundError:
    try:
        # Essayer avec un autre nom possible
        df = pd.read_csv('policy-corpus/luggage/luggage_compliance/luggage_policy_test_dataset_100.csv')
    except FileNotFoundError:
        print("Erreur: Dataset de test non trouvé.")
        exit(1)
# Sélectionner 100 lignes aléatoires sans répétition
df = df.sample(n=100, random_state=42)  # random_state est utilisé pour garantir la reproductibilité

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

# Créer un prompt avec few-shot learning
luggage_prompt_with_fewshot = PromptTemplate(
    input_variables=["policy", "travel_class", "age_category", "luggages"],
    template="""You are a luggage compliance expert. Review the following airline luggage policy:

{policy}

I'll provide you with examples of how to analyze luggage compliance before you analyze a new case.

EXAMPLE 1:
Travel Class: Economy
Age Category: Adult
Luggage Details: One carry-on backpack (50x35x20cm, 6kg) and one checked suitcase (150x60x30cm, 22kg)

REASONING:
1. Carry-on allowance for Economy: 1 bag + 1 personal item, max 7kg combined
   - One backpack: 6kg, dimensions within limits ✓
   - Complies with carry-on allowance

2. Checked baggage allowance for Economy: 1 piece, max 23kg
   - One suitcase: 22kg, dimensions within limits ✓
   - Complies with checked baggage allowance

3. No excess fees applicable

JSON RESPONSE:
{{
  "compliance_result": true,
  "compliance_message": "All baggage complies with policy limits",
  "cargo_items": [],
  "fees": 0
}}

EXAMPLE 2:
Travel Class: Economy
Age Category: Adult
Luggage Details: One carry-on suitcase (55x40x25cm, 8kg) and two checked bags (150x50x40cm, 25kg each)

REASONING:
1. Carry-on allowance for Economy: 1 bag + 1 personal item, max 7kg combined
   - One suitcase: 8kg ✗ (exceeds weight limit by 1kg)
   - Dimensions: Height exceeds limit by 2cm ✗

2. Checked baggage allowance for Economy: 1 piece, max 23kg
   - First bag: 25kg ✗ (overweight by 2kg)
   - Second bag: 25kg ✗ (extra piece AND overweight by 2kg)
   - Fees: $75 (first overweight bag) + $150 (extra piece) + $75 (second overweight bag) = $300

3. All bags are within size limits for checked baggage

JSON RESPONSE:
{{
  "compliance_result": false,
  "compliance_message": "Carry-on exceeds weight and size limits, both checked bags overweight, one extra checked bag",
  "cargo_items": [],
  "fees": 300
}}

EXAMPLE 3:
Travel Class: Business
Age Category: Child
Luggage Details: Two carry-on bags (both 55x40x23cm, 5kg each) and three checked bags (all 155x50x30cm, 30kg each)

REASONING:
1. Carry-on allowance for Business: 2 bags + 1 personal item, max 12kg combined
   - Two bags: combined 10kg ✓
   - Dimensions within limits ✓

2. Checked baggage allowance for Business: 2 pieces, each max 32kg for Child (same as adult)
   - First bag: 30kg ✓
   - Second bag: 30kg ✓
   - Third bag: 30kg ✗ (extra piece beyond allowance)
   - Fees: $150 (for the extra piece)

3. All bags are within size limits

JSON RESPONSE:
{{
  "compliance_result": false,
  "compliance_message": "One extra checked bag beyond Business Class allowance",
  "cargo_items": [],
  "fees": 150
}}

Now, analyze the following case:
•⁠  ⁠Travel Class: {travel_class}
•⁠  ⁠Age Category: {age_category}
•⁠  ⁠Luggage Details: {luggages}

IMPORTANT: You must respond ONLY with a valid JSON object in the following format without any additional text, explanations, or markdown:
{{
"compliance_result": true/false,
"compliance_message": "Brief explanation of compliance status",
"cargo_items": [],
"fees": 0
}}

Do not include any explanations, notes, or analysis outside of the JSON object. The cargo_items should be an array of strings, even if empty. The fees should be a number. Keep your compliance_message brief and concise."""
)

# Créer la chaîne de traitement avec le nouveau prompt
luggage_chain = luggage_prompt_with_fewshot | llm

# Traiter chaque entrée du dataset
results = []

# Créer un répertoire pour stocker les réponses brutes
if not os.path.exists("raw_responses_fewshot"):
    os.makedirs("raw_responses_fewshot")

for index, row in df.iterrows():
    input_data = {
        "policy": luggage_policy,
        "travel_class": row['travel_class'],
        "age_category": row['age_category'],
        "luggages": row['luggages']
    }
    
    for attempt in range(3):  # Essayer jusqu'à 3 fois
        try:
            print(f"\n--- Traitement de l'entrée {index + 1} ---")
            result = luggage_chain.invoke(input_data)
            
            # Enregistrer la réponse brute dans un fichier
            with open(f"raw_responses_fewshot/response_{index + 1}.txt", "w") as f:
                f.write(result)
            
            # Afficher le début de la réponse
            print(f"Réponse brute (début): {result[:200]}...")
            
            # Essayer de parser le JSON
            try:
                model_result = json.loads(result)
            except json.JSONDecodeError:
                # Si échec, essayer d'extraire le JSON du texte
                model_result = extract_json_from_text(result)
                if not model_result:
                    print(f"Échec de l'extraction JSON pour l'entrée {index + 1}")
                    continue
            
            # Vérifier que le résultat contient tous les champs requis
            required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
            if all(field in model_result for field in required_fields):
                # Convertir fees en nombre si c'est une chaîne
                if isinstance(model_result["fees"], str):
                    try:
                        model_result["fees"] = float(model_result["fees"].replace(',', ''))
                    except ValueError:
                        print(f"Erreur de conversion de fees en nombre pour l'entrée {index + 1}")
                        continue
                
                # S'assurer que cargo_items est une liste
                if not isinstance(model_result["cargo_items"], list):
                    model_result["cargo_items"] = []
                
                # Assurer que compliance_result est un booléen
                model_result["compliance_result"] = bool(model_result["compliance_result"])
                
                results.append({
                    'predicted': model_result,
                    'actual': {
                        'compliance_result': bool(row['compliance_result']),
                        'compliance_message': row['compliance_message'],
                        'cargo_items': row['cargo_items'] if pd.notna(row['cargo_items']) else [],
                        'fees': row['fees']
                    }
                })
                break
            else:
                print(f"Résultat incomplet pour l'entrée {index + 1}: {model_result}")
                continue
                
        except Exception as e:
            print(f"Tentative {attempt + 1} échouée pour l'entrée {index + 1}: {str(e)}")
            time.sleep(2)  # Attendre avant de réessayer
    
    # Si nous n'avons pas pu obtenir un résultat après plusieurs tentatives,
    # nous passons simplement à la prochaine entrée sans ajouter de valeur par défaut

# Calculer et afficher les métriques finales
if results:
    final_metrics = calculate_metrics(results, "ensemble du dataset")
    
    print("\n" + "="*80)
    print(" RÉSULTATS FINAUX ".center(80, "="))
    print("="*80)
    print_metrics_report(final_metrics)
else:
    print("\nAucun résultat valide obtenu.")

# Sauvegarder les résultats détaillés et les métriques dans un fichier
output_data = {
    "detailed_results": results
}

if results:
    output_data["metrics"] = {
        "final": final_metrics
    }

with open('luggage_analysis_fewshot_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nRésultats détaillés et métriques sauvegardés dans 'luggage_analysis_fewshot_results.json'")

# Générer un graphique simple des résultats (si matplotlib est disponible)
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    
    if results:
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
        plt.title('Métriques de performance sur l\'ensemble du dataset (Few-Shot Learning)')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.ylim(0, 110)  # Limiter l'axe Y à 110% pour la lisibilité
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig('luggage_metrics_fewshot_summary.png', dpi=300, bbox_inches='tight')
        print("\nGraphique récapitulatif des métriques sauvegardé dans 'luggage_metrics_fewshot_summary.png'")
        
except ImportError:
    print("Note: matplotlib n'est pas installé. Aucun graphique n'a été généré.")
except Exception as e:
    print(f"Erreur lors de la génération des graphiques: {str(e)}")