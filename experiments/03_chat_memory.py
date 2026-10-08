import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import time
import re
#from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM

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

class LuggageChatAnalyzer:
    """
    Classe pour analyser la conformité des bagages en utilisant un modèle ChatWatson
    qui maintient la politique et les instructions en mémoire.
    """
    
    def __init__(self, policy, project_id, model_id="mistralai/mistral-large"):
        """
        Initialise l'analyseur de bagages avec ChatWatson.
        
        Args:
            policy (str): Le texte de la politique de bagages
            project_id (str): ID du projet WatsonX
            model_id (str): ID du modèle à utiliser
        """
        self.policy = policy
        self.project_id = project_id
        self.model_id = model_id
        
        # Paramètres de génération de texte pour le chat
        self.parameters = {
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0.1,
            GenParams.TOP_K: 10,
            GenParams.TOP_P: 0.9,
            GenParams.STOP_SEQUENCES: ["Human:", "User:"]}
        
        # Initialiser le modèle de chat
        self.chat_model = WatsonxLLM(
            model_id=self.model_id,
            url="https://us-south.ml.cloud.ibm.com",
            apikey="YOUR_WATSONX_API_KEY",
            project_id=self.project_id,
            params=self.parameters
        )
        
        # Créer un système de message initial avec la politique et les instructions
        self.system_message = self._create_system_message()
        
        # Initialiser la session de chat
        self.messages = [{"role": "system", "content": self.system_message}]
        
        # Ajouter des exemples few-shot pour amorcer le modèle
        self._add_few_shot_examples()
        
        print("Session de chat initialisée avec système de message et exemples few-shot.")
    
    def _create_system_message(self):
        """Crée le message système initial avec la politique et les instructions."""
        return f"""Vous êtes un expert en conformité des bagages qui analyse si les bagages des passagers sont conformes à la politique de bagages suivante. Analysez chaque cas de bagage qui vous est présenté et répondez au format JSON demandé.

POLITIQUE DE BAGAGES:
{self.policy}

INSTRUCTIONS:
1. Analyser si les bagages décrits respectent les limites de nombre, poids et dimensions selon la classe de voyage et l'âge.
2. Déterminer les frais applicables pour les bagages non conformes.
3. Identifier les articles nécessitant un traitement cargo spécial.
4. Fournir une réponse structurée au format JSON UNIQUEMENT avec les champs suivants:
   - compliance_result: booléen (true/false) indiquant la conformité globale
   - compliance_message: explication brève du statut de conformité
   - cargo_items: liste des articles nécessitant un traitement cargo (liste vide si aucun)
   - fees: montant total des frais en nombre (sans symbole de devise)

Ne pas inclure d'explications, de notes ou d'analyse en dehors de l'objet JSON. 
cargo_items doit être un tableau de chaînes de caractères, même s'il est vide. 
fees doit être un nombre.
Votre compliance_message doit être bref et concis."""
    
    def _add_few_shot_examples(self):
        """Ajoute des exemples few-shot pour amorcer le modèle."""
        # Exemple 1 - Cas conforme
        self.messages.append({
            "role": "user",
            "content": "Analysez ce cas:\nTravel Class: Economy\nAge Category: Adult\nLuggage Details: One carry-on backpack (50x35x20cm, 6kg) and one checked suitcase (150x60x30cm, 22kg)"
        })
        
        self.messages.append({
            "role": "assistant",
            "content": """{"compliance_result": true, "compliance_message": "All baggage complies with policy limits", "cargo_items": [], "fees": 0}"""
        })
        
        # Exemple 2 - Cas non conforme
        self.messages.append({
            "role": "user",
            "content": "Analysez ce cas:\nTravel Class: Economy\nAge Category: Adult\nLuggage Details: One carry-on suitcase (55x40x25cm, 8kg) and two checked bags (150x50x40cm, 25kg each)"
        })
        
        self.messages.append({
            "role": "assistant",
            "content": """{"compliance_result": false, "compliance_message": "Carry-on exceeds weight and size limits, both checked bags overweight, one extra checked bag", "cargo_items": [], "fees": 300}"""
        })
        
        # Exemple 3 - Cas partiellement conforme
        self.messages.append({
            "role": "user",
            "content": "Analysez ce cas:\nTravel Class: Business\nAge Category: Child\nLuggage Details: Two carry-on bags (both 55x40x23cm, 5kg each) and three checked bags (all 155x50x30cm, 30kg each)"
        })
        
        self.messages.append({
            "role": "assistant",
            "content": """{"compliance_result": false, "compliance_message": "One extra checked bag beyond Business Class allowance", "cargo_items": [], "fees": 150}"""
        })
        
        print(f"Ajout de {len(self.messages) - 1} messages d'exemple.")
    
    def analyze_case(self, travel_class, age_category, luggages):
        """
        Analyse un cas de bagages en utilisant la session de chat.
        
        Args:
            travel_class (str): Classe de voyage (Economy, Business, etc.)
            age_category (str): Catégorie d'âge (Adult, Child, Infant)
            luggages (str): Description détaillée des bagages
            
        Returns:
            dict: Résultat d'analyse avec compliance_result, compliance_message, cargo_items et fees
            None: Si l'analyse échoue après toutes les tentatives
        """
        # Définir les champs requis dès le début de la fonction
        required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
        
        # Préparer le message utilisateur
        user_message = f"Analysez ce cas:\nTravel Class: {travel_class}\nAge Category: {age_category}\nLuggage Details: {luggages}"
        
        # Ajouter à l'historique des messages
        self.messages.append({"role": "user", "content": user_message})
        
        # Maximum de tentatives
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Obtenir la réponse du modèle
                # Le WatsonxLLM renvoie directement la réponse sous forme de chaîne
                model_response = self.chat_model.invoke(self.messages)
                
                # Ajouter la réponse à l'historique
                self.messages.append({"role": "assistant", "content": model_response})
                
                # Limiter l'historique pour éviter de dépasser les limites de tokens
                # On garde le système de message, les exemples few-shot et les 2 derniers échanges
                if len(self.messages) > 10:  # Système + 6 few-shot + 4 récents
                    # Conserver le message système et les 6 messages few-shot (3 paires)
                    few_shot_messages = self.messages[:7]
                    # Conserver les 4 derniers messages (2 paires question/réponse)
                    recent_messages = self.messages[-4:]
                    # Reconstruire la liste
                    self.messages = few_shot_messages + recent_messages
                    print("Historique des messages tronqué pour maintenir la taille du contexte.")
                
                # Essayer de parser le JSON
                try:
                    result = json.loads(model_response)
                    # Vérifier que tous les champs nécessaires sont présents
                    if all(field in result for field in required_fields):
                        # Standardiser les types
                        result["compliance_result"] = bool(result["compliance_result"])
                        if isinstance(result["fees"], str):
                            result["fees"] = float(result["fees"].replace(',', ''))
                        if not isinstance(result["cargo_items"], list):
                            result["cargo_items"] = []
                        
                        return result
                except json.JSONDecodeError:
                    # Si le JSON est invalide, essayer d'extraire un JSON valide du texte
                    extracted_json = extract_json_from_text(model_response)
                    if extracted_json and all(field in extracted_json for field in required_fields):
                        # Standardiser les types
                        extracted_json["compliance_result"] = bool(extracted_json["compliance_result"])
                        if isinstance(extracted_json["fees"], str):
                            extracted_json["fees"] = float(extracted_json["fees"].replace(',', ''))
                        if not isinstance(extracted_json["cargo_items"], list):
                            extracted_json["cargo_items"] = []
                        
                        return extracted_json
                
                # Si on arrive ici, la réponse n'est pas exploitable
                print(f"Tentative {attempt + 1}: Réponse mal formatée. Nouvelle tentative...")
                # Supprimer la dernière réponse pour réessayer
                self.messages.pop()
                
            except Exception as e:
                print(f"Erreur lors de la tentative {attempt + 1}: {str(e)}")
            
            # Attendre avant de réessayer
            if attempt < max_attempts - 1:
                time.sleep(2)
        
        # Si toutes les tentatives échouent, retourner None au lieu d'un résultat par défaut
        print("Toutes les tentatives ont échoué. Aucun résultat valide n'a été obtenu.")
        return None
    
    def reset_chat(self):
        """Réinitialise la session de chat avec le message système et les exemples few-shot."""
        self.messages = [{"role": "system", "content": self.system_message}]
        self._add_few_shot_examples()
        print("Session de chat réinitialisée.")

def main():
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
        df = pd.read_csv('policy-corpus/luggage/luggage_compliance/luggage_policy_test_dataset_1K.csv')
    except FileNotFoundError:
        try:
            # Essayer avec un autre nom possible
            df = pd.read_csv('luggage_policy_test_dataset_100.csv')
        except FileNotFoundError:
            print("Erreur: Dataset de test non trouvé.")
            exit(1)
    df = df.sample(n=100, random_state=42)
    
    # Initialiser le client API
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": "YOUR_WATSONX_API_KEY"
    }

    project_id = "a31eb1db-e559-4a08-b0fa-638fdc608777"

    # Initialiser l'analyseur de chat pour les bagages
    chat_analyzer = LuggageChatAnalyzer(
        policy=luggage_policy,
        project_id=project_id,
        model_id="mistralai/mistral-large"
    )

    # Créer un répertoire pour stocker les réponses brutes
    if not os.path.exists("chat_responses"):
        os.makedirs("chat_responses")

    # Traiter chaque entrée du dataset
    results = []
    
    # Pour suivre l'efficacité au fil du temps - ajusté aux quartiles du dataset
    dataset_size = len(df)
    segment_checkpoints = [
        dataset_size // 4,                  # Premier quartile (25%)
        dataset_size // 2,                  # Médiane (50%)
        (dataset_size * 3) // 4,            # Troisième quartile (75%)
        dataset_size                        # Dataset complet (100%)
    ]
    print(f"Points de contrôle ajustés: {segment_checkpoints} sur un total de {dataset_size} lignes")
    segment_results = {}
    
    # Pour suivre l'évolution du F1 score
    f1_evolution = []
    successful_cases = 0

    for index, row in df.iterrows():
        print(f"\n--- Traitement de l'entrée {index + 1}/{len(df)} ---")
        print(f"• Travel Class: {row['travel_class']}")
        print(f"• Age Category: {row['age_category']}")
        print(f"• Luggages: {row['luggages']}")
        
        # Analyser le cas avec ChatWatson
        predicted_result = chat_analyzer.analyze_case(
            travel_class=row['travel_class'],
            age_category=row['age_category'],
            luggages=row['luggages']
        )
        
        # Si l'analyse a échoué, sauter ce cas et passer au suivant
        if predicted_result is None:
            print(f"Échec de l'analyse pour le cas {index + 1}. Ce cas sera ignoré.")
            continue
        
        # Enregistrer la réponse brute
        with open(f"chat_responses/response_{index + 1}.json", "w") as f:
            json.dump(predicted_result, f, indent=2)
        
        print(f"Résultat prédit: {json.dumps(predicted_result, indent=2)}")
        
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
        
        # Incrémenter le compteur de cas réussis
        successful_cases += 1
        
        # Calculer et ajouter le F1 score pour chaque point
        if successful_cases > 1:  # F1 nécessite au moins 2 exemples
            actual_compliance = [bool(r['actual']['compliance_result']) for r in results]
            predicted_compliance = [bool(r['predicted']['compliance_result']) for r in results]
            current_f1 = f1_score(actual_compliance, predicted_compliance, average='binary')
            
            f1_evolution.append({
                'cases': successful_cases,
                'f1_score': current_f1
            })
        
        # Calculer les métriques aux points de contrôle
        current_count = successful_cases
        
        if current_count in segment_checkpoints or index + 1 == len(df):
            segment_name = f"premiers {current_count} cas réussis"
            segment_metrics = calculate_metrics(results[:current_count], segment_name)
            segment_results[f"quartile_{len(segment_results) + 1}"] = segment_metrics
            
            print("\n" + "="*50)
            print(f"POINT DE CONTRÔLE: {current_count} cas traités avec succès ({index + 1} tentés)")
            print("="*50)
            print_metrics_report(segment_metrics)
            
            # Réinitialiser la session de chat tous les 25 cas pour éviter l'accumulation excessive
            if current_count % 25 == 0:
                chat_analyzer.reset_chat()
                print("Session de chat réinitialisée pour maintenir les performances.")

    # Vérifier s'il y a des résultats à analyser
    if not results:
        print("Aucun cas n'a pu être analysé avec succès. Impossible de générer des rapports.")
        exit(1)

    # Calculer et afficher les métriques finales
    final_metrics = calculate_metrics(results, "ensemble du dataset")

    print("\n" + "="*80)
    print(" RÉSULTATS FINAUX ".center(80, "="))
    print("="*80)
    print_metrics_report(final_metrics)

    # Afficher un résumé de l'évolution des métriques clés
    print("\n" + "="*80)
    print(" ÉVOLUTION DES MÉTRIQUES PAR QUARTILE ".center(80, "="))
    print("="*80)

    metric_names = ["Efficacité globale", "Précision de conformité", "Précision des frais", "Score F1", "Erreur moyenne des frais"]
    headers = ["Métrique"] + [f"Quartile {i}" for i in range(1, len(segment_results) + 1)]

    # Préparer les données pour le tableau d'évolution
    evolution_table = []
    for i, name in enumerate(metric_names):
        row = [name]
        for q in range(1, len(segment_results) + 1):
            key = f"quartile_{q}"
            if key in segment_results:
                metrics = segment_results[key]
                if i == 0:  # Efficacité globale
                    value = f"{metrics['global_efficiency']:.2f}%"
                elif i == 1:  # Précision de conformité
                    value = f"{metrics['compliance_accuracy']:.2f}%"
                elif i == 2:  # Précision des frais
                    value = f"{metrics['fees_accuracy']:.2f}%"
                elif i == 3:  # Score F1
                    value = f"{metrics['f1_score']:.4f}"
                else:  # Erreur moyenne des frais
                    value = f"{metrics['mean_fee_error']:.2f}€"
            else:
                value = "N/A"
            row.append(value)
        evolution_table.append(row)

    # Afficher le tableau d'évolution
    col_widths = [25] + [20] * len(segment_results)
    row_format = "| " + " | ".join(["{:<" + str(width) + "}" for width in col_widths]) + " |"

    # En-tête du tableau
    print("+" + "+".join(["-" * (width + 2) for width in col_widths]) + "+")
    print(row_format.format(*headers))
    print("+" + "+".join(["=" * (width + 2) for width in col_widths]) + "+")

    # Corps du tableau
    for row in evolution_table:
        print(row_format.format(*row))
        print("+" + "+".join(["-" * (width + 2) for width in col_widths]) + "+")

    # Sauvegarder les résultats détaillés et les métriques dans un fichier
    output_data = {
        "detailed_results": results,
        "metrics": {
            "final": final_metrics,
            "segments": segment_results,
            "f1_evolution": f1_evolution
        }
    }

    with open('luggage_analysis_chat_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nRésultats détaillés et métriques sauvegardés dans 'luggage_analysis_chat_results.json'")

    # Générer des graphiques si matplotlib est disponible
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        
        # Graphique du F1 score au fil des cas
        plt.figure(figsize=(12, 6))
        
        # Extraire les données pour le graphique du F1 score
        case_counts = [item['cases'] for item in f1_evolution]
        f1_scores = [item['f1_score'] for item in f1_evolution]
        
        # Tracer la courbe du F1 score
        plt.plot(case_counts, f1_scores, 'o-', color='#d62728', linewidth=2, markersize=4)
        
        # Ajouter une ligne de tendance
        z = np.polyfit(case_counts, f1_scores, 2)
        p = np.poly1d(z)
        trend_x = np.linspace(min(case_counts), max(case_counts), 100)
        plt.plot(trend_x, p(trend_x), '--', color='#7f7f7f', linewidth=1.5, 
                 label=f'Tendance (degré 2)')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Nombre de cas traités avec succès')
        plt.ylabel('Score F1')
        plt.title('Évolution du F1 score au fil des cas')
        plt.legend(loc='best')
        
        # Ajouter des annotations pour quelques points clés
        if len(case_counts) > 5:
            # Annoter le premier et le dernier point
            plt.annotate(f'{f1_scores[0]:.4f}', (case_counts[0], f1_scores[0]), 
                       textcoords="offset points", xytext=(0, 10), ha='center')
            plt.annotate(f'{f1_scores[-1]:.4f}', (case_counts[-1], f1_scores[-1]), 
                       textcoords="offset points", xytext=(0, 10), ha='center')
            
            # Trouver le maximum et le minimum du F1 score
            max_f1_idx = f1_scores.index(max(f1_scores))
            min_f1_idx = f1_scores.index(min(f1_scores))
            
            # Annoter le maximum et le minimum
            if max_f1_idx != 0 and max_f1_idx != len(f1_scores) - 1:
                plt.annotate(f'{f1_scores[max_f1_idx]:.4f}', 
                           (case_counts[max_f1_idx], f1_scores[max_f1_idx]), 
                           textcoords="offset points", xytext=(0, 10), ha='center')
            
            if min_f1_idx != 0 and min_f1_idx != len(f1_scores) - 1:
                plt.annotate(f'{f1_scores[min_f1_idx]:.4f}', 
                           (case_counts[min_f1_idx], f1_scores[min_f1_idx]), 
                           textcoords="offset points", xytext=(0, -15), ha='center')
        
        plt.tight_layout()
        plt.savefig('f1_score_evolution.png', dpi=300, bbox_inches='tight')
        print("Graphique d'évolution du F1 score sauvegardé dans 'f1_score_evolution.png'")
        
        # Préparation des données pour les graphiques de quartiles
        quartile_labels = []
        checkpoints = []
        efficiencies = []
        compliance_accuracies = []
        fees_accuracies = []
        f1_scores_quartile = []
        
        # Récupérer les données des quartiles pour les graphiques
        for i in range(1, len(segment_results) + 1):
            key = f"quartile_{i}"
            if key in segment_results:
                metrics = segment_results[key]
                # Déterminer le label du quartile
                if i == len(segment_results):
                    quartile_label = "100%"
                else:
                    quartile_label = f"Q{i}"
                
                quartile_labels.append(quartile_label)
                checkpoints.append(metrics['total_cases'])
                efficiencies.append(metrics['global_efficiency'])
                compliance_accuracies.append(metrics['compliance_accuracy'])
                fees_accuracies.append(metrics['fees_accuracy'])
                f1_scores_quartile.append(metrics['f1_score'] * 100)  # Convertir en pourcentage pour l'affichage
        
        # Création du graphique principal
        plt.figure(figsize=(12, 8))
        
        # Graphique d'évolution des métriques
        plt.subplot(211)
        plt.plot(checkpoints, efficiencies, 'o-', label='Efficacité globale', linewidth=2, color='#1f77b4')
        plt.plot(checkpoints, compliance_accuracies, 's-', label='Précision de conformité', linewidth=2, color='#ff7f0e')
        plt.plot(checkpoints, fees_accuracies, '^-', label='Précision des frais', linewidth=2, color='#2ca02c')
        plt.plot(checkpoints, f1_scores_quartile, 'D-', label='Score F1 (×100)', linewidth=2, color='#d62728')
        
        # Ajouter une grille et des étiquettes pour chaque point
        plt.grid(True, linestyle='--', alpha=0.7)
        for i, txt in enumerate(quartile_labels):
            plt.annotate(txt, (checkpoints[i], efficiencies[i]), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
        
        # Configuration des axes
        plt.xlabel('Nombre de cas traités avec succès')
        plt.ylabel('Pourcentage (%)')
        plt.title('Évolution des métriques de performance par quartile (Approche Chat)')
        plt.legend(loc='best')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Ajouter un deuxième graphique pour l'erreur moyenne des frais
        plt.subplot(212)
        fee_errors = [segment_results[f"quartile_{i+1}"]['mean_fee_error'] for i in range(len(quartile_labels))]
        plt.bar(quartile_labels, fee_errors, color='#9467bd')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xlabel('Portion du dataset')
        plt.ylabel('Erreur moyenne (€)')
        plt.title('Évolution de l\'erreur moyenne des frais (Approche Chat)')
        
        # Ajouter les valeurs sur chaque barre
        for i, v in enumerate(fee_errors):
            plt.text(i, v + 5, f"{v:.2f}€", ha='center', fontweight='bold')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig('luggage_metrics_chat_evolution.png', dpi=300, bbox_inches='tight')
        print("\nGraphique d'évolution des métriques sauvegardé dans 'luggage_metrics_chat_evolution.png'")
        
        # Graphique combinant F1 score et points de quartile
        plt.figure(figsize=(12, 6))
        
        # Ajouter la courbe continue du F1 score
        plt.plot(case_counts, f1_scores, 'o-', color='#d62728', linewidth=1.5, alpha=0.6, 
                 label='F1 score (évolution continue)')
        
        # Ajouter les points des quartiles pour le F1 score
        plt.plot(checkpoints, [score / 100 for score in f1_scores_quartile], 'D-', 
                 color='#d62728', linewidth=2.5, label='F1 score (quartiles)', markersize=10)
        
        # Ajouter une grille
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ajouter des annotations pour les quartiles
        for i, (x, y) in enumerate(zip(checkpoints, [score / 100 for score in f1_scores_quartile])):
            plt.annotate(f'Q{i+1}: {y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontweight='bold')
        
        plt.xlabel('Nombre de cas traités avec succès')
        plt.ylabel('Score F1')
        plt.title('Évolution détaillée du F1 score avec points de quartile')
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('f1_score_detailed_evolution.png', dpi=300, bbox_inches='tight')
        print("Graphique détaillé du F1 score sauvegardé dans 'f1_score_detailed_evolution.png'")
        
        # Graphique supplémentaire pour les types d'erreurs
        plt.figure(figsize=(14, 8))
        
        # Collecter les types d'erreurs les plus fréquents pour chaque quartile
        error_data = {}
        
        for i in range(1, len(segment_results) + 1):
            key = f"quartile_{i}"
            if key in segment_results:
                metrics = segment_results[key]
                # Prendre les 5 premiers types d'erreurs
                top_errors = metrics['error_types'][:5] if len(metrics['error_types']) > 5 else metrics['error_types']
                
                for error in top_errors:
                    error_type = error['type']
                    if error_type not in error_data:
                        error_data[error_type] = [0] * len(segment_results)  # Un pour chaque quartile
                    error_data[error_type][i-1] = error['percentage']
        
        # Filtrer pour ne garder que les 6 types d'erreurs les plus fréquents au total
        error_sums = {k: sum(v) for k, v in error_data.items()}
        top_errors = sorted(error_data.keys(), key=lambda x: error_sums[x], reverse=True)[:6]
        
        # Créer le graphique d'évolution des erreurs
        for error_type in top_errors:
            plt.plot(quartile_labels, error_data[error_type], 'o-', label=error_type, linewidth=2)
        
        plt.xlabel('Portion du dataset')
        plt.ylabel('Pourcentage des cas (%)')
        plt.title('Évolution des types d\'erreurs principaux par quartile (Approche Chat)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Sauvegarder le graphique
        plt.savefig('luggage_error_chat_evolution.png', dpi=300, bbox_inches='tight')
        print("Graphique d'évolution des types d'erreurs sauvegardé dans 'luggage_error_chat_evolution.png'")
    except ImportError:
        print("Note: matplotlib n'est pas installé. Aucun graphique n'a été généré.")
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques: {str(e)}")
        
    total_cases_attempted = len(df)
    total_cases_successful = len(results)
    success_rate = (total_cases_successful / total_cases_attempted) * 100

    print("\n" + "="*80)
    print(" ANALYSE DU TAUX DE RÉUSSITE ".center(80, "="))
    print("="*80)
    print(f"Total des cas tentés: {total_cases_attempted}")
    print(f"Total des cas analysés avec succès: {total_cases_successful}")
    print(f"Taux de réussite d'analyse: {success_rate:.2f}%")
    print(f"Cas échoués: {total_cases_attempted - total_cases_successful} ({100 - success_rate:.2f}%)")
    print("="*80)

    print("\nRemarque: Les cas non analysés avec succès ont été exclus des métriques.")
    print("L'élimination des valeurs par défaut en cas d'échec permet de mesurer plus précisément")
    print("les performances réelles du modèle sur les cas qu'il est capable de traiter.")

if __name__ == "__main__":
    main()