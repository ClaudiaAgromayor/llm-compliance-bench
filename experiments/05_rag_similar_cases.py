import os
import json
import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
import time
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from sentence_transformers import SentenceTransformer

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

# Load local file
def load_local_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading file from {file_path}: {e}")
        return None

#Enhanced JSON extraction with better error handling and normalized cargo items to ensure consistent formatting 
def extract_json_from_text(text):
    text = re.sub(r'```json|```', '', text)
    
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            possible_json = text[start:end+1]
            return json.loads(possible_json)
    except json.JSONDecodeError:
        pass
    
    # If direct parsing fails, try regex pattern matching
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    json_matches = re.findall(json_pattern, text)
    
    if json_matches:
        for json_str in sorted(json_matches, key=len, reverse=True):
            json_str = json_str.strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    try: # If still no valid JSON, try more aggressive parsing
        compliance_result = re.search(r'"compliance_result":\s*(true|false)', text, re.IGNORECASE)
        compliance_msg = re.search(r'"compliance_message":\s*"([^"]+)"', text)
        
        if compliance_result:
            result = {}
            result["compliance_result"] = compliance_result.group(1).lower() == "true"
            result["compliance_message"] = compliance_msg.group(1) if compliance_msg else "Extracted from text"
            result["cargo_items"] = []
            result["fees"] = 0
            return result
    except:
        pass
    
    return "It was not possible to read JSON"

# Standardize and clean up result fields: ensure that all required fields exist
def standardize_result(model_result):
    required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
    for field in required_fields:
        if field not in model_result:
            model_result[field] = None if field == "compliance_message" else ([] if field == "cargo_items" else 0)
    
    if isinstance(model_result["compliance_result"], str): #checks if compliance_result is a string. if it is, it converts it to a boolean
        model_result["compliance_result"] = model_result["compliance_result"].lower() == "true"
    
    if not isinstance(model_result["compliance_message"], str): #checks if the compliance_message is a string
        model_result["compliance_message"] = str(model_result["compliance_message"] or "No message")
    
    model_result["cargo_items"] = normalize_cargo_items(model_result["cargo_items"]) 

    if isinstance(model_result["fees"], str): #if fees is string, it removers everything except numbers, dots and commas
        fees_str = re.sub(r'[^\d.,]', '', model_result["fees"])
        try:
            model_result["fees"] = float(fees_str.replace(',', '.'))
        except ValueError:
            model_result["fees"] = "Nan"
    elif not isinstance(model_result["fees"], (int, float)):
        model_result["fees"] = "Nan"
        
    return model_result

# Convert luggage data into a text representation for similarity comparison
def preprocess_luggage_data(row):
    text_repr = f"travel_class:{row['travel_class']} age_category:{row['age_category']} "
    
    if pd.notna(row['luggages']):
        try:
            luggage_items = json.loads(row['luggages'])
            for i, item in enumerate(luggage_items):
                text_repr += f"item{i+1}:{item['storage']}_"
                text_repr += f"w{item['weight']}_"
                text_repr += f"h{item['height']}_"
                text_repr += f"d{item['depth']}_"
                text_repr += f"special{item['special']}_"
        except:
            text_repr += "luggage_processing_error:not_possible_to_read_data"
    
    return text_repr.strip()

# Normalize cargo items and converts it to a consistent list format
def normalize_cargo_items(items):
    if not items:
        return []
    
    if isinstance(items, str):
        if items.strip() == "":
            return []
        try:
            parsed_items = json.loads(items)
            if isinstance(parsed_items, list):
                return parsed_items
            else:
                return [items]
        except:
            if "," in items:
                return [item.strip() for item in items.split(",")]
            else:
                return [items.strip()]
    
    if isinstance(items, list):
        return items
    return []

# =====================================================================
# RAG SYSTEM
# =====================================================================
# Vectorizes all customer cases using TF-IDF
# Finds similar cases based on travel class, age category, and luggage properties
class LuggagePolicyRAG:
    
    def __init__(self, dataset_path, top_k=15, use_sentence_transformers=False):
        self.top_k = top_k
        self.use_sentence_transformers = use_sentence_transformers
        self.load_dataset(dataset_path)
        self.vectorize_data()
    
    def load_dataset(self, dataset_path):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(dataset_path)
        self.df['enhanced_text'] = self.df.apply(self._enhance_case_representation, axis=1)
        print(f"Loaded {len(self.df)} cases")
        
    def _enhance_case_representation(self, row):
        """Create enhanced text representation for similarity comparison"""
        representation = [
            f"class_{row['travel_class']}",
            f"age_{row['age_category']}",  # Supprimer la ligne compliance_result
        ]
        
        # Process luggage items
        if pd.notna(row['luggages']):
            try:
                luggage = json.loads(row['luggages'])
                for item in luggage:
                    representation.append(
                        f"item_{item['storage']}_w{item['weight']}"
                    )
            except:
                representation.append("luggage_error")
    
        return ' '.join(representation)
    
    def vectorize_data(self):
        """Create vector representations"""
        if self.use_sentence_transformers:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.vectors = self.model.encode(self.df['enhanced_text'].tolist())
                print("Using SentenceTransformers for embeddings")
            except:
                self.use_sentence_transformers = False
                self._vectorize_with_tfidf()
        else:
            self._vectorize_with_tfidf()
            
    def _vectorize_with_tfidf(self):
        """Fallback to TF-IDF vectorization"""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.vectors = self.vectorizer.fit_transform(self.df['enhanced_text'])
        print("Using TF-IDF for vectorization")
        
    def get_similar_cases(self, query_row):
        """Version sans fuite : utilise seulement les features pour la similarité"""
        query_text = self._enhance_case_representation(query_row)  # Sans compliance_result
        
        if self.use_sentence_transformers:
            query_vec = self.model.encode([query_text])
            similarities = cosine_similarity(query_vec, self.vectors)[0]
        else:
            query_vec = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        return similarities.argsort()[-self.top_k:][::-1]  # Retourne seulement les indices

    def get_case_details(self, index):
        """Récupère TOUTES les infos d'un cas spécifique (après sélection)"""
        case = self.df.iloc[index]
        return {
            "travel_class": case['travel_class'],
            "age_category": case['age_category'],
            "luggages": case['luggages'],
            "compliance_result": case['compliance_result'],  # OK car demande explicite
            "compliance_message": case['compliance_message'],
            "fees": float(case['fees']) if pd.notna(case['fees']) else 0.0
        }
# =====================================================================
# ADVANCED METRICS FUNCTIONS
# =====================================================================

def calculate_enhanced_metrics(results):
    """
    Calculate enhanced metrics for model evaluation
    Returns detailed metrics dictionary
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to analyze"}
    
    # Extract actual and predicted values for compliance
    actual_compliance = [bool(r['actual']['compliance_result']) for r in results]
    predicted_compliance = [bool(r['predicted']['compliance_result']) for r in results]
    
    # Extract actual and predicted values for cargo items
    actual_cargo = [set(map(str, r['actual']['cargo_items'])) for r in results]
    predicted_cargo = [set(map(str, r['predicted']['cargo_items'])) for r in results]
    
    # Extract actual and predicted fees
    actual_fees = [float(r['actual']['fees']) for r in results]
    predicted_fees = [float(r['predicted']['fees']) for r in results]
    
    # Compliance metrics
    true_positives = sum(1 for i in range(total) if actual_compliance[i] and predicted_compliance[i])
    true_negatives = sum(1 for i in range(total) if not actual_compliance[i] and not predicted_compliance[i])
    false_positives = sum(1 for i in range(total) if not actual_compliance[i] and predicted_compliance[i])
    false_negatives = sum(1 for i in range(total) if actual_compliance[i] and not predicted_compliance[i])
    
    compliance_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    compliance_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    compliance_f1 = 2 * (compliance_precision * compliance_recall) / (compliance_precision + compliance_recall) if (compliance_precision + compliance_recall) > 0 else 0
    
    # Cargo items metrics (precision, recall, f1)
    cargo_tp = 0
    cargo_fp = 0
    cargo_fn = 0
    
    for act, pred in zip(actual_cargo, predicted_cargo):
        cargo_tp += len(act & pred)
        cargo_fp += len(pred - act)
        cargo_fn += len(act - pred)
    
    cargo_precision = cargo_tp / (cargo_tp + cargo_fp) if (cargo_tp + cargo_fp) > 0 else 0
    cargo_recall = cargo_tp / (cargo_tp + cargo_fn) if (cargo_tp + cargo_fn) > 0 else 0
    cargo_f1 = 2 * (cargo_precision * cargo_recall) / (cargo_precision + cargo_recall) if (cargo_precision + cargo_recall) > 0 else 0
    
    # Fees metrics
    fee_exact_match = sum(1 for i in range(total) if abs(predicted_fees[i] - actual_fees[i]) < 0.01)
    fee_errors = [abs(predicted_fees[i] - actual_fees[i]) for i in range(total)]
    mae_fees = sum(fee_errors) / total
    mse_fees = sum(e**2 for e in fee_errors) / total
    rmse_fees = mse_fees ** 0.5
    
    return {
        "compliance": {
            "precision": compliance_precision,
            "recall": compliance_recall,
            "f1_score": compliance_f1,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }
        },
        "cargo_items": {
            "precision": cargo_precision,
            "recall": cargo_recall,
            "f1_score": cargo_f1
        },
        "fees": {
            "exact_match": fee_exact_match / total * 100,
            "mae": mae_fees,
            "rmse": rmse_fees
        },
        "sample_size": total
    }

def print_metrics_summary(metrics):
    """
    Print a readable summary of the metrics
    """
    print("\n" + "="*50)
    print("COMPLIANCE METRICS")
    print("="*50)
    print(f"Precision: {metrics['compliance']['precision']:.4f}")
    print(f"Recall: {metrics['compliance']['recall']:.4f}")
    print(f"F1 Score: {metrics['compliance']['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['compliance']['confusion_matrix']
    print(f"  TP: {cm['true_positives']} | FP: {cm['false_positives']}")
    print(f"  FN: {cm['false_negatives']} | TN: {cm['true_negatives']}")
    
    print("\n" + "="*50)
    print("CARGO ITEMS METRICS")
    print("="*50)
    print(f"Precision: {metrics['cargo_items']['precision']:.4f}")
    print(f"Recall: {metrics['cargo_items']['recall']:.4f}")
    print(f"F1 Score: {metrics['cargo_items']['f1_score']:.4f}")
    
    print("\n" + "="*50)
    print("FEES ESTIMATION METRICS")
    print("="*50)
    print(f"Exact Match: {metrics['fees']['exact_match']:.2f}%")
    print(f"MAE: {metrics['fees']['mae']:.2f}")
    print(f"RMSE: {metrics['fees']['rmse']:.2f}")
    
    print(f"\nSample Size: {metrics['sample_size']} cases")

# =====================================================================
# GENERATE PREDICTIONS CSV
# =====================================================================

def generate_predictions_csv(results, df, output_file="predictions_results.csv"):
    """Generate CSV file with all predictions and actual values"""
    data = []
    
    for idx, case in enumerate(results):
        row_data = df.iloc[idx]
        
        # Convert lists to string representation for CSV
        actual_cargo = ', '.join(map(str, case['actual']['cargo_items']))
        predicted_cargo = ', '.join(map(str, case['predicted']['cargo_items']))
        
        data.append({
            'travel_class': row_data['travel_class'],
            'age_category': row_data['age_category'],
            'luggages': row_data['luggages'],
            'compliance_result': case['actual']['compliance_result'],
            'compliance_message': case['actual']['compliance_message'],
            'cargo_items': actual_cargo,
            'fees': case['actual']['fees'],
            'predicted_compliance_result': case['predicted']['compliance_result'],
            'predicted_compliance_message': case['predicted']['compliance_message'],
            'predicted_cargo_items': predicted_cargo,
            'predicted_fees': case['predicted']['fees']
        })
    
    # Create DataFrame and save to CSV
    df_output = pd.DataFrame(data)
    df_output.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nPredictions saved to {output_file}")

# =====================================================================
# MAIN PROGRAM
# =====================================================================

if __name__ == "__main__":
    # File paths
    POLICY_PATH = "data/policy.txt"
    DATASET_PATH = "data/dataset.csv"
    
    print("Starting Luggage Policy Analysis with RAG...")
    
    # Step 1: Load the luggage policy from local file
    print("Loading luggage policy...")
    luggage_policy = load_local_file(POLICY_PATH)
    if not luggage_policy:
        print("Error: Could not load luggage policy.")
        exit(1)
    
    # Step 2: Load and prepare the test dataset
    print("Loading test dataset...")
    try:
        # Load dataset from local file
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Sample 100 random customers from the dataset
    print("Sampling 100 random customers...")
    df = df.sample(n=900, random_state=42)
    
    # Step 3: Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = LuggagePolicyRAG(
        dataset_path=DATASET_PATH,
        top_k=15,
        use_sentence_transformers=True
    )
    
    # Step 4: Initialize WatsonX LLM
    print("Initializing WatsonX LLM...")
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key="YOUR_WATSONX_API_KEY"
    )
    project_id = "a31eb1db-e559-4a08-b0fa-638fdc608777"

    # Configure LLM parameters for better performance
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
        GenParams.MAX_NEW_TOKENS: 750,  # Increased for more detailed responses
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.05,    # Lower temperature for more precise responses
        GenParams.TOP_K: 10,
        GenParams.TOP_P: 0.9,
        GenParams.REPETITION_PENALTY: 1.1  # Avoid repetition
    }
    
    llm = WatsonxLLM(
        model_id="mistralai/mistral-large",
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id,
        params=parameters
    )
    
    # Step 5: Create improved prompt with specific question breakdown
    # The prompt is designed to guide the LLM through a step-by-step analysis
    luggage_prompt = PromptTemplate(
        input_variables=["policy", "travel_class", "age_category", "luggages", "similar_cases"],
        template="""Eres un experto en verificación de conformidad de equipaje para aerolíneas. Tu objetivo es analizar con precisión cada caso y determinar si el equipaje cumple con las políticas de la aerolínea.

POLÍTICA DE EQUIPAJE:
{policy}

CASO ACTUAL:
```json
{{
  "travel_class": "{travel_class}",
  "age_category": "{age_category}",
  "luggages": {luggages}
}}
```

CASOS SIMILARES (para referencia):
{similar_cases}

INSTRUCCIONES PASO A PASO:

1. ANALIZA LAS ASIGNACIONES DE EQUIPAJE PERMITIDAS:
   - ¿Qué asignación básica de equipaje corresponde a este pasajero según su clase ({travel_class}) y categoría de edad ({age_category})?
   - ¿Cuáles son los límites específicos de peso y dimensiones para cada tipo de equipaje (de mano, personal, facturado)?

2. VERIFICA CADA PIEZA DE EQUIPAJE:
   - Para cada pieza de equipaje en el JSON proporcionado:
     - ¿Cumple con las dimensiones máximas permitidas? (Calcula el volumen total si es necesario)
     - ¿Está dentro del límite de peso correspondiente?
     - ¿Requiere manejo especial o debe ir como carga?
     - ¿Excede la cantidad permitida para su tipo (mano/facturado)?

3. CALCULA LAS TARIFAS APLICABLES:
   - Para cada exceso identificado, aplica la tarifa exacta según la política:
     - Exceso de peso: ¿Cuánto se cobra por kg adicional?
     - Equipaje adicional: ¿Cuál es la tarifa por pieza extra?
     - Equipaje sobredimensionado: ¿Aplican cargos especiales?
   - Suma todas las tarifas para obtener el costo total.

4. IDENTIFICA ARTÍCULOS QUE DEBEN IR COMO CARGA:
   - ¿Hay artículos que exceden los límites absolutos y deben enviarse como carga?
   - Enumera cada uno de estos artículos.

5. DETERMINA EL RESULTADO FINAL:
   - Basado en todo el análisis anterior:
     - ¿Es conforme el equipaje en general? (true/false)
     - ¿Cuál es el motivo principal de no conformidad (si aplica)?
     - ¿Cuál es el monto total de tarifas adicionales?

RESPONDE USANDO ÚNICAMENTE ESTE FORMATO JSON:
```json
{{
  "compliance_result": true/false,
  "compliance_message": "Explicación breve y precisa de la conformidad o no conformidad",
  "cargo_items": ["item1", "item2", "..."],
  "fees": 0
}}
```

REQUISITOS IMPORTANTES:
1. "compliance_result" debe ser un valor booleano (true o false)
2. "compliance_message" debe explicar claramente el resultado
3. "cargo_items" debe ser un array (vacío si no hay elementos)
4. "fees" debe ser un número exacto sin símbolos monetarios
5. Si hay varias razones de no conformidad, menciona la principal en "compliance_message"

REQUISITOS IMPORTANTES DE FORMATO:
1. Tu respuesta DEBE estar en formato JSON válido y DEBE comenzar y terminar con llaves {{}}
2. NO incluyas ningún texto o explicación fuera del objeto JSON
3. Usa valores booleanos literales (true/false) sin comillas
4. "compliance_result" debe ser exactamente true o false (booleano)
5. "compliance_message" debe ser un string entre comillas dobles
6. "cargo_items" debe ser un array, incluso si está vacío []
7. "fees" debe ser un número (sin comillas ni símbolos)

NOTA: Aplica la política al pie de la letra para determinar la conformidad y calcular las tarifas exactas.
"""

    )
    
    # Step 6: Create chain for processing
    luggage_chain = luggage_prompt | llm

    # Step 7: Process each entry in the dataset
    results = []

    # Create directory for storing raw responses
    if not os.path.exists("raw_responses"):
        os.makedirs("raw_responses")

    print("\nProcessing customer cases...")
    for index, row in enumerate(df.iterrows()):
        row_index, row_data = row  # Unpack the tuple (index, Series)
        
        # Étape 1 : Trouver des cas similaires (sans fuite)
        similar_indices = rag_system.get_similar_cases(row_data)

        # Étape 2 : Récupérer les détails complets UNIQUEMENT pour les cas sélectionnés
        similar_cases_details = [rag_system.get_case_details(idx) for idx in similar_indices[:3]]  # Top 3 seulement

        # Étape 3 : Utiliser dans le prompt
        similar_cases_text = ""
        for case in similar_cases_details:
            similar_cases_text += f"""Cas similaire trouvé :
        - Classe : {case['travel_class']}
        - Catégorie d'âge : {case['age_category']}
        - Équipage : {case['luggages']}
        → Résultat : {'Conforme' if case['compliance_result'] else 'Non conforme'}
        → Frais : {case['fees']} €\n\n"""
        
        input_data = {
            "policy": luggage_policy,
            "travel_class": row_data['travel_class'],
            "age_category": row_data['age_category'],
            "luggages": row_data['luggages'],
            "similar_cases": similar_cases_text
        }
        
        processed = False  # Flag to track successful processing
        
        for attempt in range(3):  # Try up to 3 times
            try:
                print(f"\n--- Processing entry {index + 1}/{len(df)} ---")
                
                # Invoke the LLM
                result = luggage_chain.invoke(input_data)
                
                # Save raw response to file with proper encoding
                with open(f"raw_responses/response_{index + 1}.txt", "w", encoding="utf-8") as f:
                    f.write(result)
                
                # Clean markdown and extra text
                cleaned_result = re.sub(r'```json|```|EJEMPLO(:)?|EJEMPLO DE RESPUESTA(:)?', '', result)
                print(f"Raw response (beginning): {cleaned_result[:200]}...")
                
                # Try to parse JSON directly
                try:
                    model_result = json.loads(cleaned_result)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from text
                    model_result = extract_json_from_text(result)
                    if not model_result:
                        print(f"JSON extraction failed for entry {index + 1}, attempt {attempt + 1}")
                        time.sleep(5)  # Wait longer before retrying
                        continue
                
                # Verify that result contains all required fields
                required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
                if all(field in model_result for field in required_fields):
                    # Convert fees to number if it's a string
                    if isinstance(model_result["fees"], str):
                        try:
                            # Remove currency symbols and commas
                            fees_str = re.sub(r'[^\d.,]', '', model_result["fees"])
                            model_result["fees"] = float(fees_str.replace(',', '.'))
                        except ValueError:
                            model_result["fees"] = 0
                    
                    # Ensure cargo_items is a list
                    if not isinstance(model_result["cargo_items"], list):
                        model_result["cargo_items"] = normalize_cargo_items(model_result["cargo_items"])
                    
                    # Ensure compliance_result is boolean
                    if isinstance(model_result["compliance_result"], str):
                        model_result["compliance_result"] = model_result["compliance_result"].lower() == "true"
                    
                    # Add result to results list
                    results.append({
                        'predicted': model_result,
                        'actual': {
                            'compliance_result': row_data['compliance_result'],
                            'compliance_message': row_data['compliance_message'],
                            'cargo_items': normalize_cargo_items(row_data['cargo_items']),
                            'fees': row_data['fees']
                        }
                    })
                    processed = True  # Mark as successfully processed
                    break
                else:
                    missing_fields = [field for field in required_fields if field not in model_result]
                    print(f"Incomplete result for entry {index + 1}. Missing fields: {missing_fields}")
                    time.sleep(2)
                    continue
                    
            except KeyboardInterrupt:
                print("Operation interrupted by user")
                exit(1)  # Exit completely on interruption
            except UnicodeEncodeError as e:
                print(f"Unicode encoding error for entry {index + 1}: {e}")
                # Clean the string by removing problematic characters
                result = result.encode('ascii', 'ignore').decode('ascii')
                # Try to continue processing with cleaned string
                continue
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for entry {index + 1}: {e}")
                # Use fallback to the enhanced extraction function in the next iteration
                time.sleep(2)
                continue
            except Exception as e:
                print(f"Unexpected error for entry {index + 1}: {type(e).__name__}: {e}")
                time.sleep(3)  # Wait before retrying
                
        # If all attempts fail
        if not processed:
            print(f"Failed to process entry {index + 1}, using default value")
            results.append({
                'predicted': {
                    'compliance_result': False,
                    'compliance_message': "Analysis failed",
                    'cargo_items': [],
                    'fees': 0
                },
                'actual': {
                    'compliance_result': row_data['compliance_result'],
                    'compliance_message': row_data['compliance_message'],
                    'cargo_items': normalize_cargo_items(row_data['cargo_items']),
                    'fees': row_data['fees']
                }
            })

    # Step 8: Calculate metrics
    print("\nCalculating performance metrics...")
    total = len(results)
    if total > 0:
        # Calculer et afficher les métriques avancées
        enhanced_metrics = calculate_enhanced_metrics(results)
        print_metrics_summary(enhanced_metrics)
        
        # Sauvegarder les métriques dans un fichier JSON
        with open('luggage_metrics_summary.json', 'w', encoding="utf-8") as f:
            json.dump(enhanced_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed metrics saved to 'luggage_metrics_summary.json'")
        
        # Analyse par classe de voyage
        metrics_by_class = {}
        
        for i, r in enumerate(results):
            
            travel_class = df.iloc[i]['travel_class']
            is_correct = r['predicted']['compliance_result'] == r['actual']['compliance_result']
            
            if travel_class not in metrics_by_class:
                metrics_by_class[travel_class] = {"correct": 0, "total": 0}
            
            metrics_by_class[travel_class]["total"] += 1
            if is_correct:
                metrics_by_class[travel_class]["correct"] += 1
        
        print("\nPerformance by Travel Class:")
        for cls, data in metrics_by_class.items():
            accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
            print(f"{cls}: {accuracy:.2f}% ({data['correct']}/{data['total']})")
        
        # Analyse des erreurs les plus courantes
        error_cases = [r for r in results if r['predicted']['compliance_result'] != r['actual']['compliance_result']]
        if error_cases:
            print("\nCommon Error Analysis:")
            error_messages = [r['predicted']['compliance_message'] for r in error_cases]
            
            # Extraction de mots-clés des messages d'erreur
            all_words = []
            for msg in error_messages:
                words = re.findall(r'\b\w{4,}\b', msg.lower())
                all_words.extend(words)
            
            # Comptage des mots les plus fréquents
            from collections import Counter
            word_counts = Counter(all_words)
            most_common = word_counts.most_common(5)
            
            print("Most common terms in error messages:")
            for word, count in most_common:
                print(f"  {word}: {count} occurrences")
    else:
        print("No results were processed. Cannot calculate metrics.")
        
    #Step 9: Generate predictions CSV file
    print("\nGenerating predictions CSV file...")
    generate_predictions_csv(results, df)
