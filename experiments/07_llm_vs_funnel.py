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
from luggage_calculator import LuggageFeeCalculator


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

# Github integration: Downloads the dataset and policy directly from GitHub
def download_github_file(url):
    """
    Download a file from GitHub using the raw content URL
    """
    # Convert GitHub URL to raw content URL
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
    try:
        response = requests.get(raw_url)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return None

#Better JSON parsing: 
# Added robust JSON extraction from LLM responses and
# Normalized cargo items to ensure consistent formatting 
def extract_json_from_text(text):
    """Enhanced JSON extraction with better error handling"""
    # Remove markdown code tags and additional text
    text = re.sub(r'```json|```|EJEMPLO(:)?|EJEMPLO DE RESPUESTA(:)?', '', text)
    
    # Try direct JSON parsing first (faster)
    try:
        # Find text between first { and last }
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
        # Try each JSON match found, sorting by length (longest first)
        for json_str in sorted(json_matches, key=len, reverse=True):
            json_str = json_str.strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # If still no valid JSON, try more aggressive parsing
    try:
        # Look for key fields and build a minimal JSON
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
    
    return None

def standardize_result(model_result):
    """Standardize and clean up result fields"""
    # Ensure all required fields exist
    required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
    for field in required_fields:
        if field not in model_result:
            model_result[field] = None if field == "compliance_message" else ([] if field == "cargo_items" else 0)
    
    # Ensure correct types
    # Boolean conversion
    if isinstance(model_result["compliance_result"], str):
        model_result["compliance_result"] = model_result["compliance_result"].lower() == "true"
    
    # String conversion
    if not isinstance(model_result["compliance_message"], str):
        model_result["compliance_message"] = str(model_result["compliance_message"] or "No message")
    
    # List conversion
    model_result["cargo_items"] = normalize_cargo_items(model_result["cargo_items"])
    
    # Number conversion
    if isinstance(model_result["fees"], str):
        # Remove currency symbols and commas
        fees_str = re.sub(r'[^\d.,]', '', model_result["fees"])
        try:
            model_result["fees"] = float(fees_str.replace(',', '.'))
        except ValueError:
            model_result["fees"] = 0
    elif not isinstance(model_result["fees"], (int, float)):
        model_result["fees"] = 0
        
    return model_result
def preprocess_luggage_data(row):
    """
    Convert luggage data into a text representation for similarity comparison
    """
    # Create a text representation of the luggage details
    text_repr = f"travel_class:{row['travel_class']} age_category:{row['age_category']} "
    
    # Process luggage items if they exist
    if pd.notna(row['luggages']):
        try:
            # Parse JSON luggage data
            luggage_items = json.loads(row['luggages'])
            for i, item in enumerate(luggage_items):
                text_repr += f"item{i+1}:{item['storage']}_"
                text_repr += f"w{item['weight']}_"
                text_repr += f"h{item['height']}_"
                text_repr += f"d{item['depth']}_"
                text_repr += f"special{item['special']}_"
                text_repr += f"excess{item['excess']} "
        except:
            # If JSON parsing fails, use the raw text
            text_repr += f"luggages:{row['luggages']}"
    
    return text_repr.strip()

def normalize_cargo_items(items):
    """
    Normalize cargo items to ensure consistent formatting
    """
    if not items:
        return []
    
    if isinstance(items, str):
        if items.strip() == "":
            return []
        # Try to parse as JSON if it's a string
        try:
            parsed_items = json.loads(items)
            if isinstance(parsed_items, list):
                return parsed_items
            else:
                return [items]
        except:
            # If not JSON, split by comma if possible
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
    """
    Retrieval Augmented Generation system for luggage policy compliance
    Finds similar customer cases to enhance context for the LLM
    """
    
    def __init__(self, dataset_path, top_k=5):
        """
        Initialize the RAG system with dataset and parameters
        
        Args:
            dataset_path: Path to the dataset containing all customer cases
            top_k: Number of similar cases to retrieve
        """
        self.top_k = top_k
        self.load_dataset(dataset_path)
        self.vectorize_data()
    
    def load_dataset(self, dataset_path):
        """Load and preprocess the dataset"""
        # Handle both local files and GitHub URLs
        if dataset_path.startswith('http'):
            # Download from GitHub if URL
            content = download_github_file(dataset_path)
            if content:
                self.df = pd.read_csv(StringIO(content))
            else:
                raise ValueError(f"Could not download dataset from {dataset_path}")
        else:
            # Load from local file
            self.df = pd.read_csv(dataset_path)
        
        print(f"Loaded {len(self.df)} customer cases for RAG system")
        
        # Preprocess each row for vectorization
        self.df['text_repr'] = self.df.apply(preprocess_luggage_data, axis=1)
    
    def vectorize_data(self):
        """Create vector representations of all cases"""
        print("Vectorizing luggage data for similarity search...")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.vectors = self.vectorizer.fit_transform(self.df['text_repr'])
        print("Vectorization complete")
    
    def get_similar_cases(self, query_row):
        """
        Find similar cases for a given query
        
        Args:
            query_row: A DataFrame row containing the customer case
            
        Returns:
            List of similar cases with their similarity scores
        """
        # Create text representation of the query
        query_text = preprocess_luggage_data(query_row)
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate similarity with all cases
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get indices of top similar cases
        top_indices = similarities.argsort()[-self.top_k-1:-1][::-1]
        
        # Create list of similar cases with their similarity scores
        similar_cases = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            similar_cases.append({
                "similarity": similarities[idx],
                "travel_class": row['travel_class'],
                "age_category": row['age_category'],
                "luggages": row['luggages'],
                "compliance_result": row['compliance_result'],
                "compliance_message": row['compliance_message'],
                "cargo_items": normalize_cargo_items(row['cargo_items']),
                "fees": row['fees']
            })
        
        return similar_cases

#
# FUNNEL
#

def evaluate_with_funnel(df, policy_text):
    """Evaluate the dataset using the improved funnel approach"""
    calculator = LuggageFeeCalculator(policy_text)
    results = []
    
    for index, row in df.iterrows():
        try:
            # Parse luggage items from JSON
            luggage_json = json.loads(row['luggages'])
            
            # Process through the funnel
            prediction = calculator.process_luggage(
                travel_class=row['travel_class'],
                age_category=row['age_category'],
                luggage_items=luggage_json
            )
            
            # Debug output to diagnose issues
            print(f"Entry {index}: Class={row['travel_class']}, Age={row['age_category']}")
            print(f"  Predicted: compliance={prediction['compliance_result']}, fees={prediction['fees']}")
            print(f"  Actual: compliance={row['compliance_result']}, fees={row['fees']}")
            print(f"  Luggage items: {luggage_json}")
            
            # Store result
            results.append({
                'predicted': prediction,
                'actual': {
                    'compliance_result': row['compliance_result'],
                    'compliance_message': row['compliance_message'],
                    'cargo_items': normalize_cargo_items(row['cargo_items']),
                    'fees': row['fees']
                }
            })
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            # Add default result on error
            results.append({
                'predicted': {
                    'compliance_result': False,
                    'compliance_message': f"Processing error: {str(e)}",
                    'cargo_items': [],
                    'fees': 0
                },
                'actual': {
                    'compliance_result': row['compliance_result'],
                    'compliance_message': row['compliance_message'],
                    'cargo_items': normalize_cargo_items(row['cargo_items']),
                    'fees': row['fees']
                }
            })
    
    return results

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
    
    # Extract actual and predicted values
    actual_compliance = [bool(r['actual']['compliance_result']) for r in results]
    predicted_compliance = [bool(r['predicted']['compliance_result']) for r in results]
    actual_fees = [float(r['actual']['fees']) for r in results]
    predicted_fees = [float(r['predicted']['fees']) for r in results]
    
    # Basic compliance metrics
    compliance_correct = sum(1 for i in range(total) if predicted_compliance[i] == actual_compliance[i])
    compliance_accuracy = compliance_correct / total * 100
    
    # Calculate confusion matrix values
    true_positives = sum(1 for i in range(total) if actual_compliance[i] and predicted_compliance[i])
    true_negatives = sum(1 for i in range(total) if not actual_compliance[i] and not predicted_compliance[i])
    false_positives = sum(1 for i in range(total) if not actual_compliance[i] and predicted_compliance[i])
    false_negatives = sum(1 for i in range(total) if actual_compliance[i] and not predicted_compliance[i])
    
    # Advanced classification metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Fee metrics with different thresholds
    fee_exact_match = sum(1 for i in range(total) if abs(predicted_fees[i] - actual_fees[i]) < 0.01)
    fee_within_5 = sum(1 for i in range(total) if abs(predicted_fees[i] - actual_fees[i]) <= 5)
    fee_within_10 = sum(1 for i in range(total) if abs(predicted_fees[i] - actual_fees[i]) <= 10)
    fee_within_20 = sum(1 for i in range(total) if abs(predicted_fees[i] - actual_fees[i]) <= 20)
    
    # Calculate fee error metrics
    fee_errors = [abs(predicted_fees[i] - actual_fees[i]) for i in range(total)]
    mae_fees = sum(fee_errors) / total
    mse_fees = sum(e**2 for e in fee_errors) / total
    rmse_fees = mse_fees ** 0.5
    
    # Calculate percentage error
    # Avoid division by zero by adding small epsilon to denominator
    epsilon = 1e-10
    percentage_errors = [abs(predicted_fees[i] - actual_fees[i]) / (actual_fees[i] + epsilon) * 100 for i in range(total)]
    mape_fees = sum(percentage_errors) / total
    
    # Calculate relative improvement over baseline (always predicting 0 fee)
    baseline_mae = sum(abs(f) for f in actual_fees) / total
    improvement = (baseline_mae - mae_fees) / baseline_mae * 100 if baseline_mae > 0 else 0
    
    # Cargo items analysis
    cargo_match = []
    for r in results:
        predicted_items_raw = r['predicted']['cargo_items']
        actual_items_raw = r['actual']['cargo_items']
        
        predicted_items = set()
        
        for item in predicted_items_raw:
            if isinstance(item, dict):
                # Convertir le dictionnaire en tuple de tuples ((clé1, valeur1), (clé2, valeur2), ...)
                predicted_items.add(tuple(sorted(item.items())))
            else:
                predicted_items.add(item)
                
        actual_items = set()
        for item in actual_items_raw:
            if isinstance(item, dict):
                actual_items.add(tuple(sorted(item.items())))
            else:
                actual_items.add(item)
                
                
        union = len(predicted_items.union(actual_items))
        intersection = len(predicted_items.intersection(actual_items))
        
        jaccard = intersection / union if union > 0 else 1.0
        cargo_match.append(jaccard)
    
    avg_cargo_accuracy = sum(cargo_match) / len(cargo_match) * 100 if cargo_match else 0
    
    # Return comprehensive metrics dictionary
    return {
        "compliance": {
            "accuracy": compliance_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }
        },
        "fees": {
            "exact_match": fee_exact_match / total * 100,
            "within_$5": fee_within_5 / total * 100,
            "within_$10": fee_within_10 / total * 100,
            "within_$20": fee_within_20 / total * 100,
            "mae": mae_fees,
            "rmse": rmse_fees,
            "mape": mape_fees,
            "baseline_improvement": improvement
        },
        "cargo_items": {
            "average_accuracy": avg_cargo_accuracy
        },
        "sample_size": total
    }

def print_metrics_summary(metrics):
    """
    Print a readable summary of the metrics
    """
    print("\n" + "="*50)
    print("COMPLIANCE METRICS SUMMARY")
    print("="*50)
    
    print(f"Accuracy: {metrics['compliance']['accuracy']:.2f}%")
    print(f"Precision: {metrics['compliance']['precision']:.2f}")
    print(f"Recall: {metrics['compliance']['recall']:.2f}")
    print(f"F1 Score: {metrics['compliance']['f1_score']:.2f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['compliance']['confusion_matrix']
    print(f"  TP: {cm['true_positives']} | FP: {cm['false_positives']}")
    print(f"  FN: {cm['false_negatives']} | TN: {cm['true_negatives']}")
    
    print("\n" + "="*50)
    print("FEE ESTIMATION METRICS")
    print("="*50)
    
    print(f"Exact Match: {metrics['fees']['exact_match']:.2f}%")
    print(f"Within $5: {metrics['fees']['within_$5']:.2f}%")
    print(f"Within $10: {metrics['fees']['within_$10']:.2f}%")
    print(f"Within $20: {metrics['fees']['within_$20']:.2f}%")
    print(f"Mean Absolute Error: ${metrics['fees']['mae']:.2f}")
    print(f"Root Mean Squared Error: ${metrics['fees']['rmse']:.2f}")
    print(f"Mean Absolute Percentage Error: {metrics['fees']['mape']:.2f}%")
    print(f"Improvement over baseline: {metrics['fees']['baseline_improvement']:.2f}%")
    
    print("\n" + "="*50)
    print("CARGO ITEMS METRICS")
    print("="*50)
    print(f"Average Accuracy: {metrics['cargo_items']['average_accuracy']:.2f}%")
    
    print(f"\nSample Size: {metrics['sample_size']} cases")


# =====================================================================
# MAIN PROGRAM
# =====================================================================

if __name__ == "__main__":
    # File URLs and paths
    POLICY_URL = "https://github.com/DecisionsDev/policy-corpus/blob/main/luggage/luggage_policy.md"
    DATASET_URL = "https://github.com/DecisionsDev/policy-corpus/blob/main/luggage/luggage_compliance/luggage_policy_test_dataset_1K.csv"
    
    print("Starting Luggage Policy Analysis with RAG...")
    
    # Step 1: Load the luggage policy from GitHub
    print("Loading luggage policy...")
    luggage_policy = download_github_file(POLICY_URL)
    if not luggage_policy:
        print("Error: Could not download luggage policy.")
        exit(1)
    
    # Step 2: Load and prepare the test dataset
    print("Loading test dataset...")
    try:
        # Load dataset from GitHub
        dataset_content = download_github_file(DATASET_URL)
        if dataset_content:
            df = pd.read_csv(StringIO(dataset_content))
        else:
            print("Error: Could not download test dataset.")
            exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Sample 200 random customers from the dataset
    print("Sampling 200 random customers...")
    df = df.sample(n=200, random_state=42)
    
    # Step 3: Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = LuggagePolicyRAG(DATASET_URL)
    
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

# Step 7: Process each entry with LLM approach
llm_results = []

# Create directory for storing raw responses
if not os.path.exists("raw_responses"):
    os.makedirs("raw_responses")

print("\nProcessing customer cases with LLM approach...")
for index, row in enumerate(df.iterrows()):
    row_index, row_data = row  # Unpack the tuple (index, Series)
    
    # Get similar cases using RAG for context enhancement
    similar_cases = rag_system.get_similar_cases(row_data)
    
    # Format similar cases as text for the prompt
    similar_cases_text = ""
    for i, case in enumerate(similar_cases[:3]):  # Limit to top 3 for clarity
        similar_cases_text += f"Caso similar #{i+1} (Similitud: {case['similarity']:.2f}):\n"
        similar_cases_text += f"- Clase: {case['travel_class']}\n"
        similar_cases_text += f"- Categoría de edad: {case['age_category']}\n"
        similar_cases_text += f"- Equipaje: {case['luggages']}\n"
        similar_cases_text += f"- ¿Conforme?: {case['compliance_result']}\n"
        similar_cases_text += f"- Mensaje: {case['compliance_message']}\n"
        if case['cargo_items']:
            similar_cases_text += f"- Artículos como carga: {case['cargo_items']}\n"
        similar_cases_text += f"- Tarifas: {case['fees']}\n\n"
    
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
            print(f"\n--- Processing entry {index + 1}/{len(df)} with LLM ---")
            
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
                llm_results.append({
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
        llm_results.append({
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

# Step 8: Calculate metrics for LLM approach
print("\nCalculating LLM approach metrics...")
if len(llm_results) > 0:
    llm_metrics = calculate_enhanced_metrics(llm_results)
    print("\n=== LLM APPROACH METRICS ===")
    print_metrics_summary(llm_metrics)
    
    # Save LLM metrics to file
    with open('llm_luggage_metrics.json', 'w', encoding="utf-8") as f:
        json.dump(llm_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"LLM metrics saved to 'llm_luggage_metrics.json'")
else:
    print("No LLM results were processed. Cannot calculate metrics.")
    llm_metrics = None

# STEP 9: FUNNEL-BASED APPROACH
print("\nProcessing with funnel-based approach...")

# Initialize the funnel calculator
calculator = LuggageFeeCalculator(luggage_policy)
funnel_results = []

# Process each entry with funnel approach
for index, row in df.iterrows():
    try:
        print(f"--- Processing entry {index + 1}/{len(df)} with funnel ---")
        
        # Parse luggage items from JSON
        luggage_json = json.loads(row['luggages'])
        
        # Process through the funnel
        prediction = calculator.process_luggage(
            travel_class=row['travel_class'],
            age_category=row['age_category'],
            luggage_items=luggage_json
        )
        
        # Store result
        funnel_results.append({
            'predicted': prediction,
            'actual': {
                'compliance_result': row['compliance_result'],
                'compliance_message': row['compliance_message'],
                'cargo_items': normalize_cargo_items(row['cargo_items']),
                'fees': row['fees']
            }
        })
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for entry {index + 1}: {e}")
        # Use default values for failed entries
        funnel_results.append({
            'predicted': {
                'compliance_result': False,
                'compliance_message': f"JSON parsing error: {str(e)}",
                'cargo_items': [],
                'fees': 0
            },
            'actual': {
                'compliance_result': row['compliance_result'],
                'compliance_message': row['compliance_message'],
                'cargo_items': normalize_cargo_items(row['cargo_items']),
                'fees': row['fees']
            }
        })
    except Exception as e:
        print(f"Error processing entry {index + 1}: {e}")
        # Add default result on error
        funnel_results.append({
            'predicted': {
                'compliance_result': False,
                'compliance_message': f"Processing error: {str(e)}",
                'cargo_items': [],
                'fees': 0
            },
            'actual': {
                'compliance_result': row['compliance_result'],
                'compliance_message': row['compliance_message'],
                'cargo_items': normalize_cargo_items(row['cargo_items']),
                'fees': row['fees']
            }
        })

# Step 10: Calculate metrics for funnel approach
print("\nCalculating funnel approach metrics...")
if len(funnel_results) > 0:
    funnel_metrics = calculate_enhanced_metrics(funnel_results)
    print("\n=== FUNNEL-BASED APPROACH METRICS ===")
    print_metrics_summary(funnel_metrics)
    
    # Save funnel metrics to file
    with open('funnel_luggage_metrics.json', 'w', encoding="utf-8") as f:
        json.dump(funnel_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Funnel metrics saved to 'funnel_luggage_metrics.json'")
else:
    print("No funnel results were processed. Cannot calculate metrics.")
    funnel_metrics = None

# Step 11: Compare both approaches
if llm_metrics and funnel_metrics:
    print("\n=== PERFORMANCE COMPARISON ===")
    print("Metric               | LLM Approach   | Funnel Approach")
    print("--------------------|----------------|----------------")
    print(f"Compliance Accuracy  | {llm_metrics['compliance']['accuracy']:.2f}%         | {funnel_metrics['compliance']['accuracy']:.2f}%")
    print(f"Fee Exact Match      | {llm_metrics['fees']['exact_match']:.2f}%         | {funnel_metrics['fees']['exact_match']:.2f}%")
    print(f"Fee Within $5        | {llm_metrics['fees']['within_$5']:.2f}%         | {funnel_metrics['fees']['within_$5']:.2f}%")
    print(f"Fee MAE              | ${llm_metrics['fees']['mae']:.2f}          | ${funnel_metrics['fees']['mae']:.2f}")
    print(f"Fee RMSE             | ${llm_metrics['fees']['rmse']:.2f}          | ${funnel_metrics['fees']['rmse']:.2f}")
    print(f"Cargo Item Accuracy  | {llm_metrics['cargo_items']['average_accuracy']:.2f}%         | {funnel_metrics['cargo_items']['average_accuracy']:.2f}%")
    
    # Calculate and display improvement percentages
    fee_exact_improvement = funnel_metrics['fees']['exact_match'] - llm_metrics['fees']['exact_match']
    fee_mae_improvement = ((llm_metrics['fees']['mae'] - funnel_metrics['fees']['mae']) / llm_metrics['fees']['mae']) * 100 if llm_metrics['fees']['mae'] > 0 else 0
    
    print("\n=== KEY IMPROVEMENTS ===")
    print(f"Fee Exact Match Improvement: {fee_exact_improvement:.2f} percentage points")
    print(f"Fee MAE Improvement: {fee_mae_improvement:.2f}%")
    
    # Analyze by travel class
    print("\nPerformance by Travel Class:")
    
    # Group results by travel class for both approaches
    travel_classes = df['travel_class'].unique()
    
    for tc in travel_classes:
        llm_class_results = [r for i, r in enumerate(llm_results) if df.iloc[i]['travel_class'] == tc]
        funnel_class_results = [r for i, r in enumerate(funnel_results) if df.iloc[i]['travel_class'] == tc]
        
        llm_class_metrics = calculate_enhanced_metrics(llm_class_results)
        funnel_class_metrics = calculate_enhanced_metrics(funnel_class_results)
        
        print(f"\n{tc} Class:")
        print(f"  Accuracy: LLM {llm_class_metrics['compliance']['accuracy']:.2f}% vs Funnel {funnel_class_metrics['compliance']['accuracy']:.2f}%")
        print(f"  Fee Exact Match: LLM {llm_class_metrics['fees']['exact_match']:.2f}% vs Funnel {funnel_class_metrics['fees']['exact_match']:.2f}%")
        print(f"  Fee MAE: LLM ${llm_class_metrics['fees']['mae']:.2f} vs Funnel ${funnel_class_metrics['fees']['mae']:.2f}")
        
"""
=== FUNNEL-BASED APPROACH METRICS ===

==================================================
COMPLIANCE METRICS SUMMARY
==================================================
Accuracy: 45.50%
Precision: 0.15
Recall: 0.13
F1 Score: 0.14

Confusion Matrix:
  TP: 9 | FP: 51
  FN: 58 | TN: 82

==================================================
FEE ESTIMATION METRICS
==================================================
Exact Match: 49.50%
Within $5: 50.00%
Within $10: 50.00%
Within $20: 52.00%
Mean Absolute Error: $52.73
Root Mean Squared Error: $100.00
Mean Absolute Percentage Error: 33737600000013.80%
Improvement over baseline: -26.29%

==================================================
CARGO ITEMS METRICS
==================================================
Average Accuracy: 64.50%

Sample Size: 200 cases
Funnel metrics saved to 'funnel_luggage_metrics.json'

=== PERFORMANCE COMPARISON ===
Metric               | LLM Approach   | Funnel Approach
--------------------|----------------|----------------
Compliance Accuracy  | 93.50%         | 45.50%
Fee Exact Match      | 69.50%         | 49.50%
Fee Within $5        | 69.50%         | 50.00%
Fee MAE              | $44.62          | $52.73
Fee RMSE             | $90.71          | $100.00
Cargo Item Accuracy  | 94.75%         | 64.50%

=== KEY IMPROVEMENTS ===
Fee Exact Match Improvement: -20.00 percentage points
Fee MAE Improvement: -18.15%

Performance by Travel Class:

Business Class:
  Accuracy: LLM 93.85% vs Funnel 46.15%
  Fee Exact Match: LLM 73.85% vs Funnel 49.23%
  Fee MAE: LLM $32.31 vs Funnel $56.27

Economy Class:
  Accuracy: LLM 92.54% vs Funnel 47.76%
  Fee Exact Match: LLM 71.64% vs Funnel 44.78%
  Fee MAE: LLM $35.45 vs Funnel $47.44

First Class:
  Accuracy: LLM 94.12% vs Funnel 42.65%
  Fee Exact Match: LLM 63.24% vs Funnel 54.41%
  Fee MAE: LLM $65.44 vs Funnel $54.54
"""