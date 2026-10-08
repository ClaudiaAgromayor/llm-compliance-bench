# Input luggages: properly handles up to 6 different luggage items, Ignores any compliance information inside the luggage data
# RAG: ComplianceRAG: Focuses only on travel class and age category for compliance analysis - FeesRAG: Creates 4 clusters for fee calculation



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
from collections import Counter

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

def preprocess_luggage_data(row):
    """
    Convert luggage data into a text representation for similarity comparison,
    handling up to 6 different luggage items and ignoring compliance info
    """
    # Create a text representation of basic customer attributes
    text_repr = f"travel_class:{row['travel_class']} age_category:{row['age_category']} "
    
    # Process luggage items if they exist
    if pd.notna(row['luggages']):
        try:
            # Parse JSON luggage data
            luggage_items = json.loads(row['luggages'])
            
            # Ensure we process a maximum of 6 luggage items
            for i, item in enumerate(luggage_items[:6]):
                # Skip any items that might contain compliance info
                if isinstance(item, dict) and 'storage' in item:
                    text_repr += f"item{i+1}:{item['storage']}_"
                    text_repr += f"w{item['weight']}_"
                    text_repr += f"h{item['height']}_"
                    text_repr += f"d{item['depth']}_"
                    text_repr += f"l{item.get('length', 0)}_"  # Add length if available
                    text_repr += f"w{item.get('width', 0)}_"   # Add width if available
                    text_repr += f"special{item.get('special', False)}_"
                    text_repr += f"excess{item.get('excess', False)} "
        except:
            # If JSON parsing fails, use the raw text
            text_repr += f"luggages:{row['luggages']}"
    
    return text_repr.strip()

# =====================================================================
# RAG SYSTEMS
# =====================================================================

class ComplianceRAG:
    """
    Retrieval Augmented Generation system for luggage policy compliance
    Finds similar customer cases based on travel class and age category
    """
    
    def __init__(self, dataset_path, top_k=5):
        """
        Initialize the Compliance RAG system
        
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
        
        print(f"Loaded {len(self.df)} customer cases for Compliance RAG system")
        
        # For compliance, focus on travel class and age category
        self.df['compliance_text'] = self.df.apply(
            lambda row: f"travel_class:{row['travel_class']} age_category:{row['age_category']}", 
            axis=1
        )
    
    def vectorize_data(self):
        """Create vector representations based on travel class and age category"""
        print("Vectorizing customer data for compliance similarity search...")
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.df['compliance_text'])
        print("Compliance vectorization complete")
    
    def get_similar_cases(self, query_row):
        """
        Find similar compliance cases based on travel class and age category
        
        Args:
            query_row: A DataFrame row containing the customer case
            
        Returns:
            List of similar cases with their similarity scores
        """
        # Create text representation focusing on travel class and age
        query_text = f"travel_class:{query_row['travel_class']} age_category:{query_row['age_category']}"
        
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

class FeesRAG:
    """
    Specialized RAG system for fee calculation using clustering approach
    Groups age and category pairs into fee clusters
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the Fees RAG system with clustering
        
        Args:
            dataset_path: Path to the dataset containing all customer cases
        """
        self.load_dataset(dataset_path)
        self.create_fee_clusters()
    
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
        
        print(f"Loaded {len(self.df)} customer cases for Fees RAG system")
    
    def create_fee_clusters(self):
        """
        Create 4 clusters based on the fee structure:
        - Cluster 0: No fees ($0)
        - Cluster 1: Overweight Bags fee ($75 per bag)
        - Cluster 2: Oversized Bags fee ($100 per bag)
        - Cluster 3: Extra Pieces fee ($150 per additional piece)
        """
        print("Creating fee clusters...")
        
        # Define the cluster centroids based on fee types
        self.centroids = {
            0: 0,       # No fees
            1: 75,      # Overweight Bags
            2: 100,     # Oversized Bags
            3: 150      # Extra Pieces
        }
        
        # Map each case to a cluster based on its fee
        def assign_fee_cluster(row):
            # Skip cases with zero fees
            if row['fees'] == 0:
                return 0
            
            # Get the closest cluster based on fee value
            fee = float(row['fees'])
            distances = {k: abs(fee - v) for k, v in self.centroids.items()}
            return min(distances.items(), key=lambda x: x[1])[0]
        
        # Add cluster assignment to the dataframe
        self.df['fee_cluster'] = self.df.apply(assign_fee_cluster, axis=1)
        
        # Create a mapping of (travel_class, age_category) to common fee cluster
        self.class_age_to_cluster = {}
        grouped = self.df.groupby(['travel_class', 'age_category'])
        
        for (travel_class, age_category), group in grouped:
            # Get the most common cluster for this combination
            cluster_counts = group['fee_cluster'].value_counts()
            most_common_cluster = cluster_counts.idxmax() if len(cluster_counts) > 0 else 0
            self.class_age_to_cluster[(travel_class, age_category)] = most_common_cluster
        
        print("Fee clustering complete")
    
    def get_fee_recommendation(self, travel_class, age_category, luggages):
        """
        Get a fee recommendation based on the cluster for travel class and age category
        
        Args:
            travel_class: Customer travel class
            age_category: Customer age category
            luggages: Customer luggage details (for more precise recommendations)
            
        Returns:
            Dictionary with fee information
        """
        # Get the cluster for this combination
        key = (travel_class, age_category)
        cluster = self.class_age_to_cluster.get(key, 0)  # Default to no fee if not found
        
        # Get examples from this cluster
        examples = self.df[
            (self.df['travel_class'] == travel_class) & 
            (self.df['age_category'] == age_category) & 
            (self.df['fee_cluster'] == cluster)
        ].head(3)
        
        # Calculate average fee for this cluster (if examples exist)
        avg_fee = examples['fees'].mean() if len(examples) > 0 else self.centroids[cluster]
        
        # Determine fee category and reasoning
        fee_categories = {
            0: "No additional fees",
            1: "Overweight Bags (23-32 kg in Economy Class): $75 per bag",
            2: "Oversized Bags (Dimensions 159-203 cm): $100 per bag",
            3: "Extra Pieces: $150 per additional piece"
        }
        
        # Fine-tune calculation based on luggage details
        fee_count = 0
        reason = fee_categories[cluster]
        
        # Try to determine how many items might incur this fee type
        if cluster > 0 and luggages:
            try:
                luggage_items = json.loads(luggages) if isinstance(luggages, str) else luggages
                
                if cluster == 1:  # Overweight
                    fee_count = sum(1 for item in luggage_items if 'weight' in item and 23 <= item['weight'] <= 32)
                elif cluster == 2:  # Oversized
                    fee_count = sum(1 for item in luggage_items if 'depth' in item and 'height' in item and 
                                   159 <= (item['depth'] + item['height'] + item.get('width', 0)) <= 203)
                elif cluster == 3:  # Extra pieces
                    # Count excess items based on travel class
                    allowed = 2 if travel_class == "Business" else 1
                    fee_count = max(0, len(luggage_items) - allowed)
            except:
                # If parsing fails, just use the cluster average
                fee_count = 1 if avg_fee > 0 else 0
        
        # Calculate recommended fee
        recommended_fee = self.centroids[cluster] * fee_count if fee_count > 0 else avg_fee
        
        return {
            "cluster": cluster,
            "category": fee_categories[cluster],
            "average_fee": avg_fee,
            "recommended_fee": recommended_fee,
            "examples": examples[['travel_class', 'age_category', 'fees']].to_dict('records'),
            "reasoning": reason
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
                # Convert dictionary to tuple of tuples ((key1, value1), (key2, value2), ...)
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
    
    print("Starting Luggage Policy Analysis with Dual RAG System...")
    
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
    
    # Step 3: Initialize both RAG systems
    print("Initializing dual RAG systems...")
    compliance_rag = ComplianceRAG(DATASET_URL)
    fees_rag = FeesRAG(DATASET_URL)
    
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
        input_variables=["policy", "travel_class", "age_category", "luggages", "similar_cases", "fee_recommendation"],
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

{fee_recommendation}

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
        
        # Get similar cases for compliance from ComplianceRAG
        similar_cases = compliance_rag.get_similar_cases(row_data)
        
        # Format similar cases as text for the prompt
        similar_cases_text = ""
        for i, case in enumerate(similar_cases[:3]):  # Limit to top 3 for clarity
            similar_cases_text += f"Caso similar #{i+1} (Similitud: {case['similarity']:.2f}):\n"
            similar_cases_text += f"- Clase: {case['travel_class']}\n"
            similar_cases_text += f"- Categoría de edad: {case['age_category']}\n"
            similar_cases_text += f"- Equipaje: {case['luggages']}\n"
            similar_cases_text += f"- ¿Conforme?: {case['compliance_result']}\n"
            similar_cases_text += f"- Mensaje: {case['compliance_message']}\n"
            similar_cases_text += f"- Artículos como carga: {case['cargo_items']}\n"
            similar_cases_text += f"- Tarifas: {case['fees']}\n\n"
        
        # Get fee recommendation from FeesRAG
        fee_recommendation = fees_rag.get_fee_recommendation(
            row_data['travel_class'], 
            row_data['age_category'],
            row_data['luggages']
        )
        
        # Format fee recommendation as text
        fee_recommendation_text = f"""
GUÍA DE TARIFAS RECOMENDADA:
- Categoría de tarifa: {fee_recommendation['category']}
- Tarifa promedio para casos similares: ${fee_recommendation['average_fee']:.2f}
- Tarifa recomendada para este caso: ${fee_recommendation['recommended_fee']:.2f}
- Razonamiento: {fee_recommendation['reasoning']}
"""
        
        # Update input data to include both RAG results
        input_data = {
            "policy": luggage_policy,
            "travel_class": row_data['travel_class'],
            "age_category": row_data['age_category'],
            "luggages": row_data['luggages'],
            "similar_cases": similar_cases_text,
            "fee_recommendation": fee_recommendation_text
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
        # Calculate and display enhanced metrics
        enhanced_metrics = calculate_enhanced_metrics(results)
        print_metrics_summary(enhanced_metrics)
        
        # Save metrics to a JSON file
        with open('luggage_metrics_summary.json', 'w', encoding="utf-8") as f:
            json.dump(enhanced_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed metrics saved to 'luggage_metrics_summary.json'")
        
        # Analysis by travel class
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
        
        # Common error analysis
        error_cases = [r for r in results if r['predicted']['compliance_result'] != r['actual']['compliance_result']]
        if error_cases:
            print("\nCommon Error Analysis:")
            error_messages = [r['predicted']['compliance_message'] for r in error_cases]
            
            # Extract keywords from error messages
            all_words = []
            for msg in error_messages:
                if isinstance(msg, str):
                    words = re.findall(r'\b\w{4,}\b', msg.lower())
                    all_words.extend(words)
            
            # Count most frequent words
            word_counts = Counter(all_words)
            most_common = word_counts.most_common(5)
            
            print("Most common terms in error messages:")
            for word, count in most_common:
                print(f"  {word}: {count} occurrences")
    else:
        print("No results were processed. Cannot calculate metrics.")

"""Calculating performance metrics...

==================================================
COMPLIANCE METRICS SUMMARY
==================================================
Accuracy: 94.50%
Precision: 0.92
Recall: 0.91
F1 Score: 0.92

Confusion Matrix:
  TP: 61 | FP: 5
  FN: 6 | TN: 128

==================================================
FEE ESTIMATION METRICS
==================================================
Exact Match: 70.50%
Within $5: 70.50%
Within $10: 70.50%
Within $20: 70.50%
Mean Absolute Error: $43.12
Root Mean Squared Error: $88.34
Mean Absolute Percentage Error: 1750000000028.00%
Improvement over baseline: -3.29%

==================================================
CARGO ITEMS METRICS
==================================================
Average Accuracy: 94.50%

Sample Size: 200 cases

Sample Size: 200 cases
Sample Size: 200 cases
Detailed metrics saved to 'luggage_metrics_summary.json'

Performance by Travel Class:
Business: 95.38% (62/65)
Detailed metrics saved to 'luggage_metrics_summary.json'

Performance by Travel Class:
Business: 95.38% (62/65)

Performance by Travel Class:
Business: 95.38% (62/65)
Economy: 94.03% (63/67)
Performance by Travel Class:
Business: 95.38% (62/65)
Economy: 94.03% (63/67)
Economy: 94.03% (63/67)
First: 94.12% (64/68)

Common Error Analysis:
Most common terms in error messages:
  luggage: 8 occurrences
  with: 7 occurrences
  policy: 7 occurrences
  checked: 5 occurrences
  complies: 5 occurrences
  """