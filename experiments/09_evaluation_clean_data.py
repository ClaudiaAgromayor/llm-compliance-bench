import requests
import json
import pandas as pd
import re
from io import StringIO
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def download_github_file(url):
    """Downloads the dataset and policy directly from GitHub"""
    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    
    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        return None

def extract_json_from_text(text):
    """Enhanced JSON extraction with better error handling"""
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

def standardize_result(model_result):
    """Standardize and clean up result fields"""
    required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
    for field in required_fields:
        if field not in model_result:
            model_result[field] = None if field == "compliance_message" else ([] if field == "cargo_items" else 0)
    
    if isinstance(model_result["compliance_result"], str):
        model_result["compliance_result"] = model_result["compliance_result"].lower() == "true"
    
    if not isinstance(model_result["compliance_message"], str):
        model_result["compliance_message"] = str(model_result["compliance_message"] or "No message")
    
    model_result["cargo_items"] = normalize_cargo_items(model_result["cargo_items"]) 

    if isinstance(model_result["fees"], str):
        fees_str = re.sub(r'[^\d.,]', '', model_result["fees"])
        try:
            model_result["fees"] = float(fees_str.replace(',', '.'))
        except ValueError:
            model_result["fees"] = 0
    elif not isinstance(model_result["fees"], (int, float)):
        model_result["fees"] = 0
        
    return model_result

def normalize_cargo_items(items):
    """Normalize cargo items to a consistent list format"""
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
# LUGGAGE POLICY EVALUATION
# =====================================================================

class LuggagePolicyEvaluator:
    """Evaluates luggage policy compliance using a WatsonX LLM"""
    
    def __init__(self):
        # Download the luggage policy
        self.policy_url = "https://github.com/DecisionsDev/policy-corpus/blob/main/luggage/luggage_policy.md"
        self.policy_text = download_github_file(self.policy_url)
        if not self.policy_text:
            raise ValueError("Failed to download luggage policy")
        
        # Initialize the WatsonX LLM
        credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key="YOUR_WATSONX_API_KEY"
        )
        project_id = "a31eb1db-e559-4a08-b0fa-638fdc608777"

        # Configure LLM parameters
        parameters = {
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
            GenParams.MAX_NEW_TOKENS: 750,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0.05,
            GenParams.TOP_K: 10,
            GenParams.TOP_P: 0.9,
            GenParams.REPETITION_PENALTY: 1.1
        }
        
        self.llm = WatsonxLLM(
            model_id="mistralai/mistral-large",
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=project_id,
            params=parameters
        )
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["policy", "travel_class", "age_category", "luggages"],
            template="""You are an expert in airline baggage compliance verification. Your goal is to accurately analyze each case and determine if the luggage complies with airline policies.

BAGGAGE POLICY:
{policy}

CURRENT CASE:
```json
{{
  "travel_class": "{travel_class}",
  "age_category": "{age_category}",
  "luggages": {luggages}
}}
```

STEP-BY-STEP INSTRUCTIONS:

1. ANALYZE ALLOWED BAGGAGE ALLOCATIONS:
   - What basic baggage allowance corresponds to this passenger based on their class ({travel_class}) and age category ({age_category})?
   - What are the specific weight and dimension limits for each type of luggage (carry-on, personal item, checked)?

2. VERIFY EACH PIECE OF LUGGAGE:
   - For each piece of luggage in the provided JSON:
     - Does it comply with the maximum allowed dimensions? (Calculate total volume if necessary)
     - Is it within the corresponding weight limit?
     - Does it require special handling or should it go as cargo?
     - Does it exceed the allowed quantity for its type (carry-on/checked)?

3. CALCULATE APPLICABLE FEES:
   - For each excess identified, apply the exact fee according to policy:
     - Excess weight: How much is charged per additional kg?
     - Additional baggage: What is the fee per extra piece?
     - Oversized baggage: Do special charges apply?
   - Add up all fees to get the total cost.

4. IDENTIFY ITEMS THAT MUST GO AS CARGO:
   - Are there items that exceed absolute limits and must be sent as cargo?
   - List each of these items.

5. DETERMINE THE FINAL RESULT:
   - Based on all the above analysis:
     - Is the baggage generally compliant? (true/false)
     - What is the main reason for non-compliance (if applicable)?
     - What is the total amount of additional fees?

RESPOND USING ONLY THIS JSON FORMAT:
```json
{{
  "compliance_result": true/false,
  "compliance_message": "Brief and precise explanation of compliance or non-compliance",
  "cargo_items": ["item1", "item2", "..."],
  "fees": 0
}}
```

IMPORTANT REQUIREMENTS:
1. "compliance_result" must be a boolean value (true or false)
2. "compliance_message" must clearly explain the result
3. "cargo_items" must be an array (empty if there are no items)
4. "fees" must be an exact number without currency symbols
5. If there are multiple reasons for non-compliance, mention the main one in "compliance_message"

IMPORTANT FORMAT REQUIREMENTS:
1. Your response MUST be in valid JSON format and MUST begin and end with curly braces {{}}
2. DO NOT include any text or explanation outside the JSON object
3. Use literal boolean values (true/false) without quotes
4. "compliance_result" must be exactly true or false (boolean)
5. "compliance_message" must be a string in double quotes
6. "cargo_items" must be an array, even if empty []
7. "fees" must be a number (without quotes or symbols)

NOTE: Apply the policy to the letter to determine compliance and calculate exact fees.
"""
        )
        
        # Create the chain for processing
        self.luggage_chain = self.prompt_template | self.llm
    
    def evaluate_luggage(self, travel_class, age_category, luggages):
        """
        Evaluate luggage compliance based on input parameters
        
        Args:
            travel_class: The travel class (economy, business, first)
            age_category: The age category (adult, child, infant)
            luggages: JSON string containing luggage details
            
        Returns:
            Dictionary with compliance results
        """
        input_data = {
            "policy": self.policy_text,
            "travel_class": travel_class,
            "age_category": age_category,
            "luggages": luggages
        }
        
        # Invoke the LLM
        try:
            result = self.luggage_chain.invoke(input_data)
            
            # Clean markdown and extra text
            cleaned_result = re.sub(r'```json|```|EJEMPLO(:)?|EJEMPLO DE RESPUESTA(:)?', '', result)
            
            # Try to parse JSON directly
            try:
                model_result = json.loads(cleaned_result)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from text
                model_result = extract_json_from_text(result)
                if not model_result or isinstance(model_result, str):
                    return {
                        "error": "Failed to parse result",
                        "raw_response": result
                    }
            
            # Standardize the result
            return standardize_result(model_result)
            
        except Exception as e:
            return {
                "error": f"Error during evaluation: {str(e)}",
                "compliance_result": False,
                "compliance_message": "Analysis failed due to an error",
                "cargo_items": [],
                "fees": 0
            }

# =====================================================================
# COMPARISON FUNCTIONS
# =====================================================================

def compare_with_ground_truth(prediction, ground_truth):
    """
    Compare prediction with ground truth
    
    Args:
        prediction: Dictionary with predicted values
        ground_truth: Dictionary with ground truth values
        
    Returns:
        Dictionary with comparison results
    """
    # Ensure both have the required fields
    for field in ["compliance_result", "compliance_message", "cargo_items", "fees"]:
        if field not in prediction:
            prediction[field] = None if field == "compliance_message" else ([] if field == "cargo_items" else 0)
        if field not in ground_truth:
            ground_truth[field] = None if field == "compliance_message" else ([] if field == "cargo_items" else 0)
    
    # Normalize cargo items
    prediction["cargo_items"] = normalize_cargo_items(prediction["cargo_items"])
    ground_truth["cargo_items"] = normalize_cargo_items(ground_truth["cargo_items"])
    
    # Convert fees to float
    try:
        prediction["fees"] = float(prediction["fees"])
    except (ValueError, TypeError):
        prediction["fees"] = 0
    
    try:
        ground_truth["fees"] = float(ground_truth["fees"])
    except (ValueError, TypeError):
        ground_truth["fees"] = 0
    
    # Calculate metrics
    compliance_correct = prediction["compliance_result"] == ground_truth["compliance_result"]
    
    # Calculate cargo items accuracy using Jaccard similarity
    pred_items = set([str(item) for item in prediction["cargo_items"]])
    true_items = set([str(item) for item in ground_truth["cargo_items"]])
    
    union = len(pred_items.union(true_items))
    intersection = len(pred_items.intersection(true_items))
    
    cargo_accuracy = intersection / union if union > 0 else 1.0
    
    # Calculate fee accuracy
    fee_diff = abs(prediction["fees"] - ground_truth["fees"])
    fee_exact = fee_diff < 0.01
    fee_within_5 = fee_diff <= 5
    fee_within_10 = fee_diff <= 10
    
    return {
        "compliance_correct": compliance_correct,
        "cargo_accuracy": cargo_accuracy * 100,
        "fee_exact_match": fee_exact,
        "fee_within_5": fee_within_5,
        "fee_within_10": fee_within_10,
        "fee_difference": fee_diff
    }

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def process_luggage_case(travel_class, age_category, luggages, compare_with=None):
    """
    Process a luggage case and optionally compare with ground truth
    
    Args:
        travel_class: The travel class (economy, business, first)
        age_category: The age category (adult, child, infant)
        luggages: JSON string containing luggage details
        compare_with: Optional ground truth to compare with
        
    Returns:
        Dictionary with results and comparison
    """
    # Initialize the evaluator
    evaluator = LuggagePolicyEvaluator()
    
    # Get prediction
    prediction = evaluator.evaluate_luggage(travel_class, age_category, luggages)
    
    result = {
        "prediction": prediction
    }
    
    # Compare with ground truth if provided
    if compare_with:
        comparison = compare_with_ground_truth(prediction, compare_with)
        result["comparison"] = comparison
        result["ground_truth"] = compare_with
    
    return result

# =====================================================================
# DATA LOADING FUNCTION
# =====================================================================

def load_test_data(url, sample_size=5):
    """
    Load test data from GitHub
    
    Args:
        url: URL to the test data CSV
        sample_size: Number of samples to load
        
    Returns:
        DataFrame with test data
    """
    content = download_github_file(url)
    if not content:
        raise ValueError(f"Could not download test data from {url}")
    
    df = pd.read_csv(StringIO(content))
    return df.sample(n=sample_size, random_state=42)

# =====================================================================
# EXAMPLE USAGE
# =====================================================================

def run_example():
    """Run an example to demonstrate the code"""
    # Load a few test cases
    test_url = "https://github.com/DecisionsDev/policy-corpus/blob/main/luggage/luggage_compliance/luggage_policy_test_dataset_1K.csv"
    test_data = load_test_data(test_url, sample_size=2)
    
    for _, row in test_data.iterrows():
        print("\n" + "="*80)
        print(f"Testing case with travel class: {row['travel_class']}, age category: {row['age_category']}")
        
        # Process the case
        result = process_luggage_case(
            travel_class=row['travel_class'],
            age_category=row['age_category'],
            luggages=row['luggages'],
            compare_with={
                "compliance_result": row['compliance_result'],
                "compliance_message": row['compliance_message'],
                "cargo_items": row['cargo_items'],
                "fees": row['fees']
            }
        )
        
        # Print results
        print("\nPrediction:")
        print(f"- Compliance: {result['prediction']['compliance_result']}")
        print(f"- Message: {result['prediction']['compliance_message']}")
        print(f"- Cargo items: {result['prediction']['cargo_items']}")
        print(f"- Fees: ${result['prediction']['fees']}")
        
        print("\nGround Truth:")
        print(f"- Compliance: {result['ground_truth']['compliance_result']}")
        print(f"- Message: {result['ground_truth']['compliance_message']}")
        print(f"- Cargo items: {result['ground_truth']['cargo_items']}")
        print(f"- Fees: ${result['ground_truth']['fees']}")
        
        print("\nComparison:")
        print(f"- Compliance correct: {result['comparison']['compliance_correct']}")
        print(f"- Cargo accuracy: {result['comparison']['cargo_accuracy']:.2f}%")
        print(f"- Fee exact match: {result['comparison']['fee_exact_match']}")
        print(f"- Fee within $5: {result['comparison']['fee_within_5']}")
        print(f"- Fee within $10: {result['comparison']['fee_within_10']}")
        print(f"- Fee difference: ${result['comparison']['fee_difference']:.2f}")

# =====================================================================
# CUSTOM FUNCTION TO PROCESS USER INPUT
# =====================================================================

def process_custom_input(travel_class, age_category, luggages_json):
    """
    Process custom input from a user
    
    Args:
        travel_class: The travel class (economy, business, first)
        age_category: The age category (adult, child, infant)
        luggages_json: JSON string containing luggage details
        
    Returns:
        Dictionary with prediction results
    """
    # Clean the luggages JSON to remove any "excess" or "compliance" fields
    try:
        luggages = json.loads(luggages_json)
        
        # If it's a list, process each item
        if isinstance(luggages, list):
            for item in luggages:
                if isinstance(item, dict):
                    # Remove unwanted fields
                    if "excess" in item:
                        del item["excess"]
                    if "compliance" in item:
                        del item["compliance"]
        
        # Convert back to JSON string
        clean_luggages_json = json.dumps(luggages)
    except json.JSONDecodeError:
        # If not valid JSON, just use as is
        clean_luggages_json = luggages_json
    
    # Process the case
    result = process_luggage_case(travel_class, age_category, clean_luggages_json)
    
    return result["prediction"]

# =====================================================================
# BATCH PROCESSING FUNCTION
# =====================================================================

def calculate_enhanced_metrics(results):
    """
    Calculate enhanced metrics for model evaluation
    Returns detailed metrics dictionary similar to the original code
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to analyze"}
    
    # Extract actual and predicted values
    actual_compliance = [r["ground_truth"]["compliance_result"] for r in results]
    predicted_compliance = [r["prediction"]["compliance_result"] for r in results]
    
    # Convert to float to handle any string values
    actual_fees = []
    predicted_fees = []
    for r in results:
        try:
            actual_fees.append(float(r["ground_truth"]["fees"]))
        except (ValueError, TypeError):
            actual_fees.append(0)
            
        try:
            predicted_fees.append(float(r["prediction"]["fees"]))
        except (ValueError, TypeError):
            predicted_fees.append(0)
    
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
        predicted_items = normalize_cargo_items(r["prediction"]["cargo_items"])
        actual_items = normalize_cargo_items(r["ground_truth"]["cargo_items"])
        
        # Convert to sets of strings for comparison
        pred_set = set(str(item) for item in predicted_items)
        actual_set = set(str(item) for item in actual_items)
        
        union = len(pred_set.union(actual_set))
        intersection = len(pred_set.intersection(actual_set))
        
        jaccard = intersection / union if union > 0 else 1.0
        cargo_match.append(jaccard)
    
    avg_cargo_accuracy = sum(cargo_match) / len(cargo_match) * 100 if cargo_match else 0
    
    # Return comprehensive metrics dictionary
    return {
        "compliance": {
            "accuracy": compliance_accuracy,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100,
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
    print(f"Precision: {metrics['compliance']['precision']:.2f}%")
    print(f"Recall: {metrics['compliance']['recall']:.2f}%")
    print(f"F1 Score: {metrics['compliance']['f1_score']:.2f}%")
    
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

def analyze_by_travel_class(results, test_data):
    """
    Analyze performance by travel class
    """
    metrics_by_class = {}
    
    for i, r in enumerate(results):
        row_index = r["row_index"]
        travel_class = test_data.iloc[row_index]['travel_class']
        is_correct = r["prediction"]["compliance_result"] == r["ground_truth"]["compliance_result"]
        
        if travel_class not in metrics_by_class:
            metrics_by_class[travel_class] = {"correct": 0, "total": 0}
        
        metrics_by_class[travel_class]["total"] += 1
        if is_correct:
            metrics_by_class[travel_class]["correct"] += 1
    
    print("\nPerformance by Travel Class:")
    for cls, data in metrics_by_class.items():
        accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"{cls}: {accuracy:.2f}% ({data['correct']}/{data['total']})")

def batch_process_test_data(sample_size=200):
    """
    Process multiple test cases and generate summary metrics
    
    Args:
        sample_size: Number of samples to process
        
    Returns:
        Dictionary with overall metrics
    """
    print(f"Processing {sample_size} random cases from the dataset...")
    
    # Load test data
    test_url = "https://github.com/DecisionsDev/policy-corpus/blob/main/luggage/luggage_compliance/luggage_policy_test_dataset_1K.csv"
    test_content = download_github_file(test_url)
    if not test_content:
        print("Error: Could not download test dataset.")
        return {"error": "Could not download test dataset"}
    
    # Load as DataFrame
    df = pd.read_csv(StringIO(test_content))
    
    # Sample random cases
    test_data = df.sample(n=sample_size, random_state=42)
    
    # Initialize the evaluator
    evaluator = LuggagePolicyEvaluator()
    
    results = []
    
    # Create directory for storing raw responses if it doesn't exist
    import os
    if not os.path.exists("raw_responses"):
        os.makedirs("raw_responses")
    
    # Process each case
    for index, row in enumerate(test_data.iterrows()):
        row_index, row_data = row
        
        print(f"\n--- Processing case {index + 1}/{sample_size} ---")
        
        # Clean luggage data to remove excess and compliance fields if they exist
        try:
            luggages = json.loads(row_data['luggages'])
            if isinstance(luggages, list):
                for item in luggages:
                    if isinstance(item, dict):
                        if "excess" in item:
                            del item["excess"]
                        if "compliance" in item:
                            del item["compliance"]
            clean_luggages = json.dumps(luggages)
        except:
            clean_luggages = row_data['luggages']
        
        # Get prediction from LLM
        try:
            prediction = evaluator.evaluate_luggage(
                travel_class=row_data['travel_class'],
                age_category=row_data['age_category'],
                luggages=clean_luggages
            )
            
            # Save raw response
            with open(f"raw_responses/response_{index + 1}.txt", "w", encoding="utf-8") as f:
                f.write(str(prediction))
            
            # Verify required fields
            required_fields = ["compliance_result", "compliance_message", "cargo_items", "fees"]
            if not all(field in prediction for field in required_fields):
                for field in required_fields:
                    if field not in prediction:
                        prediction[field] = None if field == "compliance_message" else ([] if field == "cargo_items" else 0)
            
            # Add to results
            ground_truth = {
                "compliance_result": row_data['compliance_result'],
                "compliance_message": row_data['compliance_message'],
                "cargo_items": normalize_cargo_items(row_data['cargo_items']),
                "fees": row_data['fees']
            }
            
            results.append({
                "prediction": prediction,
                "ground_truth": ground_truth,
                "row_index": row_index
            })
            
            # Print brief summary
            print(f"Predicted: {prediction['compliance_result']}, Actual: {ground_truth['compliance_result']}")
            
        except Exception as e:
            print(f"Error processing case {index + 1}: {str(e)}")
            
            # Add default result for errors
            results.append({
                "prediction": {
                    "compliance_result": False,
                    "compliance_message": "Error in processing",
                    "cargo_items": [],
                    "fees": 0
                },
                "ground_truth": {
                    "compliance_result": row_data['compliance_result'],
                    "compliance_message": row_data['compliance_message'],
                    "cargo_items": normalize_cargo_items(row_data['cargo_items']),
                    "fees": row_data['fees']
                },
                "row_index": row_index,
                "error": str(e)
            })
    
    # Calculate detailed metrics
    print("\nCalculating performance metrics...")
    enhanced_metrics = calculate_enhanced_metrics(results)
    print_metrics_summary(enhanced_metrics)
    
    # Analyze by travel class
    analyze_by_travel_class(results, test_data)
    
    # Analyze errors
    error_cases = [r for r in results if r["prediction"]["compliance_result"] != r["ground_truth"]["compliance_result"]]
    if error_cases:
        print("\nCommon Error Analysis:")
        error_messages = [r["prediction"]["compliance_message"] for r in error_cases]
        
        # Extract keywords from error messages
        all_words = []
        for msg in error_messages:
            if isinstance(msg, str):
                words = re.findall(r'\b\w{4,}\b', msg.lower())
                all_words.extend(words)
        
        # Count most frequent words
        from collections import Counter
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(5)
        
        print("Most common terms in error messages:")
        for word, count in most_common:
            print(f"  {word}: {count} occurrences")
    
    # Save metrics to file
    try:
        with open('luggage_metrics_summary.json', 'w', encoding="utf-8") as f:
            json.dump(enhanced_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed metrics saved to 'luggage_metrics_summary.json'")
    except Exception as e:
        print(f"Error saving metrics to file: {str(e)}")
    
    return enhanced_metrics

if __name__ == "__main__":
    # Process 200 random samples from the dataset and calculate metrics
    print("Luggage Compliance Predictor")
    print("Processing 200 random cases from the dataset...")
    batch_process_test_data(sample_size=200)