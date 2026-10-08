import os
import json
import pandas as pd
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, f1_score
import time

# Read the luggage policy
with open('policy-corpus/luggage/luggage_policy.txt', 'r') as file:
    luggage_policy = file.read()

# Read the test dataset
df = pd.read_csv('policy-corpus/luggage/luggage_compliance/luggage_policy_test_dataset_100.csv')

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="YOUR_WATSONX_API_KEY"
)
project_id = "a31eb1db-e559-4a08-b0fa-638fdc608777"

api_client = APIClient(credentials=credentials, project_id=project_id)
model_id_1 = "meta-llama/llama-3-3-70b-instruct"

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE.value,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

llm1 = WatsonxLLM(
    model_id=model_id_1,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params=parameters
)

# Enhanced prompt for analyzing luggage compliance
luggage_prompt = PromptTemplate(
    input_variables=["policy", "travel_class", "age_category", "luggages"],
    template="""Based on the following airline luggage policy:

{policy}

Please analyze if the following luggage is compliant and calculate any excess fees:
Travel Class: {travel_class}
Age Category: {age_category}
Luggage Details: {luggages}

Return your analysis in the following JSON format:
{{
    "compliance_result": true/false,
    "compliance_message": "Brief explanation of compliance status",
    "cargo_items": [],  // List of items that must be shipped as cargo
    "fees": 0  // Total fees in currency units
}}"""
)

# Create the processing chain
luggage_chain = luggage_prompt | llm1

# Process each entry in the dataset
results = []
for index, row in df.iterrows():
    # Prepare input for the chain
    input_data = {
        "policy": luggage_policy,
        "travel_class": row['travel_class'],
        "age_category": row['age_category'],
        "luggages": row['luggages']
    }
    
    # Retry logic
    for attempt in range(3):  # Try up to 3 times
        try:
            # Get analysis from the model
            result = luggage_chain.invoke(input_data)
            
            # Print the raw result for debugging
            print(f"Raw result for index {index}: {result}")
            
            # Parse the result and compare with ground truth
            model_result = json.loads(result)
            results.append({
                'predicted': model_result,
                'actual': {
                    'compliance_result': row['compliance_result'],
                    'compliance_message': row['compliance_message'],
                    'cargo_items': row['cargo_items'] if pd.notna(row['cargo_items']) else [],
                    'fees': row['fees']
                }
            })
            break  # Exit the retry loop if successful
        except json.JSONDecodeError:
            print(f"Error parsing result for index {index}: {result}")  # Log the raw result
            break  # Exit the retry loop on parsing error
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)  # Wait before retrying

# Calculate accuracy metrics
total = len(results)

if total > 0:
    compliance_correct = sum(1 for r in results if r['predicted']['compliance_result'] == r['actual']['compliance_result'])
    fees_correct = sum(1 for r in results if abs(r['predicted']['fees'] - r['actual']['fees']) < 1)

    compliance_accuracy = compliance_correct / total
    fees_accuracy = fees_correct / total
    f1 = f1_score([r['actual']['compliance_result'] for r in results], [r['predicted']['compliance_result'] for r in results], average='binary')

    print("\nAccuracy Metrics:")
    print(f"Compliance Accuracy: {compliance_accuracy * 100:.2f}%")
    print(f"Fees Accuracy: {fees_accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
else:
    print("No results were processed. Cannot calculate accuracy metrics.")

# Save detailed results to a file
with open('luggage_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)