import json
import re
from tqdm import tqdm
import os

# Definition of error categories and their associated keywords
ERROR_CATEGORIES = {
    "EXCESSIVE_WEIGHT": [
        "weight limit", "exceeded", "overweight", "too heavy", "kg", "kilos", 
        "exceeds weight", "heavier", "too much weight", "over the limit"
    ],
    "EXCESSIVE_SIZE": [
        "size limit", "too big", "dimension", "oversized", "exceed", "cm",
        "width", "height", "depth", "length", "size", "linear dimensions"
    ],
    "EXCESSIVE_QUANTITY": [
        "too many", "quantity", "allowance", "multiple", "several", "additional", 
        "extra", "excess", "baggage count", "number of bags", "multiple bags"
    ],
    "CARGO_REQUIRED": [
        "cargo", "ship", "shipped", "must be sent", "cannot be carried"
    ],
    "COMPLIANT": [
        "comply", "complies", "allowable", "within limits", "acceptable",
        "policy", "rule", "approved", "permitted", "allowed"
    ]
}

def extract_categories(message):
    """
    Extracts error categories from a compliance message by searching for keywords.
    
    Args:
        message (str): The compliance message to analyze
        
    Returns:
        set: Set of categories found in the message
    """
    if not message:
        return set()
    
    # Convert to lowercase for case-insensitive search
    message_lower = message.lower()
    
    # Set to store found categories
    found_categories = set()
    
    # Search for keywords for each category
    for category, keywords in ERROR_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in message_lower:
                found_categories.add(category)
                break  # Move to the next category once a keyword is found
    
    # If the message is short and no category is found, try to detect compliance
    if not found_categories and len(message) < 50:
        if "comply" in message_lower or "all" in message_lower:
            found_categories.add("COMPLIANT")
    
    # Special case: if "non" or "not" precedes a compliance term, it's non-compliance
    if "COMPLIANT" in found_categories:
        negation_patterns = ["not", "doesn't", "does not", "no", "non"]
        for pattern in negation_patterns:
            if pattern in message_lower:
                found_categories.remove("COMPLIANT")
                # Add a generic non-compliance category
                found_categories.add("NON_COMPLIANT")
                break
    
    return found_categories

def compare_messages(actual_msg, predicted_msg):
    """
    Compares two compliance messages to determine if they express the same error category.
    Considers the predicted message to be correct if it identifies at least one of the
    categories present in the reference message.
    
    Args:
        actual_msg (str): The reference message
        predicted_msg (str): The message to evaluate
        
    Returns:
        bool: True if the evaluated message captures at least one of the categories of the reference message
    """
    # Extract categories from both messages
    actual_categories = extract_categories(actual_msg)
    predicted_categories = extract_categories(predicted_msg)
    
    # If both messages are empty, consider them identical
    if not actual_msg and not predicted_msg:
        return True
    
    # If one of the messages is empty, consider them different
    if not actual_msg or not predicted_msg:
        return False
    
    # Check if at least one category is common
    common_categories = actual_categories.intersection(predicted_categories)
    
    # If at least one category is common, consider that the messages express the same error
    if common_categories:
        return True
    
    # Special case: "NON_COMPLIANT" is considered a match with any error category
    # because it is a generic category
    if "NON_COMPLIANT" in predicted_categories and actual_categories - {"COMPLIANT"}:
        return True
    
    # If the reference message indicates compliance and the predicted message too, it's correct
    if "COMPLIANT" in actual_categories and "COMPLIANT" in predicted_categories:
        return True
    
    # Otherwise, consider that the messages express different errors
    return False

def check_compliance_match(actual_result, predicted_result):
    """
    Checks if the compliance results (true/false) match.
    
    Args:
        actual_result (bool): The reference compliance result
        predicted_result (bool): The predicted compliance result
        
    Returns:
        bool: True if the results match, False otherwise
    """
    # If one of the results is None, consider they do not match
    if actual_result is None or predicted_result is None:
        return False
    
    # Otherwise, check if the results are identical
    return actual_result == predicted_result

def analyze_compliance_messages(json_file_path, output_file=None, sample_size=None):
    """
    Analyzes compliance messages and calculates the agreement metric using
    a keyword-based approach.
    
    Args:
        json_file_path (str): Path to the JSON file containing the results
        output_file (str, optional): Path to save the analysis results
        sample_size (int, optional): Size of the sample to analyze
        
    Returns:
        dict: Analysis results
    """
    # Load the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"JSON file loaded: {len(results)} results found")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None
    
    # Limit the sample if necessary
    if sample_size and sample_size < len(results):
        print(f"Using a sample of {sample_size} results")
        results = results[:sample_size]
    
    # Analyze each pair of messages
    comparison_results = []
    
    for result in tqdm(results, desc="Comparing messages"):
        actual = result.get("actual", {})
        predicted = result.get("predicted", {})
        
        actual_msg = actual.get("compliance_message", "")
        predicted_msg = predicted.get("compliance_message", "")
        
        actual_compliance = actual.get("compliance_result", None)
        predicted_compliance = predicted.get("compliance_result", None)
        
        # Check the correspondence of messages and compliance results
        message_match = compare_messages(actual_msg, predicted_msg)
        compliance_match = check_compliance_match(actual_compliance, predicted_compliance)
        
        # Extract categories for the report
        actual_categories = extract_categories(actual_msg)
        predicted_categories = extract_categories(predicted_msg)
        
        # Add the results
        comparison_results.append({
            "actual_message": actual_msg,
            "predicted_message": predicted_msg,
            "actual_compliance": actual_compliance,
            "predicted_compliance": predicted_compliance,
            "message_match": message_match,
            "compliance_match": compliance_match,
            "actual_categories": list(actual_categories),
            "predicted_categories": list(predicted_categories)
        })
    
    # Calculate metrics
    valid_comparisons = [r for r in comparison_results if r["actual_message"] and r["predicted_message"]]
    
    if valid_comparisons:
        message_agreement_rate = sum(1 for r in valid_comparisons if r["message_match"]) / len(valid_comparisons) * 100
        compliance_agreement_rate = sum(1 for r in valid_comparisons if r["compliance_match"]) / len(valid_comparisons) * 100
    else:
        message_agreement_rate = 0
        compliance_agreement_rate = 0
    
    # Calculate statistics by category
    category_stats = {}
    for category in ERROR_CATEGORIES.keys():
        category_cases = [r for r in valid_comparisons if category in r["actual_categories"]]
        if category_cases:
            correct_predictions = sum(1 for r in category_cases if category in r["predicted_categories"])
            category_stats[category] = {
                "count": len(category_cases),
                "correct": correct_predictions,
                "accuracy": (correct_predictions / len(category_cases)) * 100
            }
    
    # Display the results
    print(f"\nMessage agreement rate: {message_agreement_rate:.2f}%")
    print(f"Compliance result agreement rate: {compliance_agreement_rate:.2f}%")
    print(f"Total valid comparisons: {len(valid_comparisons)}")
    
    print("\nStatistics by category:")
    for category, stats in category_stats.items():
        print(f"- {category}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['count']})")
    
    # Save the results if requested
    if output_file:
        output_data = {
            "message_agreement_rate": message_agreement_rate,
            "compliance_agreement_rate": compliance_agreement_rate,
            "total_valid_comparisons": len(valid_comparisons),
            "category_stats": category_stats,
            "results": comparison_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    return {
        "message_agreement_rate": message_agreement_rate,
        "compliance_agreement_rate": compliance_agreement_rate,
        "category_stats": category_stats,
        "results": comparison_results
    }

def main():
    # Configuration
    json_file_path = "results/evaluation_results.json"  # Path to JSON file
    output_file = "compliance_comparison_results.json"  # Output file
    sample_size = None  # Number of samples to analyze (None for all)
    
    # Analyze messages with the keyword approach
    analyze_compliance_messages(json_file_path, output_file, sample_size)

if __name__ == "__main__":
    main()