SYSTEM_JSON_ONLY = (
    "You are a luggage compliance evaluation expert.\n"
    "You must strictly respond ONLY with a valid JSON object, "
    "with no markdown formatting, no explanations, no extra text.\n"
    "Format:\n"
    "{\n"
    '  "compliance_result": true/false,\n'
    '  "compliance_message": "Brief explanation",\n'
    '  "fees": 0,\n'
    '  "cargo_items": []\n'
    "}"
)

SYSTEM_BATCH = (
    "Analyze each passenger according to the policy and return a JSON.\n\n"
    "For compliance_message, provide detailed explanations:\n"
    '- If compliant: "Complies with policy"\n'
    '- If weight exceeded: "Weight limit exceeded - baggage weighs X kg, limit is Y kg"\n'
    '- If size exceeded: "Size limit exceeded - dimensions X cm, limit is Y cm"\n'
    '- If too many bags: "Quantity limit exceeded - X bags carried, limit is Y bags"\n'
    '- If cargo required: "Cargo shipment required"\n\n'
    "Format:\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "client_id": 1,\n'
    '      "compliance_result": true,\n'
    '      "compliance_message": "detailed explanation",\n'
    '      "fees": 0,\n'
    '      "cargo_items": []\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Respond ONLY with JSON."
)

SYSTEM_CHAIN_OF_THOUGHT = (
    "For each passenger, explain briefly your reasoning step by step, "
    "THEN return ONLY the following JSON.\n"
    "Format:\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "client_id": 1,\n'
    '      "compliance_result": true,\n'
    '      "compliance_message": "Compliant",\n'
    '      "fees": 0,\n'
    '      "cargo_items": []\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "NO TEXT OUTSIDE THE JSON."
)

SYSTEM_CRITICAL_ANALYSIS = (
    "For each passenger, critically analyze all aspects of the case "
    "and justify the decision, THEN return ONLY the following JSON.\n"
    "Format:\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "client_id": 1,\n'
    '      "compliance_result": true,\n'
    '      "compliance_message": "Compliant",\n'
    '      "fees": 0,\n'
    '      "cargo_items": []\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "NO TEXT OUTSIDE THE JSON."
)

SYSTEM_RAG = (
    "You are a luggage compliance evaluation expert.\n"
    "Analyze passenger luggage according to the airline policy and similar cases provided.\n"
    "Respond ONLY with a valid JSON object, no markdown formatting, no explanations.\n\n"
    "Format:\n"
    "{\n"
    '  "compliance_result": true/false,\n'
    '  "compliance_message": "Brief explanation",\n'
    '  "fees": 0,\n'
    '  "cargo_items": []\n'
    "}"
)

POLICY_UNDERSTANDING = (
    "You are an expert in airline baggage compliance.\n\n"
    "BAGGAGE POLICY:\n{policy}\n\n"
    "Please read this policy carefully and confirm your understanding of the main rules regarding:\n"
    "- Weight and dimension limits\n"
    "- Number of bags allowed per class\n"
    "- Fees for overages\n"
    "- Items requiring cargo handling\n\n"
    "Respond by briefly explaining the key points you understood."
)

TEST_CASE_TEMPLATE = (
    "Analyze this case according to the policy:\n\n"
    "Travel Class: {travel_class}\n"
    "Age Category: {age_category}\n"
    "Luggage: {luggages}\n\n"
    "Determine:\n"
    "1. Compliance (yes/no)\n"
    "2. Applicable fees\n"
    "3. Required cargo items\n"
    "4. Explanation message\n\n"
    "JSON format:\n"
    "{{\n"
    '  "compliance_result": true/false,\n'
    '  "fees": 0,\n'
    '  "cargo_items": [],\n'
    '  "compliance_message": "explanation"\n'
    "}}\n\n"
    "Analyze this case and explain your reasoning."
)

QUESTIONS_PROMPT = (
    "After analyzing these examples, do you have any specific questions about the "
    "BAGGAGE POLICY to ensure proper interpretation?\n\n"
    "You can ask up to 3 questions only about:\n"
    "- Policy rules that are unclear\n"
    "- Edge cases not covered in the policy\n"
    "- Clarifications on fee application\n"
    "- Specifications on cargo criteria\n\n"
    "Do NOT ask questions about previous examples, only about the policy itself."
)

FSL_EXAMPLES = """
PASSENGER:
Client 101: Travel Class=Business, Age Category=adult, Luggage=[{"storage": "personal", "weight": 2.3, "height": 25.0, "width": 35.0, "depth": 15.0, "unit": "cm"}]
JSON:
{
  "client_id": 101,
  "compliance_result": true,
  "compliance_message": "Compliant",
  "fees": 0,
  "cargo_items": []
}

PASSENGER:
Client 102: Travel Class=Economy, Age Category=infant, Luggage=[{"storage": "checked", "weight": 43.91, "height": 55.0, "width": 35.0, "depth": 25.0, "unit": "cm"}]
JSON:
{
  "client_id": 102,
  "compliance_result": false,
  "compliance_message": "Bag 1: weight 43.91kg exceeds 32kg limit",
  "fees": 100,
  "cargo_items": [{"storage": "checked", "weight": 43.91, "height": 55.0, "width": 35.0, "depth": 25.0, "unit": "cm"}]
}

PASSENGER:
Client 103: Travel Class=Business, Age Category=child, Luggage=[{"storage": "carry-on", "weight": 8.0, "height": 50.0, "width": 35.0, "depth": 20.0, "unit": "cm"}, {"storage": "carry-on", "weight": 7.5, "height": 55.0, "width": 40.0, "depth": 23.0, "unit": "cm"}, {"storage": "personal", "weight": 2.0, "height": 25.0, "width": 35.0, "depth": 15.0, "unit": "cm"}, {"storage": "personal", "weight": 1.5, "height": 22.0, "width": 33.0, "depth": 13.0, "unit": "cm"}, {"storage": "carry-on", "weight": 6.0, "height": 53.0, "width": 38.0, "depth": 21.0, "unit": "cm"}]
JSON:
{
  "client_id": 103,
  "compliance_result": false,
  "compliance_message": "Exceeded carry-on quantity allowance",
  "fees": 150,
  "cargo_items": []
}

PASSENGER:
Client 104: Travel Class=First, Age Category=infant, Luggage=[{"storage": "personal", "weight": 2.5, "height": 25.0, "width": 35.0, "depth": 15.0, "unit": "cm"}]
JSON:
{
  "client_id": 104,
  "compliance_result": false,
  "compliance_message": "Infants are not allowed personal items",
  "fees": 0,
  "cargo_items": []
}
"""
