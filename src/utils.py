import json
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

from src.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    MODEL_ID,
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)


def build_llm(
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> WatsonxLLM:
    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    parameters = {
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.TEMPERATURE: temperature,
    }
    return WatsonxLLM(
        model_id=MODEL_ID,
        url=credentials["url"],
        apikey=credentials["api_key"],
        project_id=WATSONX_PROJECT_ID,
        params=parameters,
    )


def extract_json(text: str) -> Optional[dict]:
    try:
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        json_str = text[start : end + 1]
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r"\\", "", json_str)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pattern = r'(\{.*"results"\s*:\s*\[.*\]\s*\})'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def validate_prediction(json_data: dict) -> dict:
    required_fields = ["compliance_result", "compliance_message", "fees", "cargo_items"]
    for field in required_fields:
        if field not in json_data:
            json_data[field] = None

    json_data["compliance_result"] = bool(json_data.get("compliance_result", False))
    json_data["compliance_message"] = str(json_data.get("compliance_message", ""))
    try:
        json_data["fees"] = float(json_data.get("fees", 0))
    except (ValueError, TypeError):
        json_data["fees"] = 0
    if not isinstance(json_data["cargo_items"], list):
        json_data["cargo_items"] = []

    return json_data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)


def init_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["predicted_compliance_result"] = None
    result["predicted_compliance_message"] = None
    result["predicted_cargo_items"] = None
    result["predicted_fees"] = None
    return result


def store_prediction(df: pd.DataFrame, idx: int, prediction: dict) -> None:
    df.at[idx, "predicted_compliance_result"] = prediction["compliance_result"]
    df.at[idx, "predicted_compliance_message"] = prediction["compliance_message"]
    df.at[idx, "predicted_cargo_items"] = str(prediction["cargo_items"])
    df.at[idx, "predicted_fees"] = prediction["fees"]


def invoke_with_retries(
    llm: WatsonxLLM, messages: list, max_retries: int = 3
) -> Optional[str]:
    for _ in range(max_retries):
        try:
            response = llm.invoke(messages)
            if response and response.strip():
                return response
        except Exception:
            continue
    return None


def parse_cargo_items(cargo_data: Any) -> list:
    if pd.isnull(cargo_data) or cargo_data == "" or cargo_data == "[]":
        return []
    try:
        parsed = (
            json.loads(cargo_data.replace("'", '"'))
            if isinstance(cargo_data, str)
            else cargo_data
        )
        if isinstance(parsed, list):
            return [
                item["storage"]
                for item in parsed
                if isinstance(item, dict) and "storage" in item
            ]
        return []
    except (json.JSONDecodeError, ValueError):
        return []
