import json
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RAG_MAX_DF, RAG_MIN_DF, RAG_NGRAM_RANGE, RAG_TOP_K


class LuggageRAG:
    def __init__(self, dataset_path: str, top_k: int = RAG_TOP_K):
        self.top_k = top_k
        self.df = pd.read_csv(dataset_path)
        self.df["text_representation"] = self.df.apply(self._to_text, axis=1)
        self.vectorizer = TfidfVectorizer(
            ngram_range=RAG_NGRAM_RANGE, min_df=RAG_MIN_DF, max_df=RAG_MAX_DF
        )
        self.vectors = self.vectorizer.fit_transform(self.df["text_representation"])

    @staticmethod
    def _to_text(row: pd.Series) -> str:
        tokens = [f"class_{row['travel_class']}", f"age_{row['age_category']}"]
        if pd.notna(row["luggages"]):
            try:
                for item in json.loads(row["luggages"]):
                    tokens.append(
                        f"item_{item['storage']}_w{item['weight']}_h{item['height']}"
                    )
            except (json.JSONDecodeError, KeyError):
                tokens.append("luggage_error")
        return " ".join(tokens)

    def get_similar_cases(
        self, row: pd.Series, exclude_index: Optional[int] = None
    ) -> list[dict]:
        query_vec = self.vectorizer.transform([self._to_text(row)])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        ranked = similarities.argsort()[::-1]
        if exclude_index is not None:
            ranked = [i for i in ranked if i != exclude_index]
        indices = ranked[1 : self.top_k + 1]
        return [self._case_details(i) for i in indices]

    def _case_details(self, index: int) -> dict:
        case = self.df.iloc[index]
        return {
            "travel_class": case["travel_class"],
            "age_category": case["age_category"],
            "luggages": case["luggages"],
            "compliance_result": case["compliance_result"],
            "compliance_message": case["compliance_message"],
            "fees": float(case["fees"]) if pd.notna(case["fees"]) else 0.0,
        }
