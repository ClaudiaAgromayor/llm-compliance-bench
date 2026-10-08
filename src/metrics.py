import ast
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import JUDGE_MAX_NEW_TOKENS, JUDGE_TEMPERATURE
from src.utils import build_llm, parse_cargo_items


class LLMJudge:
    def __init__(self):
        self.model = build_llm(
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )

    def messages_match(self, actual: str, predicted: str) -> bool:
        if not actual and not predicted:
            return True
        if not actual or not predicted:
            return False
        actual, predicted = actual.strip(), predicted.strip()
        if actual.lower() == predicted.lower():
            return True

        prompt = (
            "Analyze these baggage compliance messages:\n\n"
            f'EXPECTED REASON: "{actual}"\n'
            f'PREDICTED RESPONSE: "{predicted}"\n\n'
            "Task: Determine if the EXPECTED REASON is covered in the PREDICTED RESPONSE.\n\n"
            "Rules:\n"
            "1. The expected reason must be identifiable in the predicted response\n"
            "2. Different wording for the same concept is acceptable\n"
            "3. Additional reasons in the predicted response are acceptable\n"
            "4. Focus on core compliance issues (weight, size, quantity, prohibited items)\n\n"
            "Answer only: MATCH or NO_MATCH"
        )
        try:
            response = self.model.invoke(prompt).strip().upper()
            return "MATCH" in response
        except Exception:
            return self._keyword_fallback(actual, predicted)

    @staticmethod
    def _keyword_fallback(actual: str, predicted: str) -> bool:
        actual_words = set(actual.lower().split())
        predicted_words = set(predicted.lower().split())
        key_terms = {
            "weight", "size", "limit", "exceed", "dimensions",
            "heavy", "large", "prohibited", "class",
        }
        return bool(actual_words.intersection(predicted_words).intersection(key_terms))


class MetricsCalculator:
    def __init__(self, csv_path: str, use_llm_judge: bool = True):
        self.df = pd.read_csv(csv_path)
        self.judge = LLMJudge() if use_llm_judge else None
        self._prepare()

    def _prepare(self) -> None:
        self.df["cargo_items"] = self.df["cargo_items"].apply(parse_cargo_items)
        self.df["predicted_cargo_items"] = self.df["predicted_cargo_items"].apply(
            parse_cargo_items
        )
        self.df = self.df.fillna(
            {
                "compliance_result": False,
                "predicted_compliance_result": False,
                "fees": 0,
                "predicted_fees": 0,
                "compliance_message": "",
                "predicted_compliance_message": "",
            }
        )
        self.compliance_true = self.df["compliance_result"].astype(bool).tolist()
        self.compliance_pred = self.df["predicted_compliance_result"].astype(bool).tolist()
        self.fees_true = self.df["fees"].astype(float).tolist()
        self.fees_pred = self.df["predicted_fees"].astype(float).tolist()
        self.message_true = self.df["compliance_message"].astype(str).tolist()
        self.message_pred = self.df["predicted_compliance_message"].astype(str).tolist()

        self.mlb = MultiLabelBinarizer()
        all_cargo = self.df["cargo_items"].tolist() + self.df["predicted_cargo_items"].tolist()
        self.mlb.fit(all_cargo)
        self.cargo_true_enc = self.mlb.transform(self.df["cargo_items"].tolist())
        self.cargo_pred_enc = self.mlb.transform(self.df["predicted_cargo_items"].tolist())

    def compliance_metrics(self) -> dict:
        return {
            "accuracy": accuracy_score(self.compliance_true, self.compliance_pred),
            "precision": precision_score(self.compliance_true, self.compliance_pred, zero_division=0),
            "recall": recall_score(self.compliance_true, self.compliance_pred, zero_division=0),
            "f1_score": f1_score(self.compliance_true, self.compliance_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(self.compliance_true, self.compliance_pred).tolist(),
        }

    def fees_metrics(self) -> dict:
        n = len(self.fees_true)
        diffs = [abs(t - p) for t, p in zip(self.fees_true, self.fees_pred)]
        return {
            "exact_match_rate": sum(1 for d in diffs if d < 0.01) / n if n else 0,
            "within_5_euros_rate": sum(1 for d in diffs if d <= 5.0) / n if n else 0,
            "within_10_euros_rate": sum(1 for d in diffs if d <= 10.0) / n if n else 0,
            "mae": mean_absolute_error(self.fees_true, self.fees_pred),
            "rmse": float(np.sqrt(mean_squared_error(self.fees_true, self.fees_pred))),
            "mape": float(
                np.mean(
                    [abs(t - p) / (abs(t) + 1e-8) for t, p in zip(self.fees_true, self.fees_pred)]
                )
                * 100
            ),
        }

    def cargo_metrics(self) -> dict:
        if len(self.mlb.classes_) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "jaccard_score": 0.0}
        return {
            "precision": precision_score(self.cargo_true_enc, self.cargo_pred_enc, average="macro", zero_division=0),
            "recall": recall_score(self.cargo_true_enc, self.cargo_pred_enc, average="macro", zero_division=0),
            "f1_score": f1_score(self.cargo_true_enc, self.cargo_pred_enc, average="macro", zero_division=0),
            "jaccard_score": jaccard_score(self.cargo_true_enc, self.cargo_pred_enc, average="macro", zero_division=0),
        }

    def message_metrics(self) -> dict:
        pairs = [
            (a, p)
            for a, p in zip(self.message_true, self.message_pred)
            if a.strip() and p.strip()
        ]
        if not pairs:
            return {"coverage_rate": 0.0, "total_evaluated": 0, "matches": 0}

        if self.judge:
            matches = [self.judge.messages_match(a, p) for a, p in pairs]
        else:
            matches = [a.lower() in p.lower() for a, p in pairs]

        return {
            "coverage_rate": float(np.mean(matches)),
            "total_evaluated": len(pairs),
            "matches": sum(matches),
        }

    def all_metrics(self) -> dict:
        return {
            "compliance_result": self.compliance_metrics(),
            "fees": self.fees_metrics(),
            "cargo_items": self.cargo_metrics(),
            "compliance_messages": self.message_metrics(),
            "total_samples": len(self.df),
        }

    def print_report(self) -> dict:
        results = self.all_metrics()

        print("=" * 60)
        print(f"EVALUATION RESULTS  ({results['total_samples']} samples)")
        print("=" * 60)

        comp = results["compliance_result"]
        print("\nCOMPLIANCE")
        print(f"  Accuracy:  {comp['accuracy']:.3f}")
        print(f"  Precision: {comp['precision']:.3f}")
        print(f"  Recall:    {comp['recall']:.3f}")
        print(f"  F1:        {comp['f1_score']:.3f}")
        cm = comp["confusion_matrix"]
        if len(cm) == 2:
            print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

        fees = results["fees"]
        print("\nFEES")
        print(f"  Exact match: {fees['exact_match_rate']:.3f}")
        print(f"  Within 5:    {fees['within_5_euros_rate']:.3f}")
        print(f"  Within 10:   {fees['within_10_euros_rate']:.3f}")
        print(f"  MAE:         {fees['mae']:.2f}")
        print(f"  RMSE:        {fees['rmse']:.2f}")
        print(f"  MAPE:        {fees['mape']:.1f}%")

        cargo = results["cargo_items"]
        print("\nCARGO ITEMS")
        print(f"  Precision: {cargo['precision']:.3f}")
        print(f"  Recall:    {cargo['recall']:.3f}")
        print(f"  F1:        {cargo['f1_score']:.3f}")
        print(f"  Jaccard:   {cargo['jaccard_score']:.3f}")

        msg = results["compliance_messages"]
        print("\nCOMPLIANCE MESSAGES")
        print(f"  Coverage: {msg['coverage_rate']:.3f}")
        print(f"  Matches:  {msg['matches']}/{msg['total_evaluated']}")

        return results
