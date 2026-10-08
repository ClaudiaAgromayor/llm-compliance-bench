import argparse
import os
from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import (
    DEFAULT_MAX_CLIENTS,
    FSL_EXCLUDED_INDICES,
    ITERATIVE_QUESTIONS_EXAMPLE_COUNT,
    ITERATIVE_QUESTIONS_MAX_QUESTIONS,
    MAX_RETRIES,
    RESULTS_DIR,
)
from src.data import DataLoader
from src.metrics import MetricsCalculator
from src.prompts import (
    FSL_EXAMPLES,
    POLICY_UNDERSTANDING,
    QUESTIONS_PROMPT,
    SYSTEM_BATCH,
    SYSTEM_CHAIN_OF_THOUGHT,
    SYSTEM_CRITICAL_ANALYSIS,
    SYSTEM_JSON_ONLY,
    SYSTEM_RAG,
    TEST_CASE_TEMPLATE,
)
from src.rag import LuggageRAG
from src.utils import (
    build_llm,
    ensure_dir,
    extract_json,
    init_prediction_columns,
    invoke_with_retries,
    save_predictions,
    store_prediction,
    validate_prediction,
)


class Experiment(ABC):
    def __init__(
        self,
        name: str,
        policy_path: str,
        dataset_path: str,
        max_clients: int = DEFAULT_MAX_CLIENTS,
    ):
        self.name = name
        self.data_loader = DataLoader(dataset_path, policy_path)
        self.max_clients = max_clients
        self.output_dir = os.path.join(RESULTS_DIR, name)
        self.output_path = os.path.join(self.output_dir, f"{name}_prediction.csv")

    @abstractmethod
    def run(self) -> str:
        pass

    def evaluate(self, use_llm_judge: bool = False) -> dict:
        calc = MetricsCalculator(self.output_path, use_llm_judge=use_llm_judge)
        return calc.print_report()


class BatchExperiment(Experiment):
    def __init__(self, policy_path: str, dataset_path: str, max_clients: int = DEFAULT_MAX_CLIENTS):
        super().__init__("batch", policy_path, dataset_path, max_clients)

    def run(self) -> str:
        llm = build_llm()
        data = self.data_loader.get_head(self.max_clients)

        clients_text = "\n".join(
            self.data_loader.format_client(idx, row) for idx, row in data.iterrows()
        )
        user_prompt = (
            f"BAGGAGE POLICY:\n{self.data_loader.policy}\n\n"
            f"CLIENTS:\n{clients_text}\n\n"
            "Return the JSON with all clients."
        )
        messages = [
            SystemMessage(content=SYSTEM_BATCH),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        json_data = extract_json(response)
        if not json_data or "results" not in json_data:
            raise ValueError("Failed to extract batch JSON results")

        results_df = init_prediction_columns(data)
        for pred in json_data["results"]:
            client_id = pred.get("client_id", 0)
            if 1 <= client_id <= len(results_df):
                idx = results_df.index[client_id - 1]
                store_prediction(results_df, idx, validate_prediction(pred))

        save_predictions(results_df, self.output_path)
        predicted = results_df["predicted_compliance_result"].notna().sum()
        print(f"[batch] {predicted}/{len(results_df)} clients predicted")
        return self.output_path


class IterativeSimpleExperiment(Experiment):
    def __init__(self, policy_path: str, dataset_path: str, max_clients: int = DEFAULT_MAX_CLIENTS):
        super().__init__("iterative_simple", policy_path, dataset_path, max_clients)

    def run(self) -> str:
        llm = build_llm()
        data = self.data_loader.get_head(self.max_clients)
        results_df = init_prediction_columns(data)

        for idx, row in data.iterrows():
            print(f"[iterative_simple] Client {idx + 1}/{self.max_clients}")
            user_prompt = (
                f"POLICY:\n{self.data_loader.policy}\n\n"
                f"PASSENGER:\n{self.data_loader.format_client(idx, row)}\n\n"
                "Analyze the passenger's luggage according to the policy and return only the JSON result."
            )
            messages = [
                SystemMessage(content=SYSTEM_JSON_ONLY),
                HumanMessage(content=user_prompt),
            ]
            response = invoke_with_retries(llm, messages, MAX_RETRIES)
            if not response:
                continue
            json_data = extract_json(response)
            if not json_data:
                continue
            store_prediction(results_df, idx, validate_prediction(json_data))

        save_predictions(results_df, self.output_path)
        return self.output_path


class IterativeFSLExperiment(Experiment):
    def __init__(self, policy_path: str, dataset_path: str, max_clients: int = DEFAULT_MAX_CLIENTS):
        super().__init__("iterative_fsl", policy_path, dataset_path, max_clients)

    def run(self) -> str:
        llm = build_llm()
        data = self.data_loader.drop_indices(FSL_EXCLUDED_INDICES, self.max_clients)
        results_df = init_prediction_columns(data)

        system_prompt = (
            "You are a luggage compliance evaluation expert.\n"
            "You must strictly respond ONLY with a valid JSON object, "
            "with no markdown formatting, no explanations, no extra text.\n\n"
            "Below are a few examples of how to evaluate luggage compliance:\n\n"
            f"{FSL_EXAMPLES}"
        )

        for idx, row in data.iterrows():
            print(f"[iterative_fsl] Client {idx + 1}/{self.max_clients}")
            user_prompt = (
                f"POLICY:\n{self.data_loader.policy}\n\n"
                f"PASSENGER:\n{self.data_loader.format_client(idx, row)}\n\n"
                "Analyze the passenger's luggage according to the policy and return only the JSON result."
            )
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            messages = [HumanMessage(content=full_prompt)]
            response = invoke_with_retries(llm, messages, MAX_RETRIES)
            if not response:
                continue
            json_data = extract_json(response)
            if not json_data:
                continue
            store_prediction(results_df, idx, validate_prediction(json_data))

        save_predictions(results_df, self.output_path)
        return self.output_path


class IterativeQuestionsExperiment(Experiment):
    def __init__(self, policy_path: str, dataset_path: str, max_clients: int = DEFAULT_MAX_CLIENTS):
        super().__init__("iterative_questions", policy_path, dataset_path, max_clients)
        self.messages: list = []
        self.base_memory_size: int = 0

    def _send(self, content: str, role: str = "human") -> str:
        if role == "human":
            self.messages.append(HumanMessage(content=content))
        else:
            self.messages.append(SystemMessage(content=content))
        llm = build_llm(max_new_tokens=1000)
        response = llm.invoke(self.messages)
        self.messages.append(AIMessage(content=response))
        return response

    def _phase_policy_understanding(self) -> None:
        print("[iterative_questions] Phase 1: Policy understanding")
        prompt = POLICY_UNDERSTANDING.format(policy=self.data_loader.policy)
        self._send(prompt, "system")

    def _phase_test_cases(self) -> None:
        step = len(self.data_loader.dataset) // (ITERATIVE_QUESTIONS_EXAMPLE_COUNT + 2)
        for i in range(ITERATIVE_QUESTIONS_EXAMPLE_COUNT):
            row = self.data_loader.dataset.iloc[step * (i + 1)]
            print(f"[iterative_questions] Phase {i + 2}: Test case {i + 1}")
            prompt = TEST_CASE_TEMPLATE.format(
                travel_class=row["travel_class"],
                age_category=row["age_category"],
                luggages=row["luggages"],
            )
            self._send(prompt)

    def _phase_questions(self) -> None:
        print("[iterative_questions] Phase: Questions")
        response = self._send(QUESTIONS_PROMPT)

        question_count = 0
        while question_count < ITERATIVE_QUESTIONS_MAX_QUESTIONS:
            question_markers = ["?", "question", "clarification", "unclear", "specify"]
            if any(w in response.lower() for w in question_markers):
                answer = input("\nYour answer to the question: ")
                if answer.strip().lower() in ["no", "none", "ok", "good"]:
                    break
                response = self._send(answer)
                question_count += 1
            else:
                break

        self.base_memory_size = len(self.messages)

    def run(self) -> str:
        self._phase_policy_understanding()
        self._phase_test_cases()
        self._phase_questions()

        start_idx = len(self.data_loader.dataset) // 2
        data = self.data_loader.get_subset(self.max_clients, start_idx)
        results_df = init_prediction_columns(data)

        for i, (idx, row) in enumerate(data.iterrows(), 1):
            print(f"[iterative_questions] Client {i}/{len(data)}")
            prompt = (
                f"Analyze this case according to the understood policy:\n\n"
                f"Travel Class: {row['travel_class']}\n"
                f"Age Category: {row['age_category']}\n"
                f"Luggage: {row['luggages']}\n\n"
                "Respond in strict JSON format:\n"
                '{"compliance_result": true/false, "fees": 0, "cargo_items": [], "compliance_message": "explanation"}'
            )
            response = self._send(prompt)
            self.messages = self.messages[: self.base_memory_size]

            json_data = extract_json(response)
            if json_data:
                store_prediction(results_df, idx, validate_prediction(json_data))

        save_predictions(results_df, self.output_path)
        return self.output_path


class RAGExperiment(Experiment):
    def __init__(self, policy_path: str, dataset_path: str, max_clients: int = DEFAULT_MAX_CLIENTS):
        super().__init__("rag", policy_path, dataset_path, max_clients)
        self.rag = LuggageRAG(dataset_path)

    @staticmethod
    def _format_similar_cases(cases: list[dict]) -> str:
        parts = []
        for i, case in enumerate(cases, 1):
            result_str = "Compliant" if case["compliance_result"] else "Non-compliant"
            parts.append(
                f"Similar Case {i}:\n"
                f"- Travel Class: {case['travel_class']}\n"
                f"- Age Category: {case['age_category']}\n"
                f"- Luggage: {case['luggages']}\n"
                f"- Result: {result_str}\n"
                f"- Message: {case['compliance_message']}\n"
                f"- Fees: {case['fees']}\n"
            )
        return "\n".join(parts)

    def run(self) -> str:
        llm = build_llm(max_new_tokens=1000)
        data = self.data_loader.get_head(self.max_clients)
        results_df = init_prediction_columns(data)

        for idx, row in data.iterrows():
            print(f"[rag] Client {idx + 1}/{self.max_clients}")
            dataset_index = data.index.get_loc(idx) if idx in data.index else None
            similar_cases = self.rag.get_similar_cases(row, exclude_index=dataset_index)
            similar_text = self._format_similar_cases(similar_cases)

            user_prompt = (
                f"LUGGAGE POLICY:\n{self.data_loader.policy}\n\n"
                f"CURRENT PASSENGER:\n"
                f"- Travel Class: {row['travel_class']}\n"
                f"- Age Category: {row['age_category']}\n"
                f"- Luggage: {row['luggages']}\n\n"
                f"SIMILAR CASES FOR REFERENCE:\n{similar_text}\n\n"
                "Analyze this passenger's luggage. Return only the JSON result."
            )
            messages = [
                SystemMessage(content=SYSTEM_RAG),
                HumanMessage(content=user_prompt),
            ]
            response = invoke_with_retries(llm, messages, MAX_RETRIES)
            if not response:
                continue
            json_data = extract_json(response)
            if not json_data:
                continue
            store_prediction(results_df, idx, validate_prediction(json_data))

        save_predictions(results_df, self.output_path)
        return self.output_path


class MultiPromptExperiment(Experiment):
    def __init__(self, policy_path: str, dataset_path: str, max_clients: int = DEFAULT_MAX_CLIENTS):
        super().__init__("multi_prompt", policy_path, dataset_path, max_clients)

    def run(self) -> str:
        llm = build_llm(max_new_tokens=10000)
        data = self.data_loader.get_head(self.max_clients)

        clients_text = "\n".join(
            self.data_loader.format_client(idx, row) for idx, row in data.iterrows()
        )
        user_prompt = (
            f"POLICY:\n{self.data_loader.policy}\n\n"
            f"CLIENTS:\n{clients_text}\n\n"
            "Return the JSON with all clients."
        )

        prompt_variants = {
            "standard": SYSTEM_BATCH,
            "chain_of_thought": SYSTEM_CHAIN_OF_THOUGHT,
            "critical_analysis": SYSTEM_CRITICAL_ANALYSIS,
        }

        for variant_name, system_prompt in prompt_variants.items():
            print(f"[multi_prompt] Running variant: {variant_name}")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)

            raw_path = os.path.join(self.output_dir, f"raw_{variant_name}.txt")
            ensure_dir(self.output_dir)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response)

            json_data = extract_json(response)
            if not json_data or "results" not in json_data:
                print(f"[multi_prompt] JSON extraction failed for {variant_name}")
                continue

            results_df = init_prediction_columns(data)
            for pred in json_data["results"]:
                client_id = pred.get("client_id", 0)
                if 1 <= client_id <= len(results_df):
                    idx = results_df.index[client_id - 1]
                    store_prediction(results_df, idx, validate_prediction(pred))

            variant_path = os.path.join(
                self.output_dir, f"multi_prompt_{variant_name}_prediction.csv"
            )
            save_predictions(results_df, variant_path)
            print(f"[multi_prompt] Saved {variant_name} -> {variant_path}")

        return self.output_dir


EXPERIMENTS = {
    "batch": BatchExperiment,
    "iterative_simple": IterativeSimpleExperiment,
    "iterative_fsl": IterativeFSLExperiment,
    "iterative_questions": IterativeQuestionsExperiment,
    "rag": RAGExperiment,
    "multi_prompt": MultiPromptExperiment,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run luggage compliance experiments")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="all",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Experiment to run",
    )
    parser.add_argument("-n", "--max-clients", type=int, default=DEFAULT_MAX_CLIENTS)
    parser.add_argument("--policy", type=str, default="data/policy.txt")
    parser.add_argument("--dataset", type=str, default="data/dataset.csv")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after running")
    args = parser.parse_args()

    experiments_to_run = (
        list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]
    )

    for exp_name in experiments_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp_name}")
        print(f"{'=' * 60}")
        exp_class = EXPERIMENTS[exp_name]
        exp = exp_class(
            policy_path=args.policy,
            dataset_path=args.dataset,
            max_clients=args.max_clients,
        )
        output = exp.run()
        print(f"Output: {output}")

        if args.evaluate and os.path.isfile(output):
            exp.evaluate(use_llm_judge=False)


if __name__ == "__main__":
    main()
