from src.metrics import MetricsCalculator
from src.plotting import MetricsComparator
from src.config import RESULTS_DIR


def main() -> None:
    csv_files = [
        f"{RESULTS_DIR}/iterative_questions/iterative_questions_prediction.csv",
        f"{RESULTS_DIR}/iterative_simple/iterative_simple_prediction.csv",
        f"{RESULTS_DIR}/iterative_fsl/iterative_fsl_prediction.csv",
        f"{RESULTS_DIR}/rag/rag_prediction.csv",
    ]
    labels = ["Iterative Questions", "Iterative Simple", "Iterative FSL", "RAG"]

    for path, label in zip(csv_files, labels):
        print(f"\n--- {label} ---")
        try:
            MetricsCalculator(path, use_llm_judge=False).print_report()
        except FileNotFoundError:
            print(f"  Not found: {path}")

    try:
        comparator = MetricsComparator(csv_files, labels)
        comparator.save_all(RESULTS_DIR)
        print(f"\nPlots saved to {RESULTS_DIR}/")
    except FileNotFoundError as e:
        print(f"\nCannot generate plots: {e}")


if __name__ == "__main__":
    main()
