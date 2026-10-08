<h1 align="center">LLM Compliance Bench</h1>

<p align="center">
  Can an LLM be trusted to apply a business policy?<br>
  Benchmarking prompting strategies and RAG on airline baggage compliance.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-blue" alt="Python 3">
  <img src="https://img.shields.io/badge/IBM-watsonx.ai-052FAD" alt="watsonx.ai">
  <img src="https://img.shields.io/badge/LLM-Mistral%20Large-orange" alt="Mistral Large">
  <img src="https://img.shields.io/badge/Method-RAG-success" alt="RAG">
  <img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey" alt="Apache 2.0">
</p>

Developed in the AI track of CentraleSupelec (2024-25) with IBM France Lab:
*Leveraging LLMs for Business Logic Extraction and Inference*.

## :dart: The problem

Given a passenger (travel class, age category, luggage items) and the airline baggage
policy, the model must return:

| Field | Meaning |
|---|---|
| `compliance_result` | Does the luggage comply with the policy? |
| `compliance_message` | Why not, if it does not comply |
| `fees` | Extra fees to pay |
| `cargo_items` | Items that must travel as cargo |

The policy sets weight, size and quantity limits per travel class, so the model has to
reason over several rules at once instead of just recognising patterns.

## :bulb: What we did

1. **Cleaned the data:** fixed inconsistencies in the policy and in the data generator
   to obtain a reliable labelled dataset.
2. **Compared five LLM strategies**, each one built from the lessons of the previous one.
3. **Built an evaluation framework** with metrics for every part of the answer, plus an
   LLM-as-a-judge for the explanations.

## :test_tube: Techniques

`Batch` -> `Iterative simple` -> `Iterative few-shot` -> `Iterative questions` -> `RAG`

| Technique | Idea |
|---|---|
| **Batch** | Policy + many passengers (up to ~145) in a single prompt. Simple, but precision drops and the JSON output becomes unreliable. |
| **Iterative simple** | One prompt per passenger, with up to 3 attempts to get valid JSON. |
| **Iterative few-shot** | Same, with example conversations (compliant, non-compliant, surcharge cases) added to the prompt. |
| **Iterative questions** | Reasoning split into sub-tasks (carry-on, checked luggage, cargo, final decision), each with its own prompt. |
| **RAG (hybrid)** | Passengers are vectorised (all-MiniLM-L6-v2 + TF-IDF); the most similar solved cases are retrieved and given to the model as examples. The closest match is excluded to avoid leaking the answer. |

**Takeaways:** one prompt per passenger is much more robust than batching; splitting the
reasoning into many steps accumulates errors; giving the model similar solved cases was
the most effective improvement.

## :straight_ruler: Metrics

| Part of the answer | Metrics |
|---|---|
| Compliance | Accuracy, precision, recall, F1, confusion matrix |
| Fees | Exact match, tolerance within 5/10, MAE, RMSE, MAPE |
| Cargo items | Precision, recall, F1, Jaccard |
| Messages | Coverage of expected reasons, judged by an LLM (LLM-as-a-judge) |

## :file_folder: How the files are organized

```
llm-compliance-bench/
|-- README.md, LICENSE, requirements.txt, .env.example
|
|-- main.py              Evaluate all approaches, generate plots
|-- research.py          CLI to run each approach
|-- src/                 Evaluation framework
|   |-- config.py        Constants, hyperparameters, paths
|   |-- data.py          Data loading
|   |-- prompts.py       Prompt templates
|   |-- rag.py           TF-IDF similarity retrieval
|   |-- metrics.py       Metrics + LLM-as-judge
|   |-- plotting.py      Comparison charts
|   |-- utils.py         LLM setup, JSON extraction
|
|-- data/
|   |-- policy.txt       Baggage policy (corrected)
|   |-- dataset.csv      Labelled passengers (ground truth)
|
|-- data_generator/      Synthetic data: generators, policy model, testers
|
|-- experiments/         Numbered development scripts (see below)
|
|-- results/
|   |-- experiments/     Outputs of the development scripts
|   |-- figures/         F1 evolution plots
|   |-- *.json           Outputs of the evaluation framework
|
|-- docs/                Report, presentation, RAG design notes
```

**Two layers of code:**
- `main.py`, `research.py`, `src/` is the **final framework**: one CLI that runs and
  compares all approaches.
- `experiments/` holds the **prototypes** written along the way. Read them in order.

<details>
<summary><b>Development scripts in <code>experiments/</code></b></summary>

| # | Script | Idea |
|---|---|---|
| 01 | `01_baseline_zero_shot.py` | Single prompt with the full policy |
| 02 | `02_few_shot.py` | Few-shot examples |
| 03 | `03_chat_memory.py` | Conversational LLM with memory |
| 04 | `04_embedding_pipeline.py` | Embeddings + retrieval pipeline |
| 05 | `05_rag_similar_cases.py` | RAG with similar solved cases |
| 06 | `06_funnel_rule_based_rag.py` | Rule-based "funnel" RAG |
| 07 | `07_llm_vs_funnel.py` | LLM vs funnel comparison |
| 08 | `08_compliance_fees_rag.py` | Final RAG: ComplianceRAG + FeesRAG |
| 09 | `09_evaluation_clean_data.py` | Extended metrics on the cleaned dataset |

Helpers: `luggage_calculator.py` (deterministic fee calculator) and
`analyze_compliance_messages.py`.
</details>

## :rocket: Quick start

```bash
pip install -r requirements.txt
cp .env.example .env     # add your watsonx credentials

python research.py -e iterative_simple -n 50     # run one approach on 50 cases
python research.py -e all -n 50 --evaluate       # run and evaluate everything
python main.py                                   # evaluate all + comparison plots
```

Available approaches (`-e`): `batch`, `iterative_simple`, `iterative_fsl`,
`iterative_questions`, `rag`, `multi_prompt`.

The scripts in `experiments/` contain a placeholder `YOUR_WATSONX_API_KEY`: use your own
key and never commit it. Run them from the repository root; some read the original
datasets from a local checkout of `DecisionsDev/policy-corpus`.

## :handshake: Acknowledgements

The luggage policy, data generator and test datasets are based on
[DecisionsDev/policy-corpus](https://github.com/DecisionsDev/policy-corpus)
(Apache License 2.0). Modifications: corrected/cleaned dataset, evaluation metrics,
prompting strategies and RAG experiments.
