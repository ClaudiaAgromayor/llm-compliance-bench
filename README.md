# LLM Compliance Bench

Can an LLM be trusted to apply a business policy? This project benchmarks several
LLM prompting strategies, up to a retrieval-augmented (RAG) approach, on a concrete
case: checking airline baggage compliance with IBM watsonx.ai (Mistral Large).

Developed in the AI track of CentraleSupelec (2024-25) with IBM France Lab:
*Leveraging LLMs for Business Logic Extraction and Inference*.

## The problem

Given a passenger (travel class, age category, list of luggage items) and the airline
baggage policy, the model must return:

- **compliance_result**: does the luggage comply with the policy?
- **compliance_message**: why not, if it does not comply
- **fees**: the extra fees to pay
- **cargo_items**: items that must travel as cargo

The policy sets weight, size and quantity limits per travel class, so the model has to
reason over several rules at once instead of just recognising patterns.

## What we did

1. **Data correction.** We reviewed the synthetic policy and data generator, fixed
   inconsistencies in `policy.txt` and in the generator, and produced a clean labelled
   dataset to use as ground truth.
2. **Prompting strategies.** We implemented and compared five approaches, each one
   built from the lessons of the previous one (see below).
3. **Evaluation framework.** We defined metrics for every part of the answer and added
   an LLM-as-a-judge to evaluate the compliance explanations.

## Techniques

| Technique | Idea |
|---|---|
| **Batch** | The whole policy and many passengers (up to ~145) go into a single prompt. Simple, but the model loses precision along the way and the JSON output becomes unreliable. |
| **Iterative simple** | One prompt per passenger, with the policy, travel class, age and luggage list. Outputs are parsed as JSON, with up to three attempts per passenger. |
| **Iterative few-shot** | Same as iterative simple, but with example conversations (compliant, non-compliant, surcharge cases) added before the question. |
| **Iterative questions** | The reasoning is split into sub-tasks (carry-on analysis, checked luggage, cargo items, final compliance and fees), each with its own prompt and fed with the previous results. |
| **RAG (hybrid)** | Each passenger is turned into a structured record. Records are vectorised (all-MiniLM-L6-v2 embeddings combined with TF-IDF) and the most similar solved cases are retrieved and added to the prompt as examples. The closest match is excluded at each step to avoid leaking the answer. |

Main takeaways: processing one passenger per prompt is far more robust than batching;
splitting the reasoning into many steps did not help because errors accumulate between
steps; giving the model similar solved cases was the most effective way to improve its
answers.

## Evaluation metrics

- **Compliance:** accuracy, precision, recall, F1 and confusion matrix.
- **Fees:** exact match, tolerance within 5/10, MAE, RMSE and MAPE.
- **Cargo items:** precision, recall, F1 and Jaccard score.
- **Compliance messages:** coverage rate of the expected reasons, evaluated by an
  LLM-as-a-judge that accepts different formulations of the same explanation.

## Repository structure

```
data/             policy.txt, dataset.csv (labelled passengers)
data_generator/   synthetic data generator and policy tester
experiments/      numbered development scripts (see below)
results/          metrics, predictions and figures
docs/             final report, presentation, RAG design notes
src/, main.py, research.py   evaluation framework (see below)
```

## Development scripts

The scripts in `experiments/` are the prototypes written while building the approaches
above. Each one adds an idea on top of the previous one.

| # | Script | Idea |
|---|---|---|
| 01 | `01_baseline_zero_shot.py` | Single prompt with the full policy |
| 02 | `02_few_shot.py` | Few-shot examples |
| 03 | `03_chat_memory.py` | Conversational LLM with memory |
| 04 | `04_embedding_pipeline.py` | Embeddings + retrieval pipeline |
| 05 | `05_rag_similar_cases.py` | RAG with similar solved cases |
| 06 | `06_funnel_rule_based_rag.py` | Rule-based "funnel" RAG |
| 07 | `07_llm_vs_funnel.py` | LLM vs funnel comparison |
| 08 | `08_compliance_fees_rag.py` | Final RAG: ComplianceRAG + FeesRAG (fee clusters) |
| 09 | `09_evaluation_clean_data.py` | Extended metrics on the cleaned dataset |

Helpers: `luggage_calculator.py` (deterministic fee calculator) and
`analyze_compliance_messages.py`. Raw outputs are in `results/experiments/` and
figures in `results/figures/`.

## Evaluation framework

`research.py` and `main.py` run the five approaches (plus a multi-prompt variant) with a
common CLI and compare them.

| Name | Strategy |
|---|---|
| `batch` | All clients in a single prompt |
| `iterative_simple` | One prompt per client |
| `iterative_fsl` | Few-shot examples + one prompt per client |
| `iterative_questions` | Multi-phase: policy comprehension, test cases, Q&A, then per-client |
| `rag` | TF-IDF similarity retrieval of similar solved cases |
| `multi_prompt` | Three prompt variants (standard, chain-of-thought, critical analysis) |

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in WATSONX_API_KEY

python research.py -e iterative_simple -n 50
python research.py -e all -n 50 --evaluate
python main.py  # evaluate all + generate comparison plots
```

```
main.py              # Evaluate all experiments and generate plots
research.py          # All experiments with CLI
src/
  config.py          # Constants, hyperparameters, paths
  data.py            # Data loading
  metrics.py         # Evaluation metrics + LLM-as-judge
  plotting.py        # Comparison bar charts
  prompts.py         # All prompt templates
  rag.py             # TF-IDF similarity retrieval
  utils.py           # LLM setup, JSON extraction, helpers
```

The development scripts contain a placeholder `YOUR_WATSONX_API_KEY`: replace it with
your own key and never commit it. Run them from the repository root; some of them read
the original datasets from a local checkout of `DecisionsDev/policy-corpus`.

## Acknowledgements

The luggage policy, data generator and test datasets are based on
[DecisionsDev/policy-corpus](https://github.com/DecisionsDev/policy-corpus)
(Apache License 2.0). Modifications: corrected/cleaned dataset, evaluation
metrics, prompting strategies and RAG experiments.

The evaluation pipeline (`main.py`, `research.py`, `src/`) is adapted from
[titouanbrunel/ibm-llm-policy-compliance](https://github.com/titouanbrunel/ibm-llm-policy-compliance).