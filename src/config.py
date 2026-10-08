import os
from dataclasses import dataclass, field
from typing import Optional


WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "a31eb1db-e559-4a08-b0fa-638fdc608777")
MODEL_ID = "mistralai/mistral-large"

DEFAULT_MAX_NEW_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.1
JUDGE_MAX_NEW_TOKENS = 100
JUDGE_TEMPERATURE = 0.1

POLICY_PATH = "data/policy.txt"
DATASET_PATH = "data/dataset.csv"
RESULTS_DIR = "results"

DEFAULT_MAX_CLIENTS = 50
MAX_RETRIES = 3

RAG_TOP_K = 3
RAG_NGRAM_RANGE = (1, 2)
RAG_MIN_DF = 1
RAG_MAX_DF = 0.95

OVERWEIGHT_FEE = 75
OVERSIZE_FEE = 100
EXTRA_PIECE_FEE = 150
MAX_CHECKED_WEIGHT_KG = 32
MAX_CHECKED_DIMENSIONS_CM = 203

CARRY_ON_SIZE_LIMIT = {"height": 55, "width": 40, "depth": 23}

ECONOMY_CARRY_ON_WEIGHT_LIMIT = 7
BUSINESS_FIRST_CARRY_ON_WEIGHT_LIMIT = 12
ECONOMY_CARRY_ON_QTY = 1
BUSINESS_FIRST_CARRY_ON_QTY = 2

ECONOMY_CHECKED_QTY = 1
ECONOMY_CHECKED_WEIGHT = 23
BUSINESS_CHECKED_QTY = 2
BUSINESS_CHECKED_WEIGHT = 32
FIRST_CHECKED_QTY = 3
FIRST_CHECKED_WEIGHT = 32
CHECKED_SIZE_LIMIT = 158

FSL_EXCLUDED_INDICES = {3, 6, 12, 17}

ITERATIVE_QUESTIONS_MAX_QUESTIONS = 3
ITERATIVE_QUESTIONS_EXAMPLE_COUNT = 3

PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_FONT_SIZE = 10
PLOT_TITLE_SIZE = 12
PLOT_LABEL_SIZE = 10
PLOT_DPI = 300
PLOT_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#17BECF", "#2CA02C"]
PLOT_BAR_WIDTH = 0.15
PLOT_BAR_ALPHA = 0.8


@dataclass
class ExperimentConfig:
    name: str
    max_clients: int = DEFAULT_MAX_CLIENTS
    output_dir: str = RESULTS_DIR
    policy_path: str = POLICY_PATH
    dataset_path: str = DATASET_PATH
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_dir, self.name, f"{self.name}_prediction.csv")

    @property
    def experiment_dir(self) -> str:
        return os.path.join(self.output_dir, self.name)
