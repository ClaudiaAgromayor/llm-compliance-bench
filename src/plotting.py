import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    PLOT_BAR_ALPHA,
    PLOT_BAR_WIDTH,
    PLOT_COLORS,
    PLOT_DPI,
    PLOT_FONT_SIZE,
    PLOT_LABEL_SIZE,
    PLOT_STYLE,
    PLOT_TITLE_SIZE,
    RESULTS_DIR,
)
from src.metrics import MetricsCalculator


class MetricsComparator:
    def __init__(self, csv_paths: list[str], labels: Optional[list[str]] = None):
        plt.style.use(PLOT_STYLE)
        plt.rcParams["font.size"] = PLOT_FONT_SIZE
        plt.rcParams["axes.titlesize"] = PLOT_TITLE_SIZE
        plt.rcParams["axes.labelsize"] = PLOT_LABEL_SIZE

        self.csv_paths = csv_paths
        self.labels = labels or [f"Model {i + 1}" for i in range(len(csv_paths))]
        self.results = [
            MetricsCalculator(path, use_llm_judge=False).all_metrics()
            for path in csv_paths
        ]

    def _grouped_bar(
        self,
        ax: plt.Axes,
        metric_keys: list[str],
        category: str,
        title: str,
        ylim: tuple = (0, 1.1),
    ) -> None:
        n_groups = len(metric_keys)
        x = np.arange(n_groups)
        for i, label in enumerate(self.labels):
            values = [self.results[i][category][k] for k in metric_keys]
            offset = (i - len(self.labels) // 2) * PLOT_BAR_WIDTH
            bars = ax.bar(
                x + offset,
                values,
                PLOT_BAR_WIDTH,
                label=label,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
                alpha=PLOT_BAR_ALPHA,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_", " ").capitalize() for k in metric_keys])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold", pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

    def plot_cargo(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        self._grouped_bar(ax, ["precision", "recall", "f1_score"], "cargo_items", "Cargo Items Metrics")
        plt.tight_layout()
        return fig

    def plot_compliance(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        self._grouped_bar(
            ax,
            ["precision", "recall", "f1_score", "accuracy"],
            "compliance_result",
            "Compliance Result Metrics",
        )
        plt.tight_layout()
        return fig

    def plot_fees(self) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(1)
        for i, label in enumerate(self.labels):
            val = self.results[i]["fees"]["exact_match_rate"]
            offset = (i - len(self.labels) // 2) * PLOT_BAR_WIDTH
            bars = ax1.bar(
                x + offset,
                [val],
                PLOT_BAR_WIDTH,
                label=label,
                color=PLOT_COLORS[i % len(PLOT_COLORS)],
                alpha=PLOT_BAR_ALPHA,
                edgecolor="white",
                linewidth=0.5,
            )
            ax1.text(
                bars[0].get_x() + bars[0].get_width() / 2,
                bars[0].get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax1.set_xticks([0])
        ax1.set_xticklabels(["Exact Match"])
        ax1.set_ylim(0, 1.1)
        ax1.set_title("Exact Match Rate", fontweight="bold")
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)

        self._grouped_bar(
            ax2,
            ["mae", "rmse"],
            "fees",
            "Prediction Errors",
            ylim=(0, None),
        )
        ax2.set_ylim(bottom=0)
        plt.tight_layout()
        return fig

    def plot_fees_comparison(self, csv_index: int, max_samples: int = 50) -> plt.Figure:
        calc = MetricsCalculator(self.csv_paths[csv_index], use_llm_judge=False)
        n = min(len(calc.fees_true), max_samples)
        true_fees = calc.fees_true[:n]
        pred_fees = calc.fees_pred[:n]
        x = np.arange(n)
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(12, n * 0.25), 7))
        ax.bar(x - width / 2, true_fees, width, label="Actual Fees", color=PLOT_COLORS[0], alpha=PLOT_BAR_ALPHA)
        ax.bar(x + width / 2, pred_fees, width, label="Predicted Fees", color=PLOT_COLORS[2], alpha=PLOT_BAR_ALPHA)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Fee Amount")
        ax.set_title(f"Actual vs Predicted Fees - {self.labels[csv_index]}", fontweight="bold")
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def save_all(self, output_dir: str = RESULTS_DIR, prefix: str = "comparison") -> None:
        os.makedirs(output_dir, exist_ok=True)
        plots = {
            "cargo": self.plot_cargo,
            "compliance": self.plot_compliance,
            "fees": self.plot_fees,
        }
        for name, plot_fn in plots.items():
            fig = plot_fn()
            fig.savefig(
                os.path.join(output_dir, f"{prefix}_{name}.png"),
                dpi=PLOT_DPI,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

        for i, label in enumerate(self.labels):
            fig = self.plot_fees_comparison(i)
            fig.savefig(
                os.path.join(output_dir, f"{prefix}_fees_{label.lower().replace(' ', '_')}.png"),
                dpi=PLOT_DPI,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
