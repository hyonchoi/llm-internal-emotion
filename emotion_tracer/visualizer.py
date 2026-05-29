import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Colors for Human/Assistant turn background bands
TURN_COLORS = {"Human": "#fff3cd", "Assistant": "#d1ecf1"}

class Visualizer:
    """Visualize emotion trajectories as heatmaps, timelines, JSON, and CSV."""

    def plot_heatmap(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        turn_boundaries: list[tuple] | None = None,
        output_path: str = "heatmap.png",
    ):
        """Heatmap: X=tokens, Y=emotions, color=activation strength.

        Args:
            scores: [num_emotions, seq_len]
            tokens: token strings for x-axis labels
            emotions: emotion names for y-axis labels
            turn_boundaries: list of (start_idx, end_idx, label) for background shading
            output_path: destination PNG path
        """
        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(4, len(emotions) * 0.4)))

        if turn_boundaries:
            for start, end, label in turn_boundaries:
                color = TURN_COLORS.get(label, "#f8f9fa")
                ax.axvspan(start - 0.5, end + 0.5, alpha=0.3, color=color, zorder=0)

        sns.heatmap(
            scores,
            ax=ax,
            xticklabels=tokens,
            yticklabels=emotions,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Activation"},
        )
        ax.set_xlabel("Token position")
        ax.set_ylabel("Emotion")
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

    def plot_timeline(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        top_k: int = 5,
        output_path: str = "timeline.png",
    ):
        """Line graph of top-K emotions across token positions.

        Args:
            scores: [num_emotions, seq_len]
            top_k: number of most active emotions to plot
        """
        mean_scores = scores.mean(axis=1)
        top_indices = np.argsort(mean_scores)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), 4))
        x = np.arange(len(tokens))
        for idx in top_indices:
            ax.plot(x, scores[idx], label=emotions[idx], linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Emotion activation")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

    def save_json(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        output_path: str = "trajectory.json",
    ):
        """Save per-token emotion scores as JSON."""
        records = []
        for t, token in enumerate(tokens):
            records.append({
                "token": token,
                "position": t,
                "emotion_scores": {e: float(scores[i, t]) for i, e in enumerate(emotions)},
            })
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(records, ensure_ascii=False, indent=2))

    def save_csv(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        output_path: str = "trajectory.csv",
    ):
        """Save per-token emotion scores as CSV (rows=tokens, cols=emotions)."""
        df = pd.DataFrame(scores.T, columns=emotions)
        df.insert(0, "token", tokens)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
