import json
import numpy as np
import pytest
from pathlib import Path

from emotion_tracer.visualizer import Visualizer

EMOTIONS = ["happy", "sad", "angry"]
TOKENS = ["Hello", ",", "I", "am", "so", "sad", "today", "."]
SEQ_LEN = len(TOKENS)
NUM_EMOTIONS = len(EMOTIONS)

@pytest.fixture
def scores():
    rng = np.random.default_rng(42)
    return rng.standard_normal((NUM_EMOTIONS, SEQ_LEN))

def test_save_heatmap_creates_png(tmp_path, scores):
    viz = Visualizer()
    out = tmp_path / "heatmap.png"
    viz.plot_heatmap(scores, TOKENS, EMOTIONS, output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0

def test_save_timeline_creates_png(tmp_path, scores):
    viz = Visualizer()
    out = tmp_path / "timeline.png"
    viz.plot_timeline(scores, TOKENS, EMOTIONS, top_k=2, output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0

def test_save_json(tmp_path, scores):
    viz = Visualizer()
    out = tmp_path / "trajectory.json"
    viz.save_json(scores, TOKENS, EMOTIONS, output_path=str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == SEQ_LEN
    assert "token" in data[0]
    assert "position" in data[0]
    assert "emotion_scores" in data[0]
    assert set(data[0]["emotion_scores"].keys()) == set(EMOTIONS)

def test_save_csv(tmp_path, scores):
    import pandas as pd
    viz = Visualizer()
    out = tmp_path / "trajectory.csv"
    viz.save_csv(scores, TOKENS, EMOTIONS, output_path=str(out))
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == SEQ_LEN
    for e in EMOTIONS:
        assert e in df.columns

def test_turn_boundaries_accepted(tmp_path, scores):
    """plot_heatmap must not raise when turn_boundaries is provided."""
    viz = Visualizer()
    out = tmp_path / "heatmap_turns.png"
    # boundaries: list of (start_token_idx, end_token_idx, label)
    boundaries = [(0, 3, "Human"), (4, 7, "Assistant")]
    viz.plot_heatmap(
        scores, TOKENS, EMOTIONS,
        turn_boundaries=boundaries,
        output_path=str(out),
    )
    assert out.exists()
