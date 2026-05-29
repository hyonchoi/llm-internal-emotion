"""Minimal runnable example (requires downloaded model and pre-built vectors).

Demonstrates the full trace pipeline on a short conversation.
Run: python examples/example_conversation.py
"""
from pathlib import Path
import numpy as np

from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.model_wrapper import ModelWrapper
from emotion_tracer.emotion_vectors import EmotionVectorBuilder
from emotion_tracer.trajectory import TrajectoryExtractor
from emotion_tracer.visualizer import Visualizer

CONVERSATION = (
    "Human: I just got some really bad news. My grandmother passed away.\n"
    "Assistant: I'm so sorry for your loss. That must be incredibly painful."
)
VECTORS_PATH = "data/emotion_vectors/vectors_mean-diff_layer21.npz"
OUTPUT_DIR = "results/example"

def main():
    cfg = EmotionTracerConfig(device="cpu")

    if not Path(VECTORS_PATH).exists():
        print(f"Run build_vectors.py first to create {VECTORS_PATH}")
        return

    wrapper = ModelWrapper(cfg)
    wrapper.load()
    layer_indices = cfg.resolve_layer_indices(wrapper.num_layers)
    layer = layer_indices[0]

    hs_map = wrapper.get_hidden_states(CONVERSATION, layer_indices=[layer])
    tokens = wrapper.get_tokens(CONVERSATION)

    builder = EmotionVectorBuilder()
    builder.load(VECTORS_PATH)

    extractor = TrajectoryExtractor()
    scores = extractor.extract(hs_map[layer], builder.vectors)
    emotions = list(builder.vectors.keys())

    viz = Visualizer()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    viz.plot_heatmap(scores, tokens, emotions, output_path=f"{OUTPUT_DIR}/heatmap.png")
    viz.plot_timeline(scores, tokens, emotions, top_k=5, output_path=f"{OUTPUT_DIR}/timeline.png")
    viz.save_json(scores, tokens, emotions, output_path=f"{OUTPUT_DIR}/trajectory.json")
    viz.save_csv(scores, tokens, emotions, output_path=f"{OUTPUT_DIR}/trajectory.csv")

    print(f"Done. Results in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
