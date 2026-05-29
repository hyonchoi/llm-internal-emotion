#!/usr/bin/env python3
"""Extract emotion trajectory for a conversation.

Usage:
    python scripts/trace.py \
        --input "Human: I just lost my job. Assistant: I'm so sorry." \
        --emotions happy sad anxious calm \
        --vectors data/emotion_vectors/vectors_mean-diff_layer21.npz \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --output results/
"""
import argparse
import numpy as np
from pathlib import Path

from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.model_wrapper import ModelWrapper
from emotion_tracer.emotion_vectors import EmotionVectorBuilder
from emotion_tracer.trajectory import TrajectoryExtractor
from emotion_tracer.visualizer import Visualizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Conversation text or path to text file")
    p.add_argument("--emotions", nargs="+", default=None, help="Filter to these emotions only")
    p.add_argument("--vectors", required=True, help="Path to .npz emotion vectors file")
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--output", default="results/")
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def detect_turn_boundaries(tokens: list[str]) -> list[tuple]:
    """Heuristic: find 'Human' and 'Assistant' token positions to shade turn backgrounds."""
    boundaries = []
    current_label = None
    start = 0
    for i, tok in enumerate(tokens):
        if "Human" in tok:
            if current_label is not None:
                boundaries.append((start, i - 1, current_label))
            current_label = "Human"
            start = i
        elif "Assistant" in tok:
            if current_label is not None:
                boundaries.append((start, i - 1, current_label))
            current_label = "Assistant"
            start = i
    if current_label is not None:
        boundaries.append((start, len(tokens) - 1, current_label))
    return boundaries


def main():
    args = parse_args()

    # D6: Guard for missing vectors file
    vectors_path = Path(args.vectors)
    if not vectors_path.exists():
        print(f"ERROR: Vectors file not found: {vectors_path}")
        print(f"Run build_vectors.py first to create it.")
        raise SystemExit(1)

    # Load conversation text
    input_path = Path(args.input)
    if input_path.exists():
        conversation = input_path.read_text()
    else:
        conversation = args.input

    cfg = EmotionTracerConfig(
        model_name=args.model,
        device=args.device,
    )

    print(f"Loading model: {cfg.model_name}")
    wrapper = ModelWrapper(cfg)
    wrapper.load()
    layer_indices = cfg.resolve_layer_indices(wrapper.num_layers)
    layer = layer_indices[0]

    print("Extracting hidden states...")
    hs_map = wrapper.get_hidden_states(conversation, layer_indices=[layer])
    hidden_states = hs_map[layer]  # [seq_len, hidden_dim]
    tokens = wrapper.get_tokens(conversation)

    print(f"Loading emotion vectors: {args.vectors}")
    builder = EmotionVectorBuilder()
    builder.load(args.vectors)
    emotion_vectors = builder.vectors

    if args.emotions:
        emotion_vectors = {e: v for e, v in emotion_vectors.items() if e in args.emotions}

    print(f"Projecting {len(emotion_vectors)} emotions over {len(tokens)} tokens...")
    extractor = TrajectoryExtractor()
    scores = extractor.extract(hidden_states, emotion_vectors)
    emotions = list(emotion_vectors.keys())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_boundaries = detect_turn_boundaries(tokens)

    viz = Visualizer()
    viz.plot_heatmap(scores, tokens, emotions, turn_boundaries=turn_boundaries,
                     output_path=str(out_dir / "heatmap.png"))
    viz.plot_timeline(scores, tokens, emotions, top_k=args.top_k,
                      output_path=str(out_dir / "timeline.png"))
    viz.save_json(scores, tokens, emotions, output_path=str(out_dir / "trajectory.json"))
    viz.save_csv(scores, tokens, emotions, output_path=str(out_dir / "trajectory.csv"))

    print(f"Results saved to {out_dir}/")
    print(f"  heatmap.png, timeline.png, trajectory.json, trajectory.csv")


if __name__ == "__main__":
    main()
