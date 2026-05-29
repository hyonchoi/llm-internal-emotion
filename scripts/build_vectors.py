#!/usr/bin/env python3
"""Pre-compute emotion vectors and save to cache.

Orchestrates the full vector-building pipeline:
1. Load model + tokenizer
2. Generate stories (emotion + neutral) — or load from --stories-file
3. Extract hidden states (last token)
4. Compute vectors (mean-diff and/or logistic probe)
5. Save to .npz cache

Usage:
    python scripts/build_vectors.py \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --emotions happy sad angry afraid calm \\
        --method both \\
        --cache-dir data/emotion_vectors \\
        --stories 20 \\
        --device cuda
"""
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.model_wrapper import ModelWrapper
from emotion_tracer.story_generator import StoryGenerator
from emotion_tracer.emotion_vectors import EmotionVectorBuilder


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute emotion vectors and save to cache."
    )
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument(
        "--emotions",
        nargs="+",
        default=["happy", "sad", "angry", "afraid", "calm"],
    )
    p.add_argument(
        "--method",
        choices=["mean-diff", "logistic-probe", "both"],
        default="both",
    )
    p.add_argument("--cache-dir", default="data/emotion_vectors")
    p.add_argument("--stories", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--stories-file",
        default=None,
        help="Path to pre-generated stories JSON file "
             "(format: {\"emotion\": [stories...], \"neutral\": [stories...], ...})",
    )
    return p.parse_args()


def validate_cache(
    cache_dir: str, emotions: list[str], methods: list[str], layer_indices: list[int]
) -> bool:
    """Check that all expected .npz files exist before reporting cache hit.

    Returns True only if every method × layer combination has a corresponding
    .npz file, AND each file contains keys for all emotions.
    """
    path = Path(cache_dir)
    for method in methods:
        for l in layer_indices:
            save_path = path / f"vectors_{method}_layer{l}.npz"
            if not save_path.exists():
                return False
            # Also verify the file contains all expected emotion keys
            data = np.load(str(save_path))
            for emotion in emotions:
                if emotion not in data.files:
                    return False
    return True


def collect_last_token_hs(
    wrapper: ModelWrapper, stories: list[str], layer_indices: list[int]
) -> dict[int, list[np.ndarray]]:
    """Extract last-token hidden states for each story and layer.

    Args:
        wrapper: loaded ModelWrapper instance.
        stories: list of text strings to process.
        layer_indices: transformer layer indices to extract.

    Returns:
        {layer_idx: [hs_0, hs_1, ...]} where each hs has shape [hidden_dim].
    """
    hs_per_layer = {l: [] for l in layer_indices}
    for story in stories:
        hs_map = wrapper.get_hidden_states(story, layer_indices)
        for l in layer_indices:
            hs_per_layer[l].append(hs_map[l][-1])  # last token
    return hs_per_layer


def load_stories_from_file(
    stories_file: str, emotions: list[str]
) -> dict[str, list[str]]:
    """Load pre-generated stories from a JSON file.

    Expected format:
        {
            "happy": ["story1", "story2", ...],
            "sad": ["story1", ...],
            "neutral": ["story1", ...],
            ...
        }
    """
    with open(stories_file) as f:
        data = json.load(f)
    return data


def main():
    args = parse_args()
    cfg = EmotionTracerConfig(
        model_name=args.model,
        emotions=args.emotions,
        stories_per_emotion=args.stories,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    methods = (
        ["mean-diff", "logistic-probe"]
        if args.method == "both"
        else [args.method]
    )

    print(f"Loading model: {cfg.model_name}")
    wrapper = ModelWrapper(cfg)
    wrapper.load()
    layer_indices = cfg.resolve_layer_indices(wrapper.num_layers)
    print(f"Using layers: {layer_indices}")

    # D3: Check cache completeness across ALL layer_indices (not just [0])
    if validate_cache(args.cache_dir, cfg.emotions, methods, layer_indices):
        print("Cache is complete. Skipping vector computation.")
        return

    # D4: Support external stories file
    gen = None
    if args.stories_file:
        print(f"Loading stories from: {args.stories_file}")
        stories_data = load_stories_from_file(args.stories_file, cfg.emotions)
        neutral_stories = stories_data.get("neutral", [])
        emotion_stories = {
            e: stories_data.get(e, []) for e in cfg.emotions
        }
    else:
        gen = StoryGenerator(cfg, wrapper)
        neutral_stories = gen.get_neutral_stories()
        emotion_stories = None

    # Collect neutral hidden states
    print("Collecting neutral story activations...")

    # D5: Use the DRY helper to collect last-token hidden states
    neutral_hs = collect_last_token_hs(wrapper, neutral_stories, layer_indices)
    for l in layer_indices:
        neutral_hs[l] = np.stack(neutral_hs[l])

    # Compute vectors for each method × layer combination
    for method in methods:
        for l in layer_indices:
            builder = EmotionVectorBuilder()

            for emotion in tqdm(cfg.emotions, desc=f"building vectors ({method}, layer {l})"):
                if emotion_stories:
                    stories = emotion_stories[emotion]
                else:
                    stories = gen.get_stories(emotion)

                # D5: Same helper for positive stories
                pos_hs = collect_last_token_hs(wrapper, stories, layer_indices)
                pos_hs[l] = np.stack(pos_hs[l])

                if method == "mean-diff":
                    vec = builder.mean_diff(pos_hs[l], neutral_hs[l])
                else:
                    vec = builder.logistic_probe(pos_hs[l], neutral_hs[l])

                builder.vectors[emotion] = vec

            save_path = Path(args.cache_dir) / f"vectors_{method}_layer{l}.npz"
            builder.save(str(save_path))
            print(f"Saved: {save_path}")

    print("Vector computation complete.")


if __name__ == "__main__":
    main()
