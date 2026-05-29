# LLM Internal Emotion Trajectory Extractor

> Visualize the emotional path a language model takes as it processes a conversation — token by token.

When a human speaks, emotion-related brain regions activate and deactivate in real time. Large language models have an analogous phenomenon: as they process each word, certain internal "directions" in their neural network activate. This project measures those directions and draws them as a map.

**Based on:** [Sofroniew et al., Anthropic 2024](https://arxiv.org/abs/2604.07729)  
**Target models:** Mistral / Mixtral (HuggingFace open-source)

---

## Quick Start

```bash
# Install
uv sync

# Step 1: Pre-compute emotion vectors (one-time, ~10-30 min, GPU recommended)
python scripts/build_vectors.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --emotions happy sad angry afraid calm anxious \
    --method both

# Step 2: Trace a conversation
python scripts/trace.py \
    --input "Human: I lost my job today. Assistant: I'm so sorry, that must be devastating." \
    --vectors data/emotion_vectors/vectors_mean-diff_layer21.npz \
    --output results/
```

Results appear in `results/`: `heatmap.png`, `timeline.png`, `trajectory.json`, `trajectory.csv`.

---

## What the Output Means

**Red in the heatmap** = that emotion is strongly activated at that token  
**Blue** = that emotion is suppressed  
**Yellow background** = Human turn  
**Blue background** = Assistant turn

The token just before "Assistant:" is especially significant: research shows the model's emotion at that exact moment predicts the emotional tone of its response.

See [docs/explanation/how_to_read_the_plots.md](docs/explanation/how_to_read_the_plots.md) for a full walkthrough.

---

## Architecture

```
build_vectors.py → story_generator → model_wrapper → emotion_vectors → data/*.npz
trace.py         → model_wrapper → trajectory → visualizer → PNG + JSON + CSV
```

---

## Deeper Reading

- [What are emotion vectors?](docs/explanation/what_are_emotion_vectors.md)
- [How to read the plots](docs/explanation/how_to_read_the_plots.md)
- [Paper summary (non-expert)](docs/explanation/paper_summary.md)
- [Limitations of this approach](docs/explanation/limitations.md)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Hardware

Works on CPU, but extracting hidden states from a 7B model is slow (~minutes per conversation). GPU or Apple Silicon (MPS) strongly recommended. Set `--device cuda` or `--device mps` as appropriate.
