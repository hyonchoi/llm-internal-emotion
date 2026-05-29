import numpy as np
import pytest
from unittest.mock import MagicMock
import torch

from emotion_tracer.config import EmotionTracerConfig

HIDDEN_DIM = 64
NUM_LAYERS = 6
SEQ_LEN = 10

@pytest.fixture
def config():
    return EmotionTracerConfig(
        model_name="mock-model",
        layer_indices=[4],
        emotions=["happy", "sad", "angry"],
        stories_per_emotion=4,
        device="cpu",
        cache_dir="/tmp/test_emotion_vectors",
    )

@pytest.fixture
def mock_model(config):
    """Minimal mock that looks like a HuggingFace causal LM."""
    model = MagicMock()
    tokenizer = MagicMock()

    def fake_tokenize(text, return_tensors="pt"):
        seq = min(len(text.split()), SEQ_LEN)
        return {
            "input_ids": torch.zeros(1, seq, dtype=torch.long),
            "attention_mask": torch.ones(1, seq, dtype=torch.long),
        }

    tokenizer.side_effect = fake_tokenize
    tokenizer.convert_ids_to_tokens = lambda ids: [f"tok{i}" for i in range(len(ids))]

    class FakeOutput:
        def __init__(self, seq):
            self.hidden_states = tuple(
                torch.randn(1, seq, HIDDEN_DIM) for _ in range(NUM_LAYERS + 1)
            )

    def fake_forward(**kwargs):
        seq = kwargs["input_ids"].shape[1]
        return FakeOutput(seq)

    model.side_effect = fake_forward
    model.config = MagicMock()
    model.config.num_hidden_layers = NUM_LAYERS
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    return model, tokenizer

@pytest.fixture
def sample_hidden_states():
    """Pre-built hidden states: 8 emotion samples + 8 neutral."""
    rng = np.random.default_rng(42)
    direction = np.zeros(HIDDEN_DIM)
    direction[0] = 1.0
    pos = rng.standard_normal((8, HIDDEN_DIM)) + direction * 3
    neu = rng.standard_normal((8, HIDDEN_DIM))
    return pos, neu
