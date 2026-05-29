import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.model_wrapper import ModelWrapper

HIDDEN_DIM = 64
NUM_LAYERS = 6

def make_fake_loader(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
    """Returns (fake_model, fake_tokenizer) pair suitable for patching."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    tokenizer.convert_ids_to_tokens.return_value = [f"tok{i}" for i in range(5)]

    model = MagicMock()
    model.config.num_hidden_layers = num_layers
    model.eval.return_value = model
    model.to.return_value = model

    class FakeOut:
        hidden_states = tuple(
            torch.randn(1, 5, hidden_dim, dtype=torch.bfloat16) for _ in range(num_layers + 1)
        )

    model.return_value = FakeOut()

    return model, tokenizer

def test_get_hidden_states_shape():
    cfg = EmotionTracerConfig(
        model_name="mock",
        layer_indices=[4],
        device="cpu",
    )
    fake_model, fake_tok = make_fake_loader()

    with patch("emotion_tracer.model_wrapper.AutoModelForCausalLM") as mock_cls, \
         patch("emotion_tracer.model_wrapper.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = fake_model
        mock_tok_cls.from_pretrained.return_value = fake_tok

        wrapper = ModelWrapper(cfg)
        wrapper.load()
        result = wrapper.get_hidden_states("hello world", layer_indices=[4])

    assert 4 in result
    arr = result[4]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, HIDDEN_DIM)  # [seq_len, hidden_dim]

def test_get_tokens():
    cfg = EmotionTracerConfig(model_name="mock", layer_indices=[4], device="cpu")
    fake_model, fake_tok = make_fake_loader()

    with patch("emotion_tracer.model_wrapper.AutoModelForCausalLM") as mock_cls, \
         patch("emotion_tracer.model_wrapper.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = fake_model
        mock_tok_cls.from_pretrained.return_value = fake_tok

        wrapper = ModelWrapper(cfg)
        wrapper.load()
        tokens = wrapper.get_tokens("hello world")

    assert len(tokens) == 5
    assert tokens[0] == "tok0"

def test_num_layers():
    cfg = EmotionTracerConfig(model_name="mock", layer_indices=[], device="cpu")
    fake_model, fake_tok = make_fake_loader()

    with patch("emotion_tracer.model_wrapper.AutoModelForCausalLM") as mock_cls, \
         patch("emotion_tracer.model_wrapper.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = fake_model
        mock_tok_cls.from_pretrained.return_value = fake_tok

        wrapper = ModelWrapper(cfg)
        wrapper.load()

    assert wrapper.num_layers == NUM_LAYERS
