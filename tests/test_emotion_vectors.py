import numpy as np
import pytest
from pathlib import Path

from emotion_tracer.emotion_vectors import EmotionVectorBuilder

HIDDEN_DIM = 64

def make_separable_data(seed=0):
    """8 positive samples clearly offset from 8 neutral samples."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(HIDDEN_DIM)
    direction[0] = 1.0
    pos = rng.standard_normal((8, HIDDEN_DIM)) + direction * 5
    neu = rng.standard_normal((8, HIDDEN_DIM))
    return pos, neu

def test_mean_diff_is_unit_norm():
    pos, neu = make_separable_data()
    builder = EmotionVectorBuilder()
    vec = builder.mean_diff(pos, neu)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6

def test_mean_diff_points_in_right_direction():
    pos, neu = make_separable_data()
    builder = EmotionVectorBuilder()
    vec = builder.mean_diff(pos, neu)
    # vec should point toward positive samples
    assert np.dot(vec, pos.mean(axis=0) - neu.mean(axis=0)) > 0

def test_save_and_load_npz(tmp_path):
    pos, neu = make_separable_data()
    builder = EmotionVectorBuilder()
    vec = builder.mean_diff(pos, neu)

    builder.vectors["happy"] = vec
    save_path = tmp_path / "vectors.npz"
    builder.save(str(save_path))

    builder2 = EmotionVectorBuilder()
    builder2.load(str(save_path))
    np.testing.assert_array_almost_equal(builder2.vectors["happy"], vec)
