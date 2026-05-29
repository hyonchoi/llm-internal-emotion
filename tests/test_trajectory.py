import numpy as np
import pytest

from emotion_tracer.trajectory import TrajectoryExtractor

HIDDEN_DIM = 64
SEQ_LEN = 12

def make_sad_hidden_states():
    """Hidden states clearly aligned with 'sad' direction (first dimension)."""
    rng = np.random.default_rng(7)
    hs = rng.standard_normal((SEQ_LEN, HIDDEN_DIM))
    hs[:, 0] += 5.0  # strongly activate dimension 0
    return hs

def make_emotion_vectors():
    """Emotion vectors: 'sad' aligned with dim 0, others perpendicular."""
    vecs = {}
    sad = np.zeros(HIDDEN_DIM)
    sad[0] = 1.0
    vecs["sad"] = sad
    happy = np.zeros(HIDDEN_DIM)
    happy[1] = 1.0
    vecs["happy"] = happy
    angry = np.zeros(HIDDEN_DIM)
    angry[2] = 1.0
    vecs["angry"] = angry
    return vecs

def test_output_shape():
    hs = make_sad_hidden_states()
    vecs = make_emotion_vectors()
    extractor = TrajectoryExtractor()
    scores = extractor.extract(hs, vecs)
    assert scores.shape == (len(vecs), SEQ_LEN)

def test_sad_text_ranks_sad_highest():
    hs = make_sad_hidden_states()
    vecs = make_emotion_vectors()
    extractor = TrajectoryExtractor()
    scores = extractor.extract(hs, vecs)
    emotions = list(vecs.keys())
    mean_scores = scores.mean(axis=1)
    ranked = sorted(zip(emotions, mean_scores), key=lambda x: -x[1])
    assert ranked[0][0] == "sad", f"Expected 'sad' top, got {ranked[0][0]}"

def test_tokens_extracted():
    hs = make_sad_hidden_states()
    vecs = make_emotion_vectors()
    extractor = TrajectoryExtractor()
    _, tokens = extractor.extract_with_tokens(
        hs, vecs, ["tok"] * SEQ_LEN
    )
    assert len(tokens) == SEQ_LEN
