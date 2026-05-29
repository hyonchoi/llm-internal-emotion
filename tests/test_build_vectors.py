"""Tests for scripts/build_vectors.py — validate_cache, collect_last_token_hs, load_stories_from_file."""
import json
import numpy as np
from pathlib import Path
import pytest

from scripts.build_vectors import (
    validate_cache,
    collect_last_token_hs,
    load_stories_from_file,
)

# ---------------------------------------------------------------------------
# validate_cache
# ---------------------------------------------------------------------------

class TestValidateCache:
    @pytest.fixture
    def cache_dir(self, tmp_path):
        return str(tmp_path / "vectors")

    @pytest.fixture
    def emotions(self):
        return ["happy", "sad"]

    @pytest.fixture
    def methods(self):
        return ["mean-diff", "logistic-probe"]

    @pytest.fixture
    def layers(self):
        return [16, 24]

    def _make_npz(self, cache_dir, method, layer, emotions):
        p = Path(cache_dir) / f"vectors_{method}_layer{layer}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **{e: np.random.randn(4096) for e in emotions})

    def test_empty_cache(self, cache_dir, emotions, methods, layers):
        assert not validate_cache(cache_dir, emotions, methods, layers)

    def test_complete_cache(self, cache_dir, emotions, methods, layers):
        for method in methods:
            for l in layers:
                self._make_npz(cache_dir, method, l, emotions)
        assert validate_cache(cache_dir, emotions, methods, layers)

    def test_missing_layer(self, cache_dir, emotions, methods, layers):
        # Create all except one layer
        for method in methods:
            for l in layers[:-1]:
                self._make_npz(cache_dir, method, l, emotions)
        assert not validate_cache(cache_dir, emotions, methods, layers)

    def test_missing_method(self, cache_dir, emotions, methods, layers):
        # Create only mean-diff files
        for l in layers:
            self._make_npz(cache_dir, "mean-diff", l, emotions)
        assert not validate_cache(cache_dir, emotions, methods, layers)

    def test_file_missing_emotion_key(self, cache_dir, emotions, methods, layers):
        for method in methods:
            for l in layers:
                p = Path(cache_dir) / f"vectors_{method}_layer{l}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                # Only "happy", not "sad"
                np.savez(p, happy=np.random.randn(4096))
        assert not validate_cache(cache_dir, emotions, methods, layers)

    def test_partial_method_works(self, cache_dir, emotions, layers):
        """When only one method is requested, validate_cache succeeds with fewer files."""
        methods = ["mean-diff"]
        for method in methods:
            for l in layers:
                self._make_npz(cache_dir, method, l, emotions)
        assert validate_cache(cache_dir, emotions, methods, layers)

# ---------------------------------------------------------------------------
# collect_last_token_hs
# ---------------------------------------------------------------------------

class TestCollectLastTokenHs:
    def test_collects_last_token(self):
        """Mock ModelWrapper to verify we extract the last token per layer."""
        class FakeWrapper:
            def __init__(self):
                self.calls = []
            def get_hidden_states(self, text, layer_indices):
                self.calls.append((text, layer_indices))
                return {l: np.arange(l * 10, l * 10 + 10).reshape(2, 5) for l in layer_indices}

        wrapper = FakeWrapper()
        stories = ["story A", "story B"]
        layer_indices = [16, 24]

        result = collect_last_token_hs(wrapper, stories, layer_indices)

        assert len(result) == 2
        assert len(result[16]) == 2
        assert len(result[24]) == 2
        # reshape(2,5): last row is [165-169] for layer 16, [245-249] for layer 24
        assert result[16][0][0] == 165
        assert result[24][1][0] == 245

    def test_calls_wrapper_for_each_story(self):
        class FakeWrapper:
            def __init__(self):
                self.call_count = 0
            def get_hidden_states(self, text, layer_indices):
                self.call_count += 1
                return {l: np.zeros((3, 5)) for l in layer_indices}

        wrapper = FakeWrapper()
        stories = ["A", "B", "C"]
        collect_last_token_hs(wrapper, stories, [16])
        assert wrapper.call_count == 3

# ---------------------------------------------------------------------------
# load_stories_from_file
# ---------------------------------------------------------------------------

class TestLoadStoriesFromFile:
    def test_loads_correctly(self, tmp_path):
        data = {"happy": ["h1", "h2"], "neutral": ["n1"]}
        f = tmp_path / "stories.json"
        f.write_text(json.dumps(data))
        result = load_stories_from_file(str(f), ["happy"])
        assert result["happy"] == ["h1", "h2"]
        assert result["neutral"] == ["n1"]
