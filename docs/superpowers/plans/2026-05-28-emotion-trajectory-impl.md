# LLM Internal Emotion Trajectory Extractor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage pipeline to extract and visualize token-level emotion trajectories from Mistral/Mixtral LLM hidden states.

**Architecture:** Stage 1 (`build_vectors.py`) generates synthetic stories, extracts last-token hidden states, and computes per-emotion direction vectors (mean-diff + logistic probe). Stage 2 (`trace.py`) projects each token's hidden state onto those vectors to produce a `[num_emotions, seq_len]` score matrix, then renders heatmap PNG, timeline PNG, JSON, and CSV outputs.

**Tech Stack:** Python 3.11+, uv, PyTorch ≥2.0, HuggingFace Transformers ≥4.40, scikit-learn ≥1.3, matplotlib ≥3.7, seaborn ≥0.12, pandas ≥2.0, numpy ≥1.24, tqdm ≥4.65, pytest

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | uv project config + all dependencies |
| `emotion_tracer/__init__.py` | public re-exports |
| `emotion_tracer/config.py` | `EmotionTracerConfig` dataclass — single source of truth for hyperparams |
| `emotion_tracer/model_wrapper.py` | `ModelWrapper` — loads HF model, registers PyTorch hooks, returns `{layer_idx: np.ndarray}` |
| `emotion_tracer/story_generator.py` | `StoryGenerator` — generates emotion/neutral stories via model text generation; caches to disk |
| `emotion_tracer/emotion_vectors.py` | `EmotionVectorBuilder` — mean-diff and logistic probe; saves/loads `.npz` cache |
| `emotion_tracer/trajectory.py` | `TrajectoryExtractor` — dot-product projection per token; returns `np.ndarray[num_emotions, seq_len]` |
| `emotion_tracer/visualizer.py` | `Visualizer` — heatmap PNG, timeline PNG, JSON, CSV |
| `scripts/build_vectors.py` | CLI: orchestrates story generation → vector computation → cache save |
| `scripts/trace.py` | CLI: loads cached vectors → runs trajectory → calls visualizer |
| `examples/example_conversation.py` | Runnable example with a short hard-coded conversation |
| `tests/conftest.py` | Shared pytest fixtures (tiny mock model, small hidden states) |
| `tests/test_config.py` | Config defaults and `resolve_layer_indices` |
| `tests/test_model_wrapper.py` | Hook capture shape check |
| `tests/test_story_generator.py` | Cache hit/miss, story count |
| `tests/test_emotion_vectors.py` | mean-diff L2 norm, probe accuracy > 70%, npz round-trip |
| `tests/test_trajectory.py` | Sad text ranks "sad" in top-3 |
| `tests/test_visualizer.py` | PNG/JSON/CSV file creation |
| `docs/explanation/what_are_emotion_vectors.md` | Non-expert: what are emotion vectors |
| `docs/explanation/how_to_read_the_plots.md` | Non-expert: plot interpretation |
| `docs/explanation/paper_summary.md` | Non-expert: paper key findings |
| `docs/explanation/limitations.md` | Non-expert: what this tool cannot do |
| `README.md` | Install + quick start + output interpretation |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `emotion_tracer/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `data/emotion_vectors/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "llm-internal-emotion"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "scikit-learn>=1.3",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pandas>=2.0",
    "tqdm>=4.65",
    "accelerate>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["emotion_tracer"]
```

- [ ] **Step 2: Create `emotion_tracer/__init__.py`**

```python
from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.model_wrapper import ModelWrapper
from emotion_tracer.story_generator import StoryGenerator
from emotion_tracer.emotion_vectors import EmotionVectorBuilder
from emotion_tracer.trajectory import TrajectoryExtractor
from emotion_tracer.visualizer import Visualizer

__all__ = [
    "EmotionTracerConfig",
    "ModelWrapper",
    "StoryGenerator",
    "EmotionVectorBuilder",
    "TrajectoryExtractor",
    "Visualizer",
]
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty file)

- [ ] **Step 4: Create `tests/conftest.py`**

```python
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

    # tokenizer returns a dict of tensors
    def fake_tokenize(text, return_tensors="pt"):
        seq = min(len(text.split()), SEQ_LEN)
        return {
            "input_ids": torch.zeros(1, seq, dtype=torch.long),
            "attention_mask": torch.ones(1, seq, dtype=torch.long),
        }

    tokenizer.side_effect = fake_tokenize
    tokenizer.convert_ids_to_tokens = lambda ids: [f"tok{i}" for i in range(len(ids))]

    # model forward returns a namedtuple-like with hidden_states
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
    # emotion direction is [1, 0, 0, ...]
    direction = np.zeros(HIDDEN_DIM)
    direction[0] = 1.0
    pos = rng.standard_normal((8, HIDDEN_DIM)) + direction * 3
    neu = rng.standard_normal((8, HIDDEN_DIM))
    return pos, neu
```

- [ ] **Step 5: Create directory stubs**

```bash
mkdir -p emotion_tracer scripts tests examples data/emotion_vectors docs/explanation
touch data/emotion_vectors/.gitkeep
touch scripts/__init__.py
touch examples/__init__.py
```

- [ ] **Step 6: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
data/emotion_vectors/*.npz
results/
*.png
*.csv
*.json
.env
```

- [ ] **Step 7: Install dependencies**

```bash
uv sync --extra dev
```

Expected: resolves without errors, `.venv/` created.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml emotion_tracer/__init__.py tests/__init__.py tests/conftest.py data/emotion_vectors/.gitkeep .gitignore scripts/__init__.py examples/__init__.py
git commit -m "feat: project scaffolding and dependencies"
```

---

## Task 2: `config.py`

**Files:**
- Create: `emotion_tracer/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config.py`:

```python
from emotion_tracer.config import EmotionTracerConfig


def test_default_emotions():
    cfg = EmotionTracerConfig()
    assert "happy" in cfg.emotions
    assert "sad" in cfg.emotions
    assert len(cfg.emotions) == 20


def test_resolve_layer_indices_default():
    cfg = EmotionTracerConfig()
    # With 32 layers, 2/3 depth = index 21
    indices = cfg.resolve_layer_indices(32)
    assert indices == [21]


def test_resolve_layer_indices_explicit():
    cfg = EmotionTracerConfig(layer_indices=[10, 20])
    indices = cfg.resolve_layer_indices(32)
    assert indices == [10, 20]


def test_default_device_is_cpu():
    cfg = EmotionTracerConfig()
    assert cfg.device == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `config.py` doesn't exist yet.

- [ ] **Step 3: Implement `emotion_tracer/config.py`**

```python
from dataclasses import dataclass, field


DEFAULT_EMOTIONS = [
    "happy", "sad", "angry", "afraid", "calm", "anxious",
    "loving", "hopeful", "frustrated", "curious", "disgusted",
    "surprised", "guilty", "proud", "lonely", "grateful",
    "excited", "bored", "confused", "nervous",
]


@dataclass
class EmotionTracerConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    layer_indices: list[int] = field(default_factory=list)
    emotions: list[str] = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    stories_per_emotion: int = 20
    device: str = "cpu"
    cache_dir: str = "data/emotion_vectors"

    def resolve_layer_indices(self, num_layers: int) -> list[int]:
        if self.layer_indices:
            return self.layer_indices
        return [int(num_layers * 2 / 3)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add emotion_tracer/config.py tests/test_config.py
git commit -m "feat: EmotionTracerConfig with layer index resolution"
```

---

## Task 3: `model_wrapper.py`

**Files:**
- Create: `emotion_tracer/model_wrapper.py`
- Create: `tests/test_model_wrapper.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model_wrapper.py`:

```python
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
        hidden_states = tuple(torch.randn(1, 5, hidden_dim) for _ in range(num_layers + 1))

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model_wrapper.py -v
```

Expected: `ModuleNotFoundError` — `model_wrapper.py` not created yet.

- [ ] **Step 3: Implement `emotion_tracer/model_wrapper.py`**

```python
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion_tracer.config import EmotionTracerConfig


class ModelWrapper:
    def __init__(self, config: EmotionTracerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            output_hidden_states=True,
            torch_dtype=torch.float32,
        )
        self.model.to(self.config.device)
        self.model.eval()

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def get_tokens(self, text: str) -> list[str]:
        inputs = self.tokenizer(text, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_hidden_states(self, text: str, layer_indices: list[int]) -> dict[int, np.ndarray]:
        """Return {layer_idx: np.ndarray[seq_len, hidden_dim]} for each requested layer."""
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, hidden_dim]
        # index 0 = embedding layer, index i = layer i output
        result = {}
        for idx in layer_indices:
            # layer_indices are 0-based transformer layer indices; offset by 1 for embedding
            hs = outputs.hidden_states[idx + 1]  # [1, seq_len, hidden_dim]
            result[idx] = hs[0].cpu().numpy()  # [seq_len, hidden_dim]
        return result

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Run autoregressive generation and return only the newly generated text."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model_wrapper.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add emotion_tracer/model_wrapper.py tests/test_model_wrapper.py
git commit -m "feat: ModelWrapper with hook-free hidden state extraction"
```

---

## Task 4: `story_generator.py`

**Files:**
- Create: `emotion_tracer/story_generator.py`
- Create: `tests/test_story_generator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_story_generator.py`:

```python
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.story_generator import StoryGenerator


@pytest.fixture
def cfg(tmp_path):
    return EmotionTracerConfig(
        model_name="mock",
        emotions=["happy", "sad"],
        stories_per_emotion=3,
        cache_dir=str(tmp_path / "cache"),
    )


def make_wrapper(stories):
    """Wrapper mock that cycles through provided stories."""
    wrapper = MagicMock()
    wrapper.generate.side_effect = stories
    return wrapper


def test_generates_correct_count(cfg):
    wrapper = make_wrapper(["story"] * 100)
    gen = StoryGenerator(cfg, wrapper)
    result = gen.get_stories("happy")
    assert len(result) == cfg.stories_per_emotion


def test_neutral_stories_generated(cfg):
    wrapper = make_wrapper(["story"] * 100)
    gen = StoryGenerator(cfg, wrapper)
    result = gen.get_neutral_stories()
    assert len(result) == cfg.stories_per_emotion


def test_cache_prevents_regeneration(cfg):
    wrapper = make_wrapper(["story"] * 100)
    gen = StoryGenerator(cfg, wrapper)
    gen.get_stories("happy")
    call_count_after_first = wrapper.generate.call_count

    gen2 = StoryGenerator(cfg, wrapper)
    gen2.get_stories("happy")
    # call count must not increase after cache hit
    assert wrapper.generate.call_count == call_count_after_first


def test_cache_file_created(cfg, tmp_path):
    wrapper = make_wrapper(["story"] * 100)
    gen = StoryGenerator(cfg, wrapper)
    gen.get_stories("happy")
    cache_file = tmp_path / "cache" / "stories_happy.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert len(data) == cfg.stories_per_emotion
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_story_generator.py -v
```

Expected: `ModuleNotFoundError` — `story_generator.py` not created yet.

- [ ] **Step 3: Implement `emotion_tracer/story_generator.py`**

```python
import json
import os
from pathlib import Path

from emotion_tracer.config import EmotionTracerConfig


EMOTION_PROMPT = (
    "Write a short story (3-4 sentences) where the main character clearly feels {emotion}. "
    "Be concise and emotionally vivid."
)
NEUTRAL_PROMPT = (
    "Write a short neutral story (3-4 sentences) about a person's ordinary daily routine. "
    "Do not mention any emotions."
)


class StoryGenerator:
    def __init__(self, config: EmotionTracerConfig, model_wrapper):
        self.config = config
        self.wrapper = model_wrapper
        self._cache_dir = Path(config.cache_dir)

    def get_stories(self, emotion: str) -> list[str]:
        cache_path = self._cache_dir / f"stories_{emotion}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        stories = self._generate(EMOTION_PROMPT.format(emotion=emotion))
        self._save(cache_path, stories)
        return stories

    def get_neutral_stories(self) -> list[str]:
        cache_path = self._cache_dir / "stories_neutral.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        stories = self._generate(NEUTRAL_PROMPT)
        self._save(cache_path, stories)
        return stories

    def _generate(self, prompt: str) -> list[str]:
        stories = []
        for _ in range(self.config.stories_per_emotion):
            text = self.wrapper.generate(prompt, max_new_tokens=120)
            stories.append(text.strip())
        return stories

    def _save(self, path: Path, stories: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stories, ensure_ascii=False, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_story_generator.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add emotion_tracer/story_generator.py tests/test_story_generator.py
git commit -m "feat: StoryGenerator with disk caching"
```

---

## Task 5: `emotion_vectors.py` — mean-diff method

**Files:**
- Create: `emotion_tracer/emotion_vectors.py`
- Create: `tests/test_emotion_vectors.py`

- [ ] **Step 1: Write failing tests for mean-diff**

Create `tests/test_emotion_vectors.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_emotion_vectors.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `emotion_tracer/emotion_vectors.py` (mean-diff only)**

```python
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression


class EmotionVectorBuilder:
    def __init__(self):
        self.vectors: dict[str, np.ndarray] = {}

    def mean_diff(self, pos_hidden: np.ndarray, neu_hidden: np.ndarray) -> np.ndarray:
        """Compute L2-normalized mean difference vector.

        Args:
            pos_hidden: shape [n_pos, hidden_dim] — hidden states from emotion-positive stories
            neu_hidden: shape [n_neu, hidden_dim] — hidden states from neutral stories

        Returns:
            unit-norm vector of shape [hidden_dim]
        """
        diff = pos_hidden.mean(axis=0) - neu_hidden.mean(axis=0)
        norm = np.linalg.norm(diff)
        return diff / norm

    def logistic_probe(self, pos_hidden: np.ndarray, neu_hidden: np.ndarray) -> np.ndarray:
        """Fit a logistic regression probe; return L2-normalized coefficient vector.

        Args:
            pos_hidden: shape [n_pos, hidden_dim]
            neu_hidden: shape [n_neu, hidden_dim]

        Returns:
            unit-norm vector of shape [hidden_dim]
        """
        X = np.concatenate([pos_hidden, neu_hidden], axis=0)
        y = np.array([1] * len(pos_hidden) + [0] * len(neu_hidden))
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        coef = clf.coef_[0]
        return coef / np.linalg.norm(coef)

    def probe_accuracy(self, pos_hidden: np.ndarray, neu_hidden: np.ndarray) -> float:
        """Return cross-validated accuracy of a logistic probe on the given data."""
        from sklearn.model_selection import cross_val_score
        X = np.concatenate([pos_hidden, neu_hidden], axis=0)
        y = np.array([1] * len(pos_hidden) + [0] * len(neu_hidden))
        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, y, cv=min(4, len(pos_hidden) // 2))
        return float(scores.mean())

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.vectors)

    def load(self, path: str):
        data = np.load(path)
        self.vectors = {k: data[k] for k in data.files}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_emotion_vectors.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add emotion_tracer/emotion_vectors.py tests/test_emotion_vectors.py
git commit -m "feat: EmotionVectorBuilder with mean-diff and logistic probe"
```

---

## Task 6: `emotion_vectors.py` — logistic probe tests

**Files:**
- Modify: `tests/test_emotion_vectors.py`

- [ ] **Step 1: Add probe accuracy test**

Append to `tests/test_emotion_vectors.py`:

```python
def test_probe_accuracy_above_70_percent():
    """Logistic probe on clearly separable data must exceed 70% CV accuracy."""
    pos, neu = make_separable_data()
    builder = EmotionVectorBuilder()
    acc = builder.probe_accuracy(pos, neu)
    assert acc > 0.70, f"Expected >70% accuracy, got {acc:.2%}"


def test_logistic_probe_is_unit_norm():
    pos, neu = make_separable_data()
    builder = EmotionVectorBuilder()
    vec = builder.logistic_probe(pos, neu)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_emotion_vectors.py -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_emotion_vectors.py
git commit -m "test: add logistic probe accuracy and norm tests"
```

---

## Task 7: `trajectory.py`

**Files:**
- Create: `emotion_tracer/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trajectory.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trajectory.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `emotion_tracer/trajectory.py`**

```python
import numpy as np


class TrajectoryExtractor:
    def extract(
        self,
        hidden_states: np.ndarray,
        emotion_vectors: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute dot-product projection of each token hidden state onto each emotion vector.

        Args:
            hidden_states: shape [seq_len, hidden_dim]
            emotion_vectors: {emotion_name: unit-norm vector [hidden_dim]}

        Returns:
            scores: shape [num_emotions, seq_len]
        """
        emotions = list(emotion_vectors.keys())
        matrix = np.stack([emotion_vectors[e] for e in emotions], axis=0)  # [E, D]
        scores = matrix @ hidden_states.T  # [E, seq_len]
        return scores

    def extract_with_tokens(
        self,
        hidden_states: np.ndarray,
        emotion_vectors: dict[str, np.ndarray],
        tokens: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Return (scores [E, seq_len], tokens)."""
        return self.extract(hidden_states, emotion_vectors), tokens
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_trajectory.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add emotion_tracer/trajectory.py tests/test_trajectory.py
git commit -m "feat: TrajectoryExtractor with dot-product projection"
```

---

## Task 8: `visualizer.py`

**Files:**
- Create: `emotion_tracer/visualizer.py`
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_visualizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_visualizer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `emotion_tracer/visualizer.py`**

```python
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# Colors for Human/Assistant turn background bands
TURN_COLORS = {"Human": "#fff3cd", "Assistant": "#d1ecf1"}


class Visualizer:
    def plot_heatmap(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        turn_boundaries: list[tuple] | None = None,
        output_path: str = "heatmap.png",
    ):
        """Heatmap: X=tokens, Y=emotions, color=activation strength.

        Args:
            scores: [num_emotions, seq_len]
            tokens: token strings for x-axis labels
            emotions: emotion names for y-axis labels
            turn_boundaries: list of (start_idx, end_idx, label) for background shading
            output_path: destination PNG path
        """
        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), max(4, len(emotions) * 0.4)))

        if turn_boundaries:
            for start, end, label in turn_boundaries:
                color = TURN_COLORS.get(label, "#f8f9fa")
                ax.axvspan(start - 0.5, end + 0.5, alpha=0.3, color=color, zorder=0)

        sns.heatmap(
            scores,
            ax=ax,
            xticklabels=tokens,
            yticklabels=emotions,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Activation"},
        )
        ax.set_xlabel("Token position")
        ax.set_ylabel("Emotion")
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

    def plot_timeline(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        top_k: int = 5,
        output_path: str = "timeline.png",
    ):
        """Line graph of top-K emotions across token positions.

        Args:
            scores: [num_emotions, seq_len]
            top_k: number of most active emotions to plot
        """
        mean_scores = scores.mean(axis=1)
        top_indices = np.argsort(mean_scores)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.5), 4))
        x = np.arange(len(tokens))
        for idx in top_indices:
            ax.plot(x, scores[idx], label=emotions[idx], linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Emotion activation")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

    def save_json(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        output_path: str = "trajectory.json",
    ):
        """Save per-token emotion scores as JSON."""
        records = []
        for t, token in enumerate(tokens):
            records.append({
                "token": token,
                "position": t,
                "emotion_scores": {e: float(scores[i, t]) for i, e in enumerate(emotions)},
            })
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(records, ensure_ascii=False, indent=2))

    def save_csv(
        self,
        scores: np.ndarray,
        tokens: list[str],
        emotions: list[str],
        output_path: str = "trajectory.csv",
    ):
        """Save per-token emotion scores as CSV (rows=tokens, cols=emotions)."""
        df = pd.DataFrame(scores.T, columns=emotions)
        df.insert(0, "token", tokens)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_visualizer.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add emotion_tracer/visualizer.py tests/test_visualizer.py
git commit -m "feat: Visualizer with heatmap, timeline, JSON, CSV output"
```

---

## Task 9: `scripts/build_vectors.py`

**Files:**
- Create: `scripts/build_vectors.py`

- [ ] **Step 1: Implement `scripts/build_vectors.py`**

```python
#!/usr/bin/env python3
"""Pre-compute emotion vectors and save to cache.

Usage:
    python scripts/build_vectors.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --emotions happy sad angry afraid calm \
        --method both \
        --cache-dir data/emotion_vectors \
        --stories 20 \
        --device cuda
"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from emotion_tracer.config import EmotionTracerConfig
from emotion_tracer.model_wrapper import ModelWrapper
from emotion_tracer.story_generator import StoryGenerator
from emotion_tracer.emotion_vectors import EmotionVectorBuilder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--emotions", nargs="+", default=["happy", "sad", "angry", "afraid", "calm"])
    p.add_argument("--method", choices=["mean-diff", "logistic-probe", "both"], default="both")
    p.add_argument("--cache-dir", default="data/emotion_vectors")
    p.add_argument("--stories", type=int, default=20)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = EmotionTracerConfig(
        model_name=args.model,
        emotions=args.emotions,
        stories_per_emotion=args.stories,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    print(f"Loading model: {cfg.model_name}")
    wrapper = ModelWrapper(cfg)
    wrapper.load()
    layer_indices = cfg.resolve_layer_indices(wrapper.num_layers)
    print(f"Using layers: {layer_indices}")

    gen = StoryGenerator(cfg, wrapper)

    print("Collecting neutral story activations...")
    neutral_stories = gen.get_neutral_stories()
    neutral_hs_per_layer = {l: [] for l in layer_indices}
    for story in tqdm(neutral_stories, desc="neutral"):
        hs_map = wrapper.get_hidden_states(story, layer_indices)
        for l in layer_indices:
            neutral_hs_per_layer[l].append(hs_map[l][-1])  # last token

    for l in layer_indices:
        neutral_hs_per_layer[l] = np.stack(neutral_hs_per_layer[l])

    methods = ["mean-diff", "logistic-probe"] if args.method == "both" else [args.method]

    for method in methods:
        builder = EmotionVectorBuilder()
        for emotion in tqdm(cfg.emotions, desc=f"building vectors ({method})"):
            stories = gen.get_stories(emotion)
            pos_hs_per_layer = {l: [] for l in layer_indices}
            for story in stories:
                hs_map = wrapper.get_hidden_states(story, layer_indices)
                for l in layer_indices:
                    pos_hs_per_layer[l].append(hs_map[l][-1])
            for l in layer_indices:
                pos_hs_per_layer[l] = np.stack(pos_hs_per_layer[l])

            # Use first (only) layer for now; extend to multi-layer if needed
            l = layer_indices[0]
            if method == "mean-diff":
                vec = builder.mean_diff(pos_hs_per_layer[l], neutral_hs_per_layer[l])
            else:
                vec = builder.logistic_probe(pos_hs_per_layer[l], neutral_hs_per_layer[l])
            builder.vectors[emotion] = vec

        save_path = Path(args.cache_dir) / f"vectors_{method}_layer{layer_indices[0]}.npz"
        builder.save(str(save_path))
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable without errors**

```bash
python -c "import scripts.build_vectors"
```

Expected: no output (no import errors).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_vectors.py
git commit -m "feat: build_vectors CLI script"
```

---

## Task 10: `scripts/trace.py`

**Files:**
- Create: `scripts/trace.py`

- [ ] **Step 1: Implement `scripts/trace.py`**

```python
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
```

- [ ] **Step 2: Verify importable**

```bash
python -c "import scripts.trace"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/trace.py
git commit -m "feat: trace CLI script for conversation trajectory"
```

---

## Task 11: Example and Documentation

**Files:**
- Create: `examples/example_conversation.py`
- Create: `docs/explanation/what_are_emotion_vectors.md`
- Create: `docs/explanation/how_to_read_the_plots.md`
- Create: `docs/explanation/paper_summary.md`
- Create: `docs/explanation/limitations.md`
- Create: `README.md`

- [ ] **Step 1: Create `examples/example_conversation.py`**

```python
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
```

- [ ] **Step 2: Create `docs/explanation/what_are_emotion_vectors.md`**

```markdown
# What Are Emotion Vectors?

## The Basic Idea

Imagine a large language model as a vast, high-dimensional space — thousands of dimensions, each representing some abstract concept. When the model processes text, it moves through this space, creating a unique point (called a "hidden state") for each word it reads.

Researchers have discovered that emotions correspond to specific *directions* in this space. "Happy" is roughly one direction; "sad" is roughly another. These directions are called **emotion vectors**.

## How We Find Them

1. We generate dozens of short stories where a character explicitly feels a specific emotion (e.g., "She beamed with joy as she opened the gift").
2. We generate equally many "neutral" versions — same structure, but no emotion ("She opened the gift").
3. We run both sets through the model and record the internal activations.
4. We subtract the average neutral activation from the average emotional activation.
5. The result, after normalizing to unit length, is the **emotion vector** for that emotion.

Think of it like tuning a compass: the emotional stories pull the needle in one direction; neutral stories stay put; the difference tells us which way "happy" points.

## Why Does It Work?

Because the model learned from human text, and human text is soaked in emotion. The model implicitly learned to represent emotions in its internal geometry — not because we told it to, but because it had to in order to predict text accurately.
```

- [ ] **Step 3: Create `docs/explanation/how_to_read_the_plots.md`**

```markdown
# How to Read the Emotion Plots

## The Heatmap

The heatmap shows the full emotion picture across every token in the conversation.

- **X-axis** — each column is one token (word or punctuation mark) in the conversation
- **Y-axis** — each row is one emotion
- **Color** — red means that emotion is strongly active; blue means it is suppressed; white is neutral

**What to look for:**
- A bright red patch in the "sad" row around the word "died" tells you: the model internally registered sadness at that point
- Yellow/cream background = Human turn; light blue background = Assistant turn

## The Timeline

The timeline zooms in on the top-5 most active emotions and shows how each one rises and falls across the conversation.

- Each line is one emotion
- Peaks = the model is strongly activating that emotion at that token
- Valleys = that emotion is suppressed

The token just before "Assistant:" (the colon token) is especially interesting — research shows that the model's emotion at *that exact point* predicts the emotional tone of the response it's about to generate.

## The Numbers (JSON / CSV)

Each row in the CSV corresponds to one token. The numeric value in each emotion column is the "dot product" — how strongly that token's hidden state points in the emotion's direction. Positive = active; negative = suppressed; near zero = neutral.
```

- [ ] **Step 4: Create `docs/explanation/paper_summary.md`**

```markdown
# Paper Summary: Emotions Inside a Language Model

**Source:** "Emotion Concepts and their Function in a Large Language Model"  
Sofroniew et al., Anthropic (arXiv:2604.07729v1)

## What Did They Discover?

1. **Emotions have geometry.** Inside Claude, emotions correspond to specific directions in the model's internal space — just like "up" and "down" are directions in physical space.

2. **These directions are causal.** It's not just that the model *represents* emotions — they actually influence what the model says next. When researchers artificially amplified the "fear" direction, the model started generating fearful text. When they suppressed it, the fear disappeared.

3. **Different layers tell different stories.** Early layers encode the emotional meaning of the *current* token. Later layers (around 2/3 depth) encode the emotion that will *drive* the next generated token.

4. **The colon predicts the response.** In Claude's conversation format, the ":" token before the Assistant's response carries the strongest signal about what emotional tone the response will have.

## Why Does This Matter?

This research suggests that "emotion" in language models isn't just a metaphor — it's a measurable, functional internal state that can be studied, amplified, suppressed, or redirected.

This tool applies these findings to open-source Mistral models, making the same kind of analysis accessible without needing access to Claude.
```

- [ ] **Step 5: Create `docs/explanation/limitations.md`**

```markdown
# Limitations of This Approach

## What This Tool Cannot Tell You

### 1. Whether the model "feels" anything
The numbers this tool produces are geometric measurements — how much an internal vector points in the direction we labeled "happy." They do not tell us whether the model has any subjective experience. "Activation" ≠ "feeling."

### 2. Whether the labeling is correct
We call a direction "sadness" because it activates when we show the model sad stories. But the same direction might be capturing something adjacent — loss, reflection, downward valence. The label is ours, not the model's.

### 3. Persistent emotional states
Transformer models have no memory between tokens except what fits in the context window. Each token's hidden state is freshly computed from the full context. The model doesn't "carry" an emotion the way a person does — it re-reads the entire conversation and re-computes everything at each step.

### 4. Generalization across models
Emotion vectors computed from Mistral 7B may not transfer to Mixtral 8x7B or a fine-tuned variant. Each model has its own internal geometry.

### 5. External validation
We have no ground truth for what the model's "real" internal state is. We can only measure structure in the activations. Whether that structure corresponds to anything meaningful is a scientific question, not an engineering one.

## Bottom Line

Use this tool as an **exploratory instrument**, not a measurement device. It can surface interesting patterns and generate hypotheses. It cannot prove or disprove that a model has emotions.
```

- [ ] **Step 6: Create `README.md`**

```markdown
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
```

- [ ] **Step 7: Run all tests one final time**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add examples/example_conversation.py docs/explanation/ README.md
git commit -m "feat: examples, explanation docs, and README"
```

---

## Self-Review

### Spec Coverage Check

| Spec Section | Covered by Task |
|---|---|
| `config.py` with `EmotionTracerConfig` dataclass | Task 2 |
| `model_wrapper.py` with PyTorch hook / hidden state extraction | Task 3 |
| `story_generator.py` with prompt templates + cache | Task 4 |
| `emotion_vectors.py` mean-diff (L2 norm) | Task 5 |
| `emotion_vectors.py` logistic probe (accuracy > 70%) | Task 6 |
| `.npz` cache save/load | Task 5 |
| `trajectory.py` dot-product projection → `[E, seq_len]` | Task 7 |
| `visualizer.py` heatmap PNG | Task 8 |
| `visualizer.py` timeline PNG | Task 8 |
| `visualizer.py` JSON + CSV | Task 8 |
| Human/Assistant turn background shading | Task 8 (`turn_boundaries` param) |
| `build_vectors.py` CLI | Task 9 |
| `trace.py` CLI | Task 10 |
| `docs/explanation/` (4 files) | Task 11 |
| `README.md` | Task 11 |
| `pyproject.toml` with all deps | Task 1 |
| Tests: model_wrapper shape | Task 3 |
| Tests: mean-diff L2 norm | Task 5 |
| Tests: probe accuracy > 70% | Task 6 |
| Tests: sad text ranks sad highest | Task 7 |
| Tests: PNG/JSON/CSV created | Task 8 |

All spec requirements covered. No gaps found.

### Placeholder Scan

No TBDs, TODOs, or "similar to Task N" references found.

### Type Consistency

- `EmotionTracerConfig` defined in Task 2, used in Tasks 3, 4, 5, 9, 10 — consistent
- `ModelWrapper.get_hidden_states(text, layer_indices) -> dict[int, np.ndarray]` — defined Task 3, used Tasks 9, 10 — consistent
- `EmotionVectorBuilder.vectors: dict[str, np.ndarray]` — defined Task 5, accessed Tasks 9, 10 — consistent
- `TrajectoryExtractor.extract(hidden_states, emotion_vectors) -> np.ndarray` — defined Task 7, used Tasks 10, 11 — consistent

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Voice | `/plan-eng-review` | Independent 2nd opinion | 1 | issues_found | 2 new P1 bugs caught (D10, D11) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 11 issues, 2 critical gaps flagged |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **UNRESOLVED:** 0 decisions unresolved
- **VERDICT:** ENG CLEARED — ready to implement. Address all 10 implementation tasks (T1–T10) during implementation.

### Key decisions from review

| # | Decision | Outcome |
|---|---|---|
| D2 | torch.no_grad() missing in get_hidden_states() | Add it (T1) |
| D3 | build_vectors.py silently uses only layer_indices[0] | Loop over all layers (T6) |
| D4 | Story generation uses probed model (circular) | Add --stories-file option (T7) |
| D5 | DRY: hidden state collection loop duplicated | Extract collect_last_token_hs() helper (T8) |
| D6 | trace.py has no guard for missing cache | Add helpful error message (T9) |
| D7 | test_trajectory uses random mock — 60% false-pass rate | Replace with deterministic math test (T5) |
| D8 | No integration test | Add tests/test_integration.py (T10) |
| D9 | model loads in float32 (28 GB VRAM) | Switch to bfloat16 + device_map='auto' (T3) |
| D10 | Logistic probe class imbalance (outside voice) | Add class_weight='balanced' (T4) |
| D11 | bfloat16 tensor cannot convert to NumPy (outside voice) | Add .float() cast (T2) |
- `Visualizer.plot_heatmap(scores, tokens, emotions, ...)` — defined Task 8, used Tasks 10, 11 — consistent
