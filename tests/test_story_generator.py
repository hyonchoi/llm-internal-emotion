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
