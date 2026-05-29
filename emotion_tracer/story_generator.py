import json
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
