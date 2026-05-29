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
