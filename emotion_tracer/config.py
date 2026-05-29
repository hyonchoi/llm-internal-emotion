from dataclasses import dataclass, field

DEFAULT_EMOTIONS = [
    "happy", "sad", "angry", "afraid", "calm", "anxious",
    "loving", "hopeful", "frustrated", "curious", "disgusted",
    "surprised", "guilty", "proud", "lonely", "grateful",
    "excited", "bored", "confused", "nervous",
]

@dataclass
class EmotionTracerConfig:
    """Configuration for the LLM emotion trajectory extractor.

    All hyperparameters for the pipeline live here — model name, layer
    indices, emotion list, cache path, etc.
    """
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    layer_indices: list[int] = field(default_factory=list)
    emotions: list[str] = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    stories_per_emotion: int = 20
    device: str = "cpu"
    cache_dir: str = "data/emotion_vectors"

    def resolve_layer_indices(self, num_layers: int) -> list[int]:
        """Return which transformer layers to extract hidden states from.

        If ``layer_indices`` was set explicitly, use those.  Otherwise
        pick the layer at 2/3 depth (the "mid-late" layer where emotion
        signals are most operative per Sofroniew et al.).

        Args:
            num_layers: total number of transformer layers in the model.

        Raises:
            ValueError: if ``num_layers`` is zero or negative and no
                explicit indices were provided.
        """
        if self.layer_indices:
            return list(self.layer_indices)
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0 when layer_indices is not set, got {num_layers}"
            )
        return [num_layers * 2 // 3]
