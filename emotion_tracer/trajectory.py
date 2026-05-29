import numpy as np

class TrajectoryExtractor:
    """Extract emotion trajectory from hidden states via dot-product projection."""

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
