import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression


class EmotionVectorBuilder:
    """Build emotion direction vectors from hidden states.

    Two methods are provided:

    - **mean-diff**: L2-normalized difference of mean hidden states
      (emotion-positive minus neutral). Fast and simple.

    - **logistic-probe**: Fit a logistic regression classifier and use the
      L2-normalized coefficient vector as the emotion direction. More robust
      when the boundary is non-linear.
    """

    def __init__(self):
        self.vectors: dict[str, np.ndarray] = {}

    def mean_diff(self, pos_hidden: np.ndarray, neu_hidden: np.ndarray) -> np.ndarray:
        """Compute L2-normalized mean difference vector.

        Args:
            pos_hidden: shape [n_pos, hidden_dim] — hidden states from
                emotion-positive stories.
            neu_hidden: shape [n_neu, hidden_dim] — hidden states from
                neutral stories.

        Returns:
            unit-norm vector of shape [hidden_dim].
        """
        diff = pos_hidden.mean(axis=0) - neu_hidden.mean(axis=0)
        norm = np.linalg.norm(diff)
        return diff / norm

    def logistic_probe(self, pos_hidden: np.ndarray, neu_hidden: np.ndarray) -> np.ndarray:
        """Fit a logistic regression probe; return L2-normalized coefficient vector.

        Args:
            pos_hidden: shape [n_pos, hidden_dim].
            neu_hidden: shape [n_neu, hidden_dim].

        Returns:
            unit-norm vector of shape [hidden_dim].
        """
        X = np.concatenate([pos_hidden, neu_hidden], axis=0)
        y = np.array([1] * len(pos_hidden) + [0] * len(neu_hidden))
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(X, y)
        coef = clf.coef_[0]
        return coef / np.linalg.norm(coef)

    def probe_accuracy(self, pos_hidden: np.ndarray, neu_hidden: np.ndarray) -> float:
        """Return cross-validated accuracy of a logistic probe on the given data."""
        from sklearn.model_selection import cross_val_score

        X = np.concatenate([pos_hidden, neu_hidden], axis=0)
        y = np.array([1] * len(pos_hidden) + [0] * len(neu_hidden))
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        scores = cross_val_score(clf, X, y, cv=min(4, len(pos_hidden) // 2))
        return float(scores.mean())

    def save(self, path: str):
        """Save all vectors to an ``.npz`` file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.vectors)

    def load(self, path: str):
        """Load vectors from an ``.npz`` file."""
        data = np.load(path)
        self.vectors = {k: data[k] for k in data.files}
