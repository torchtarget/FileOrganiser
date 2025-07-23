import math
import re
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

class NaiveBayesFileClassifier:
    """Simple multinomial Naive Bayes classifier for file names."""

    def __init__(self) -> None:
        self.vocab: set[str] = set()
        self.class_token_counts: Dict[str, Counter] = {}
        self.class_counts: Dict[str, int] = {}
        self.total_files = 0

    def _tokenize(self, name: str) -> List[str]:
        tokens = re.split(r"[^a-zA-Z0-9]+", name.lower())
        return [t for t in tokens if t]

    def fit(self, data: Dict[str, Iterable[str]]) -> None:
        for cls, files in data.items():
            files = list(files)
            self.class_counts[cls] = len(files)
            token_counter = Counter()
            for fname in files:
                tokens = self._tokenize(Path(fname).name)
                token_counter.update(tokens)
                self.vocab.update(tokens)
            self.class_token_counts[cls] = token_counter
        self.total_files = sum(self.class_counts.values())

    def predict(self, name: str) -> str:
        tokens = self._tokenize(Path(name).name)
        best_cls = None
        best_score = float("-inf")
        for cls in self.class_counts:
            prior = math.log(self.class_counts[cls] / self.total_files)
            denom = sum(self.class_token_counts[cls].values()) + len(self.vocab)
            score = prior
            for t in tokens:
                count = self.class_token_counts[cls].get(t, 0)
                prob = (count + 1) / denom
                score += math.log(prob)
            if score > best_score:
                best_score = score
                best_cls = cls
        assert best_cls is not None
        return best_cls

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the classifier to the given file."""
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NaiveBayesFileClassifier":
        """Load a classifier previously saved with :meth:`save`."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("File does not contain a NaiveBayesFileClassifier")
        return obj
