import math
import pickle
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

class NaiveBayesFileClassifier:
    """Simple multinomial Naive Bayes classifier based on file contents."""

    def __init__(self) -> None:
        self.vocab: set[str] = set()
        self.class_token_counts: Dict[str, Counter] = {}
        self.class_counts: Dict[str, int] = {}
        self.total_files = 0

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t]

    def _extract_text(self, path: Path) -> str:
        """Best-effort extraction of text from various file types."""
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".md", ".py", ".csv", ""}:
                return path.read_text(errors="ignore")
            if suffix == ".pdf":
                try:
                    import PyPDF2  # type: ignore
                    reader = PyPDF2.PdfReader(str(path))
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception:
                    return ""
            if suffix == ".docx":
                try:
                    with zipfile.ZipFile(path) as z:
                        with z.open("word/document.xml") as f:
                            tree = ET.parse(f)
                            return " ".join(e.text or "" for e in tree.iter() if e.text)
                except Exception:
                    return ""
            if suffix == ".pptx":
                text = []
                try:
                    with zipfile.ZipFile(path) as z:
                        for name in z.namelist():
                            if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                                with z.open(name) as f:
                                    tree = ET.parse(f)
                                    text.extend(e.text or "" for e in tree.iter() if e.text)
                    return " ".join(text)
                except Exception:
                    return ""
            if suffix in {".xlsx", ".xls"}:
                text = []
                try:
                    with zipfile.ZipFile(path) as z:
                        for name in z.namelist():
                            if name.startswith("xl/") and name.endswith(".xml"):
                                with z.open(name) as f:
                                    tree = ET.parse(f)
                                    text.extend(e.text or "" for e in tree.iter() if e.text)
                    return " ".join(text)
                except Exception:
                    return ""
            # Fallback: read as text if possible
            return path.read_text(errors="ignore")
        except Exception:
            return ""

    def fit(self, data: Dict[str, Iterable[Path]]) -> None:
        for cls, files in data.items():
            file_list = list(files)
            self.class_counts[cls] = len(file_list)
            token_counter = Counter()
            for fpath in file_list:
                text = self._extract_text(fpath)
                if not text:
                    text = fpath.name
                tokens = self._tokenize(text)
                token_counter.update(tokens)
                self.vocab.update(tokens)
            self.class_token_counts[cls] = token_counter
        self.total_files = sum(self.class_counts.values())

    def predict(self, path: Union[str, Path]) -> str:
        fpath = Path(path)
        text = self._extract_text(fpath)
        if not text:
            text = fpath.name
        tokens = self._tokenize(text)
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
