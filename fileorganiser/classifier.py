import math
import re
import pickle
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union
from zipfile import ZipFile

class NaiveBayesFileClassifier:
    """Simple multinomial Naive Bayes classifier for file contents."""

    def __init__(self) -> None:
        self.vocab: set[str] = set()
        self.class_token_counts: Dict[str, Counter] = {}
        self.class_counts: Dict[str, int] = {}
        self.total_files = 0

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t]

    def _extract_pdf(self, path: Path) -> str:
        data = path.read_bytes()
        matches = re.findall(rb"\(([^\)]{1,200})\)", data)
        return " ".join(m.decode("latin1", errors="ignore") for m in matches)

    def _extract_office(self, path: Path) -> str:
        text_parts: list[str] = []
        try:
            with ZipFile(path) as z:
                if path.suffix.lower() == ".docx" and "word/document.xml" in z.namelist():
                    with z.open("word/document.xml") as f:
                        tree = ET.parse(f)
                        text_parts.extend(t.text or "" for t in tree.iter() if t.text)
                elif path.suffix.lower() == ".pptx":
                    for name in z.namelist():
                        if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                            with z.open(name) as f:
                                tree = ET.parse(f)
                                text_parts.extend(t.text or "" for t in tree.iter() if t.text)
                elif path.suffix.lower() == ".xlsx":
                    if "xl/sharedStrings.xml" in z.namelist():
                        with z.open("xl/sharedStrings.xml") as f:
                            tree = ET.parse(f)
                            text_parts.extend(t.text or "" for t in tree.iter() if t.text)
                    for name in z.namelist():
                        if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                            with z.open(name) as f:
                                tree = ET.parse(f)
                                text_parts.extend(t.text or "" for t in tree.iter() if t.text)
        except Exception:
            return ""
        return " ".join(text_parts)

    def _read_file(self, path: Path) -> str:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                return self._extract_pdf(path)
            elif ext in {".docx", ".pptx", ".xlsx"}:
                return self._extract_office(path)
            else:
                return path.read_text(errors="ignore")
        except Exception:
            return ""

    def fit(self, data: Dict[str, Iterable[Union[str, Path]]]) -> None:
        for cls, files in data.items():
            file_paths = [Path(f) for f in files]
            self.class_counts[cls] = len(file_paths)
            token_counter = Counter()
            for path in file_paths:
                text = path.name + " " + self._read_file(path)
                tokens = self._tokenize(text)
                token_counter.update(tokens)
                self.vocab.update(tokens)
            self.class_token_counts[cls] = token_counter
        self.total_files = sum(self.class_counts.values())

    def predict(self, file: Union[str, Path]) -> str:
        path = Path(file)
        text = path.name + " " + self._read_file(path)
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
