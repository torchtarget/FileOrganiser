from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

from .classifier import NaiveBayesFileClassifier


def learn_structure(root: Path) -> NaiveBayesFileClassifier:
    """Learn folder structure from a root directory."""
    data: Dict[str, list[str]] = {}
    for child in root.iterdir():
        if child.is_dir():
            files = [f.name for f in child.iterdir() if f.is_file()]
            if files:
                data[child.name] = files
    clf = NaiveBayesFileClassifier()
    clf.fit(data)
    return clf


def classify_files(clf: NaiveBayesFileClassifier, incoming: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for file in incoming.iterdir():
        if file.is_file():
            folder = clf.predict(file.name)
            mapping[file.name] = folder
    return mapping


def write_mapping(mapping: Dict[str, str], path: Path) -> None:
    with path.open("w") as f:
        for name, folder in mapping.items():
            f.write(f"{name} -> {folder}\n")


def apply_mapping(mapping: Dict[str, str], incoming: Path, root: Path) -> None:
    for name, folder in mapping.items():
        src = incoming / name
        dest_dir = root / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        shutil.move(str(src), str(dest))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI-powered file organiser")
    parser.add_argument("training_root", type=Path)
    parser.add_argument("incoming_dir", type=Path)
    parser.add_argument("mapping_file", type=Path)
    parser.add_argument("--apply", action="store_true", help="move files")
    args = parser.parse_args(argv)

    clf = learn_structure(args.training_root)
    mapping = classify_files(clf, args.incoming_dir)
    write_mapping(mapping, args.mapping_file)
    if args.apply:
        apply_mapping(mapping, args.incoming_dir, args.training_root)


if __name__ == "__main__":
    main()
