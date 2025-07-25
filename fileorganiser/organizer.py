from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict
import sys

from .classifier import NaiveBayesFileClassifier, PROGRESS_INTERVAL


def learn_structure(root: Path) -> NaiveBayesFileClassifier:
    """Learn folder structure from a root directory recursively."""
    data: Dict[str, list[Path]] = {}
    count = 0
    for file in root.rglob("*"):
        if file.is_file():
            rel_folder = str(file.parent.relative_to(root))
            data.setdefault(rel_folder, []).append(file)
            count += 1
            if PROGRESS_INTERVAL and count % PROGRESS_INTERVAL == 0:
                msg = f"Scanning: {count} files processed"
                print(msg, end="\r", file=sys.stderr, flush=True)
    if count >= PROGRESS_INTERVAL:
        print(f"Scanning complete: {count} files found.", file=sys.stderr)
    clf = NaiveBayesFileClassifier()
    clf.fit(data)
    return clf


def classify_files(clf: NaiveBayesFileClassifier, incoming: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for file in incoming.rglob("*"):
        if file.is_file():
            folder = clf.predict(file)
            rel = str(file.relative_to(incoming))
            mapping[rel] = folder
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
        dest = dest_dir / Path(name).name
        shutil.move(str(src), str(dest))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI-powered file organiser")
    parser.add_argument("training_root", type=Path)
    parser.add_argument("incoming_dir", type=Path)
    parser.add_argument("mapping_file", type=Path)
    parser.add_argument("--apply", action="store_true", help="move files")
    parser.add_argument("--save-model", type=Path, help="path to save trained classifier")
    parser.add_argument("--load-model", type=Path, help="load classifier from file")
    args = parser.parse_args(argv)

    if args.load_model:
        clf = NaiveBayesFileClassifier.load(args.load_model)
    else:
        clf = learn_structure(args.training_root)
        if args.save_model:
            clf.save(args.save_model)
    mapping = classify_files(clf, args.incoming_dir)
    write_mapping(mapping, args.mapping_file)
    if args.apply:
        apply_mapping(mapping, args.incoming_dir, args.training_root)


if __name__ == "__main__":
    main()
