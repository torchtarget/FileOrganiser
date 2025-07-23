from pathlib import Path
from fileorganiser.organizer import learn_structure, classify_files


def test_classify(tmp_path: Path) -> None:
    root = tmp_path / "root"
    docs = root / "docs"
    images = root / "images"
    code = root / "code"
    docs.mkdir(parents=True)
    images.mkdir()
    code.mkdir()
    (docs / "a.txt").write_text("sample")
    (docs / "b.txt").write_text("sample")
    (images / "x.jpg").write_text("sample")
    (images / "y.png").write_text("sample")
    (code / "main.py").write_text("sample")
    (code / "util.py").write_text("sample")

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "notes.txt").write_text("sample")
    (incoming / "photo.jpg").write_text("sample")
    (incoming / "script.py").write_text("sample")

    clf = learn_structure(root)
    mapping = classify_files(clf, incoming)

    assert mapping["notes.txt"] == "docs"
    assert mapping["photo.jpg"] == "images"
    assert mapping["script.py"] == "code"
