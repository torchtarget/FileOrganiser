from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fileorganiser.organizer import learn_structure, classify_files
from fileorganiser.classifier import NaiveBayesFileClassifier

def test_classify_contents_and_nested(tmp_path: Path) -> None:
    root = tmp_path / "root"
    docs = root / "docs" / "text"
    images = root / "media" / "images"
    code = root / "code" / "scripts"
    docs.mkdir(parents=True)
    images.mkdir(parents=True)
    code.mkdir(parents=True)
    (docs / "a.dat").write_text("document data")
    (docs / "b.dat").write_text("document file")
    (images / "x.bin").write_text("photo image picture")
    (images / "y.bin").write_text("image photo")
    (code / "main.xxx").write_text("python code script")
    (code / "util.xxx").write_text("python function code")

    incoming = tmp_path / "incoming"
    nested = incoming / "nested"
    incoming.mkdir()
    nested.mkdir()
    (incoming / "doc1.unk").write_text("document content")
    (nested / "pic1.unk").write_text("picture photo")
    (nested / "prog1.unk").write_text("python code")

    clf = learn_structure(root)
    mapping = classify_files(clf, incoming)

    assert mapping["doc1.unk"] == "docs/text"
    assert mapping["nested/pic1.unk"] == "media/images"
    assert mapping["nested/prog1.unk"] == "code/scripts"


def test_save_and_load(tmp_path: Path) -> None:
    root = tmp_path / "root"
    folder = root / "docs" / "text"
    folder.mkdir(parents=True)
    (folder / "a.dat").write_text("sample document")
    (folder / "b.dat").write_text("another document")

    model = learn_structure(root)
    model_file = tmp_path / "model.pkl"
    model.save(model_file)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "doc.txt").write_text("sample document")

    loaded = NaiveBayesFileClassifier.load(model_file)
    mapping = classify_files(loaded, incoming)

    assert mapping["doc.txt"] == "docs/text"
