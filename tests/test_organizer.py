from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fileorganiser.organizer import learn_structure, classify_files
from fileorganiser.classifier import NaiveBayesFileClassifier


def write_docx(path: Path, text: str) -> None:
    content_types = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        "<Default Extension='xml' ContentType='application/xml'/>"
        "<Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
        "</Types>"
    )
    rels = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        "<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>"
        "</Relationships>"
    )
    document = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
        "<w:body><w:p><w:r><w:t>%s</w:t></w:r></w:p></w:body></w:document>" % text
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


def test_classify_with_contents(tmp_path: Path) -> None:
    root = tmp_path / "root"
    docs = root / "docs" / "letters"
    images = root / "images" / "cats"
    docs.mkdir(parents=True)
    images.mkdir(parents=True)

    write_docx(docs / "a.docx", "hello world")
    (images / "img.txt").write_text("a cute cat")

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    write_docx(incoming / "new.docx", "hello world")
    (incoming / "pic.txt").write_text("cat laying")

    clf = learn_structure(root)
    mapping = classify_files(clf, incoming)

    assert mapping["new.docx"] == "docs/letters"
    assert mapping["pic.txt"] == "images/cats"


def test_save_and_load(tmp_path: Path) -> None:
    root = tmp_path / "root"
    docs = root / "docs"
    docs.mkdir(parents=True)
    write_docx(docs / "a.docx", "save this")

    model = learn_structure(root)
    model_file = tmp_path / "model.pkl"
    model.save(model_file)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    write_docx(incoming / "b.docx", "save this")

    loaded = NaiveBayesFileClassifier.load(model_file)
    mapping = classify_files(loaded, incoming)

    assert mapping["b.docx"] == "docs"
