"""Extraction tests for the supported file kinds."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from tailor_core.context_files.extraction import (
    ExtractionError,
    detect_kind,
    extract_text,
)
from tailor_core.context_files.models import ContextFileKind


def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Build an in-memory PDF with the given page texts."""
    writer = PdfWriter()
    for _text in pages:
        writer.add_blank_page(width=300, height=300)
    # pypdf's writer doesn't expose a simple API to inject text-extractable
    # content without writing a content stream. For tests we instead use a
    # known minimal PDF text format if needed; here we just verify that
    # PdfReader can open the writer's output (extracted text may be empty).
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# -- detect_kind ------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("doc.pdf", ContextFileKind.PDF),
        ("DOC.PDF", ContextFileKind.PDF),
        ("data.csv", ContextFileKind.CSV),
        ("notes.txt", ContextFileKind.TEXT),
        ("plan.md", ContextFileKind.MARKDOWN),
        ("plan.markdown", ContextFileKind.MARKDOWN),
    ],
)
def test_detect_kind(filename: str, expected: ContextFileKind) -> None:
    assert detect_kind(filename) is expected


def test_detect_kind_rejects_unsupported_extension() -> None:
    with pytest.raises(ExtractionError, match="unsupported"):
        detect_kind("photo.jpg")


def test_detect_kind_rejects_extensionless_filename() -> None:
    with pytest.raises(ExtractionError, match="unsupported"):
        detect_kind("README")


# -- extract_text -----------------------------------------------------------


def test_extract_text_for_text_file() -> None:
    kind, text = extract_text("notes.txt", b"hello\nworld\n")
    assert kind is ContextFileKind.TEXT
    assert text == "hello\nworld\n"


def test_extract_text_for_markdown_file() -> None:
    kind, text = extract_text("plan.md", b"# Plan\n\n- item 1\n")
    assert kind is ContextFileKind.MARKDOWN
    assert text.startswith("# Plan")


def test_extract_text_for_csv_file() -> None:
    kind, text = extract_text("data.csv", b"name,score\nAlex,42\n")
    assert kind is ContextFileKind.CSV
    assert "Alex,42" in text


def test_extract_text_rejects_invalid_utf8() -> None:
    with pytest.raises(ExtractionError, match="UTF-8"):
        extract_text("notes.txt", b"\xff\xfe\x00\x00not-valid-utf8")


def test_extract_text_rejects_unsupported_extension() -> None:
    with pytest.raises(ExtractionError, match="unsupported"):
        extract_text("photo.jpg", b"\x00\x01\x02")


def test_extract_text_for_pdf_returns_pdf_kind() -> None:
    """We can't easily round-trip extractable text through pypdf without
    pulling in another dep, but we can at least verify the PDF kind is
    detected and the extractor doesn't crash on a valid empty PDF."""
    pdf = _make_pdf_bytes([""])

    kind, text = extract_text("notes.pdf", pdf)

    assert kind is ContextFileKind.PDF
    # Blank pages may yield empty text; that's fine.
    assert isinstance(text, str)


def test_extract_text_raises_on_corrupt_pdf() -> None:
    with pytest.raises(ExtractionError, match="PDF could not be read"):
        extract_text("doc.pdf", b"not really a pdf")


def test_extract_text_raises_on_per_page_extraction_failure(
    mocker: object,
) -> None:
    """If pypdf throws on a single page's ``extract_text``, surface it as
    ``ExtractionError`` with the underlying message."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    fake_page = MagicMock()
    fake_page.extract_text.side_effect = ValueError("bad content stream")
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]
    mocker.patch("pypdf.PdfReader", return_value=fake_reader)  # type: ignore[attr-defined]

    with pytest.raises(ExtractionError, match="page text extraction failed"):
        extract_text("doc.pdf", b"%PDF-fake")
