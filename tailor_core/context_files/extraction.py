"""Turn an uploaded file's bytes into searchable plain text.

Supported kinds:

- ``.pdf`` → ``pypdf`` text extraction
- ``.csv`` → keep verbatim (the LLM handles structured CSVs better than
  any auto-summary)
- ``.txt`` / ``.md`` / ``.markdown`` → decode as UTF-8

Anything else returns ``ExtractionError``. The caller (the upload route)
surfaces the message in a flash banner.
"""

from __future__ import annotations

import io

from tailor_core.context_files.models import ContextFileKind


class ExtractionError(RuntimeError):
    """Raised when a supplied filename / bytes pair can't be extracted."""


_KIND_BY_EXTENSION: dict[str, ContextFileKind] = {
    "pdf": ContextFileKind.PDF,
    "csv": ContextFileKind.CSV,
    "txt": ContextFileKind.TEXT,
    "md": ContextFileKind.MARKDOWN,
    "markdown": ContextFileKind.MARKDOWN,
}


def detect_kind(filename: str) -> ContextFileKind:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _KIND_BY_EXTENSION:
        raise ExtractionError(
            f"unsupported file extension {ext!r} (filename {filename!r}) — "
            "supported: pdf, csv, txt, md"
        )
    return _KIND_BY_EXTENSION[ext]


def extract_text(filename: str, data: bytes) -> tuple[ContextFileKind, str]:
    """Return the kind + extracted text for an uploaded file."""
    kind = detect_kind(filename)
    if kind is ContextFileKind.PDF:
        return kind, _extract_pdf(data)
    return kind, _decode_text(data)


def _extract_pdf(data: bytes) -> str:
    # Deferred import — pypdf is heavy and tests that don't touch PDFs
    # shouldn't pay its import cost.
    from pypdf import PdfReader  # noqa: PLC0415
    from pypdf.errors import PdfReadError  # noqa: PLC0415

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ExtractionError(f"PDF could not be read: {exc}") from exc
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except (ValueError, KeyError, AttributeError) as exc:
            raise ExtractionError(f"PDF page text extraction failed: {exc}") from exc
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(
            f"file is not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
