"""Vision QC scaffolding -- rasterise a rendered PDF + dispatch to a vision LLM.

The text verifier in :mod:`tailor_core.verifier.scaffold` can't see layout --
widow lines, awkward gaps, section-header orphaning, content overflowing the
right margin, density issues. This module rasterises each page of a rendered
PDF to PNG via pypdfium2 and ships the images to the Anthropic messages API
for a vision review.

Why a separate auth path?
The text-mode pipeline shells out to the ``claude`` CLI. The CLI doesn't
accept inline image bytes (``--file`` takes a pre-uploaded Files-API ID).
The Anthropic Python SDK does -- and it accepts the same Max-plan OAuth
token via ``CLAUDE_CODE_OAUTH_TOKEN``, so vision works without an API key.

Failures degrade gracefully: a missing token, network error, SDK exception,
or malformed model response all produce ``None`` rather than blocking the
run. The consumer's run still surfaces the text-mode verifier's findings;
visual ones are best-effort.

This module is generic; consumers (resumeai, coverletterai) supply the
SYSTEM_PROMPT describing what to look for and the user prompt accompanying
the rendered pages.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import TYPE_CHECKING

from anthropic.types import (
    ImageBlockParam,
    TextBlock,
    TextBlockParam,
)

from tailor_core.verifier.scaffold import VerifierError, parse_verifier_response

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from tailor_core.verifier.models import VerificationResult

_log = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "claude-opus-4-7"
DEFAULT_VISION_MAX_TOKENS = 2000
# PDF pages render to PNG at this DPI; 150 is the sweet spot between
# small payloads and "Claude can actually read the text on the page".
DEFAULT_VISION_DPI = 150


class VisionVerifierError(RuntimeError):
    """Raised when the vision verifier can't complete its check."""


def run_vision_verification(
    pdf_path: Path,
    *,
    system_prompt: str,
    user_prompt: str,
    oauth_token: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = DEFAULT_VISION_DPI,
    max_tokens: int = DEFAULT_VISION_MAX_TOKENS,
) -> VerificationResult | None:
    """Rasterise ``pdf_path``, send to the vision API, return the parsed result.

    Returns ``None`` when the vision pass can't run -- missing OAuth
    token, missing SDK, network error, malformed response. Callers
    should treat ``None`` as "no visual signal" and proceed with the
    text-only verification result. Never raises -- all failure modes
    log a warning and degrade silently.
    """
    token = oauth_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        _log.info("vision verifier skipped: CLAUDE_CODE_OAUTH_TOKEN not set")
        return None

    try:
        pages = rasterise_pdf_to_pngs(pdf_path, dpi=dpi)
    except (OSError, VisionVerifierError) as exc:
        _log.warning("vision verifier skipped: PDF rasterisation failed: %s", exc)
        return None

    if not pages:
        _log.warning("vision verifier skipped: PDF has zero pages")
        return None

    try:
        raw = call_vision_api(
            pages=pages,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            oauth_token=token,
            model=model,
            max_tokens=max_tokens,
        )
    except VisionVerifierError as exc:
        _log.warning("vision verifier skipped: API call failed: %s", exc)
        return None

    try:
        return parse_verifier_response(raw)
    except VerifierError as exc:
        _log.warning("vision verifier skipped: response parse failed: %s", exc)
        return None


def rasterise_pdf_to_pngs(pdf_path: Path, *, dpi: int = DEFAULT_VISION_DPI) -> list[bytes]:
    """Use pypdfium2 to rasterise every page of ``pdf_path`` to PNG bytes."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        raise VisionVerifierError(f"could not open PDF: {exc}") from exc

    pages: list[bytes] = []
    # pypdfium2 uses 72 DPI as native; render scale = dpi / 72.
    scale = dpi / 72.0
    try:
        for page_idx in range(len(document)):
            page = document[page_idx]
            try:
                image = page.render(scale=scale).to_pil()
            except Exception as exc:
                raise VisionVerifierError(f"rasterising page {page_idx + 1} failed: {exc}") from exc
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            pages.append(buffer.getvalue())
    finally:
        document.close()
    return pages


def call_vision_api(
    *,
    pages: Iterable[bytes],
    system_prompt: str,
    user_prompt: str,
    oauth_token: str,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = DEFAULT_VISION_MAX_TOKENS,
) -> str:
    """Send the rendered pages + user prompt to the Anthropic messages API.

    Returns the raw text from the response. Caller is responsible for
    parsing -- typically via
    :func:`tailor_core.verifier.scaffold.parse_verifier_response`.
    """
    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise VisionVerifierError("anthropic SDK not installed") from exc

    # The SDK accepts an OAuth token via ``auth_token``. The Max-plan
    # OAuth token from ``claude setup-token`` works with this path.
    client = Anthropic(auth_token=oauth_token)

    content: list[ImageBlockParam | TextBlockParam] = []
    for png_bytes in pages:
        content.append(
            ImageBlockParam(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(png_bytes).decode("ascii"),
                },
            )
        )
    content.append(TextBlockParam(type="text", text=user_prompt))

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        raise VisionVerifierError(f"messages API call failed: {exc}") from exc

    text_chunks: list[str] = []
    for block in response.content:
        # Narrow with isinstance -- the anthropic SDK's response content is a
        # union of many block types (text, thinking, tool-use, etc.) and a
        # ``getattr(...) == "text"`` check doesn't narrow the type for mypy.
        if isinstance(block, TextBlock):
            text_chunks.append(block.text)
    if not text_chunks:
        raise VisionVerifierError("response had no text blocks")
    return "\n".join(text_chunks)
