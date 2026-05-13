"""Vision verifier tests -- PDF rasterisation + Anthropic vision API call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tailor_core.verifier.models import IssueSeverity, VerificationStatus
from tailor_core.verifier.vision import (
    VisionVerifierError,
    rasterise_pdf_to_pngs,
    run_vision_verification,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# -- fixtures --------------------------------------------------------------


def _write_pdf(path: Path, *, pages: int = 1) -> None:
    from pypdf import PdfWriter  # noqa: PLC0415

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as fh:
        writer.write(fh)


_PASSED_VISION_RESPONSE = json.dumps(
    {
        "status": "passed",
        "summary": "clean layout",
        "issues": [],
        "rationale": "everything looks fine.",
    }
)

_CONCERNS_VISION_RESPONSE = json.dumps(
    {
        "status": "concerns",
        "summary": "one widow line on page 3",
        "issues": [
            {
                "severity": "warn",
                "category": "orphan_line",
                "message": "Page 3 starts with 3 words finishing a bullet from page 2.",
                "suggestion": "Shorten the bullet on page 2 to fit on one page.",
            }
        ],
        "rationale": "small layout issue.",
    }
)


_SYS_PROMPT = "Review the rendered pages for layout issues. Return verification JSON."
_USER_PROMPT = "Above are the rendered pages of a document; report issues."


# -- rasterise_pdf_to_pngs -------------------------------------------------


def test_rasterise_pdf_to_pngs_returns_one_png_per_page(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, pages=2)
    pages = rasterise_pdf_to_pngs(pdf)
    assert len(pages) == 2
    # PNG magic bytes
    assert pages[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert pages[1][:8] == b"\x89PNG\r\n\x1a\n"


def test_rasterise_pdf_to_pngs_raises_on_unreadable_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"definitely not a pdf")
    with pytest.raises(VisionVerifierError, match="could not open PDF"):
        rasterise_pdf_to_pngs(pdf)


# -- run_vision_verification skip paths ------------------------------------


def test_run_vision_verification_returns_none_without_oauth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    assert run_vision_verification(pdf, system_prompt=_SYS_PROMPT, user_prompt=_USER_PROMPT) is None


def test_run_vision_verification_returns_none_on_unreadable_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"not a real pdf")
    assert (
        run_vision_verification(
            pdf,
            system_prompt=_SYS_PROMPT,
            user_prompt=_USER_PROMPT,
            oauth_token="sk-ant-oat01-fake",
        )
        is None
    )


def test_run_vision_verification_returns_none_when_api_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "tailor_core.verifier.vision.call_vision_api",
        side_effect=VisionVerifierError("API down"),
    )
    assert (
        run_vision_verification(
            pdf,
            system_prompt=_SYS_PROMPT,
            user_prompt=_USER_PROMPT,
            oauth_token="sk-ant-oat01-fake",
        )
        is None
    )


def test_run_vision_verification_returns_none_when_response_is_malformed(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "tailor_core.verifier.vision.call_vision_api",
        return_value="not valid json garbage",
    )
    assert (
        run_vision_verification(
            pdf,
            system_prompt=_SYS_PROMPT,
            user_prompt=_USER_PROMPT,
            oauth_token="sk-ant-oat01-fake",
        )
        is None
    )


def test_run_vision_verification_returns_none_when_rasterisation_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "tailor_core.verifier.vision.rasterise_pdf_to_pngs",
        side_effect=VisionVerifierError("pdfium broken"),
    )
    assert (
        run_vision_verification(
            pdf,
            system_prompt=_SYS_PROMPT,
            user_prompt=_USER_PROMPT,
            oauth_token="sk-ant-oat01-fake",
        )
        is None
    )


# -- run_vision_verification happy path ------------------------------------


def test_run_vision_verification_returns_parsed_result_when_api_succeeds(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf, pages=2)
    mocker.patch(
        "tailor_core.verifier.vision.call_vision_api",
        return_value=_CONCERNS_VISION_RESPONSE,
    )
    result = run_vision_verification(
        pdf,
        system_prompt=_SYS_PROMPT,
        user_prompt=_USER_PROMPT,
        oauth_token="sk-ant-oat01-fake",
    )
    assert result is not None
    assert result.status is VerificationStatus.CONCERNS
    assert result.issues[0].category == "orphan_line"
    assert result.issues[0].severity is IssueSeverity.WARN


def test_run_vision_verification_uses_env_var_token(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``oauth_token`` arg is absent the env var is read."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-env")
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    api_mock = mocker.patch(
        "tailor_core.verifier.vision.call_vision_api",
        return_value=_PASSED_VISION_RESPONSE,
    )
    run_vision_verification(pdf, system_prompt=_SYS_PROMPT, user_prompt=_USER_PROMPT)
    assert api_mock.call_args.kwargs["oauth_token"] == "sk-ant-oat01-env"
    assert api_mock.call_args.kwargs["system_prompt"] == _SYS_PROMPT
    assert api_mock.call_args.kwargs["user_prompt"] == _USER_PROMPT


def test_run_vision_verification_forwards_passed_response(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "tailor_core.verifier.vision.call_vision_api",
        return_value=_PASSED_VISION_RESPONSE,
    )
    result = run_vision_verification(
        pdf,
        system_prompt=_SYS_PROMPT,
        user_prompt=_USER_PROMPT,
        oauth_token="sk-ant-oat01-fake",
    )
    assert result is not None
    assert result.status is VerificationStatus.PASSED
    assert result.issues == ()


# -- call_vision_api -------------------------------------------------------


def test_call_vision_api_raises_when_anthropic_sdk_missing(mocker: MockerFixture) -> None:
    """If the ``anthropic`` SDK can't be imported (eg slimmed dev env),
    surface a clear VisionVerifierError instead of an ImportError leak."""
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fail_anthropic_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    mocker.patch("builtins.__import__", side_effect=fail_anthropic_import)

    from tailor_core.verifier.vision import call_vision_api  # noqa: PLC0415

    with pytest.raises(VisionVerifierError, match="anthropic SDK not installed"):
        call_vision_api(
            pages=[b"\x89PNG"],
            system_prompt="SYS",
            user_prompt="USER",
            oauth_token="sk-ant-oat01-fake",
        )


def test_call_vision_api_wraps_messages_api_failure(mocker: MockerFixture) -> None:
    """A failure inside ``client.messages.create`` becomes
    VisionVerifierError so the orchestrator's silent-skip logic can
    recognise it."""
    from tailor_core.verifier.vision import call_vision_api  # noqa: PLC0415

    fake_client = mocker.MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("rate limited")
    mocker.patch("anthropic.Anthropic", return_value=fake_client)

    with pytest.raises(VisionVerifierError, match="messages API call failed"):
        call_vision_api(
            pages=[b"\x89PNG"],
            system_prompt="SYS",
            user_prompt="USER",
            oauth_token="sk-ant-oat01-fake",
        )


def test_call_vision_api_returns_concatenated_text_blocks(mocker: MockerFixture) -> None:
    """Happy path: response with multiple TextBlocks concatenates."""
    from anthropic.types import TextBlock  # noqa: PLC0415

    from tailor_core.verifier.vision import call_vision_api  # noqa: PLC0415

    fake_client = mocker.MagicMock()
    # Mix a TextBlock with a non-text block to also exercise the
    # ``isinstance(block, TextBlock)`` narrowing branch.
    non_text = mocker.MagicMock(spec=object)  # not a TextBlock
    fake_client.messages.create.return_value = mocker.MagicMock(
        content=[
            TextBlock(text="line one", citations=None, type="text"),
            non_text,
            TextBlock(text="line two", citations=None, type="text"),
        ]
    )
    mocker.patch("anthropic.Anthropic", return_value=fake_client)

    result = call_vision_api(
        pages=[b"\x89PNG"],
        system_prompt="SYS",
        user_prompt="USER",
        oauth_token="sk-ant-oat01-fake",
    )
    assert result == "line one\nline two"


def test_call_vision_api_raises_when_response_has_no_text_blocks(mocker: MockerFixture) -> None:
    """A response with only non-text blocks (thinking, tool-use, ...) is a
    parse error -- the caller can't make a VerificationResult out of it."""
    from tailor_core.verifier.vision import call_vision_api  # noqa: PLC0415

    fake_client = mocker.MagicMock()
    non_text = mocker.MagicMock(spec=object)
    fake_client.messages.create.return_value = mocker.MagicMock(content=[non_text])
    mocker.patch("anthropic.Anthropic", return_value=fake_client)

    with pytest.raises(VisionVerifierError, match="no text blocks"):
        call_vision_api(
            pages=[b"\x89PNG"],
            system_prompt="SYS",
            user_prompt="USER",
            oauth_token="sk-ant-oat01-fake",
        )


def test_run_vision_verification_returns_none_when_pdf_has_zero_pages(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """A successfully-opened but page-less PDF degrades silently."""
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "tailor_core.verifier.vision.rasterise_pdf_to_pngs",
        return_value=[],
    )
    assert (
        run_vision_verification(
            pdf,
            system_prompt=_SYS_PROMPT,
            user_prompt=_USER_PROMPT,
            oauth_token="sk-ant-oat01-fake",
        )
        is None
    )


def test_rasterise_pdf_to_pngs_wraps_per_page_render_failure(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """A PDF that opens but whose pages can't be rendered (eg pypdfium2 +
    Pillow internals throw) surfaces as VisionVerifierError so the
    orchestrator's silent-skip path can catch it cleanly."""
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)

    class _FakeDoc:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> object:
            class _FakePage:
                def render(self, *_: object, **__: object) -> object:
                    raise RuntimeError("pillow blew up")

            return _FakePage()

        def close(self) -> None:
            pass

    mocker.patch("pypdfium2.PdfDocument", return_value=_FakeDoc())
    with pytest.raises(VisionVerifierError, match="rasterising page 1 failed"):
        rasterise_pdf_to_pngs(pdf)
