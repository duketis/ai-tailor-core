"""BaseOrchestrator template-method tests.

The orchestrator skeleton is exercised via a tiny subclass that implements
``_tailor`` / ``_render`` / ``_verify`` with deterministic stubs. The
shared JD-fetch, JD-parse, context-load steps are mocked at the
orchestrator module level so the tests don't touch the network, the
``claude`` CLI, or tectonic.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tailor_core.context.models import UserContext
from tailor_core.jd.models import (
    EmploymentType,
    FetchedJD,
    JobRequirements,
    RemoteType,
    RoleType,
    Seniority,
)
from tailor_core.llm.client import FakeLLMClient
from tailor_core.runs import orchestrator as orch_mod
from tailor_core.runs.models import (
    RenderResult,
    RunStatus,
    TailorRequest,
)
from tailor_core.runs.orchestrator import (
    BaseOrchestrator,
    OrchestratorError,
    generate_run_id,
)
from tailor_core.runs.store import InMemoryRunsStore
from tailor_core.settings.models import BaseRuntimeSettings
from tailor_core.settings.store import InMemorySettingsStore
from tailor_core.verifier.models import (
    VerificationResult,
    VerificationStatus,
)
from tests.unit.runs._stub_tailored import StubTailored

if TYPE_CHECKING:
    from tailor_core.context_files.models import ContextFile


# -- fixtures + canned outputs ----------------------------------------------


def _requirements() -> JobRequirements:
    return JobRequirements(
        title="Software Developer",
        company="GoSource",
        location="Canberra",
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.MID,
        employment_type=EmploymentType.FULL_TIME,
        remote_type=RemoteType.REMOTE,
    )


def _render_result(run_id: str = "run") -> RenderResult:
    return RenderResult(
        doc_id=run_id,
        doc_url=f"file:///tmp/{run_id}/doc.pdf",
        pdf_size_bytes=1024,
    )


def _passed_verification() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.PASSED,
        summary="clean",
        rationale="no issues",
    )


class _StubOrchestrator(BaseOrchestrator[StubTailored, BaseRuntimeSettings]):
    """Tiny concrete subclass with deterministic hooks."""

    def __init__(self, *, calls: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._calls = calls

    def _tailor(
        self,
        requirements: JobRequirements,
        context: UserContext,
        request: TailorRequest,
        context_files: tuple[ContextFile, ...],
    ) -> StubTailored:
        self._calls["tailor_request"] = request
        self._calls["tailor_files"] = context_files
        return StubTailored(label="tailored")

    def _render(
        self,
        tailored: StubTailored,
        requirements: JobRequirements,
        output_dir: Path,
    ) -> RenderResult:
        self._calls["render_dir"] = output_dir
        self._calls["render_tailored"] = tailored
        return _render_result(output_dir.name)

    def _verify(
        self,
        requirements: JobRequirements,
        tailored: StubTailored,
        pdf_path: Path,
    ) -> VerificationResult:
        self._calls["verify_pdf"] = pdf_path
        return _passed_verification()


@pytest.fixture
def runs() -> InMemoryRunsStore[StubTailored]:
    return InMemoryRunsStore[StubTailored]()


@pytest.fixture
def settings() -> InMemorySettingsStore[BaseRuntimeSettings]:
    return InMemorySettingsStore(settings_cls=BaseRuntimeSettings)


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(default_response="{}")


@pytest.fixture
def calls() -> dict[str, Any]:
    return {}


@pytest.fixture
def orchestrator(
    runs: InMemoryRunsStore[StubTailored],
    settings: InMemorySettingsStore[BaseRuntimeSettings],
    llm: FakeLLMClient,
    calls: dict[str, Any],
    tmp_path: Path,
) -> _StubOrchestrator:
    return _StubOrchestrator(
        calls=calls,
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the shared JD-fetch / JD-parse / context-load helpers with
    deterministic stubs so the pipeline runs without external services."""

    def fake_fetch_jd(url: str, **_: object) -> FetchedJD:
        return FetchedJD(
            source_url=url,
            raw_html="<html/>",
            cleaned_text="JD body for " + url,
            fetched_at=datetime(2026, 5, 11, tzinfo=UTC),
            ats="unknown",
        )

    def fake_parse_jd_text(_text: str, **_: object) -> JobRequirements:
        return _requirements()

    def fake_load_user_context(_root: Path) -> UserContext:
        return UserContext()

    monkeypatch.setattr(orch_mod, "fetch_jd", fake_fetch_jd)
    monkeypatch.setattr(orch_mod, "parse_jd_text", fake_parse_jd_text)
    monkeypatch.setattr(orch_mod, "load_user_context", fake_load_user_context)


# -- run id ----------------------------------------------------------------


def test_generate_run_id_has_expected_shape() -> None:
    run_id = generate_run_id()
    assert run_id.startswith("run_")
    # ``run_YYYYMMDDhhmmss_<token>`` -- timestamp piece is 14 digits.
    assert run_id[4:18].isdigit()


# -- create_run -----------------------------------------------------------


def test_create_run_persists_a_pending_record(
    orchestrator: _StubOrchestrator,
    runs: InMemoryRunsStore[StubTailored],
) -> None:
    run = orchestrator.create_run(TailorRequest(jd_text="paste"))
    assert run.status is RunStatus.PENDING
    assert runs.get(run.id) == run


# -- execute() happy paths -------------------------------------------------


def test_execute_drives_pipeline_to_succeeded_with_jd_url(
    orchestrator: _StubOrchestrator,
    runs: InMemoryRunsStore[StubTailored],
    calls: dict[str, Any],
    patched_pipeline: None,
) -> None:
    run = orchestrator.create_run(TailorRequest(jd_url="https://example.com/job/1"))
    finished = asyncio.run(orchestrator.execute(run.id))

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.detail == "render + verification complete"
    assert finished.requirements is not None
    assert finished.tailored == StubTailored(label="tailored")
    assert finished.result is not None
    assert finished.verification is not None
    assert finished.error is None
    # The render step received the per-run output dir.
    assert calls["render_dir"].name == run.id
    # Saved into the same record the test holds.
    assert runs.get(run.id) == finished


def test_execute_with_jd_text_skips_the_fetch_step(
    orchestrator: _StubOrchestrator,
    runs: InMemoryRunsStore[StubTailored],
    monkeypatch: pytest.MonkeyPatch,
    patched_pipeline: None,
) -> None:
    fetch_calls: list[str] = []

    def tracking_fetch(url: str, **_: object) -> FetchedJD:
        fetch_calls.append(url)
        raise AssertionError("fetch should not be called when jd_text supplied")

    monkeypatch.setattr(orch_mod, "fetch_jd", tracking_fetch)

    run = orchestrator.create_run(TailorRequest(jd_text="paste"))
    finished = asyncio.run(orchestrator.execute(run.id))

    assert finished.status is RunStatus.SUCCEEDED
    assert fetch_calls == []


def test_execute_propagates_model_override_into_tailor(
    orchestrator: _StubOrchestrator,
    calls: dict[str, Any],
    patched_pipeline: None,
) -> None:
    run = orchestrator.create_run(TailorRequest(jd_text="paste", model="claude-sonnet-4-6"))
    asyncio.run(orchestrator.execute(run.id))
    assert calls["tailor_request"].model == "claude-sonnet-4-6"


# -- execute() failure paths -----------------------------------------------


def test_execute_marks_run_failed_when_a_phase_raises(
    runs: InMemoryRunsStore[StubTailored],
    settings: InMemorySettingsStore[BaseRuntimeSettings],
    llm: FakeLLMClient,
    tmp_path: Path,
    patched_pipeline: None,
) -> None:
    class ExplodingOrchestrator(BaseOrchestrator[StubTailored, BaseRuntimeSettings]):
        def _tailor(self, *args: object, **kwargs: object) -> StubTailored:
            raise RuntimeError("tailor blew up")

        def _render(self, *args: object, **kwargs: object) -> RenderResult:
            raise AssertionError("should not reach render")

        def _verify(self, *args: object, **kwargs: object) -> VerificationResult:
            raise AssertionError("should not reach verify")

    orch = ExplodingOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )

    run = orch.create_run(TailorRequest(jd_text="paste"))
    failed = asyncio.run(orch.execute(run.id))

    assert failed.status is RunStatus.FAILED
    assert failed.error is not None
    assert "tailor blew up" in failed.error


def test_execute_raises_for_unknown_run_id_via_mark_failed(
    orchestrator: _StubOrchestrator,
) -> None:
    failed = asyncio.run(orchestrator.execute("never-created"))
    # No KeyError leaks; ``_mark_failed`` synthesises a one-off FAILED Run.
    assert failed.status is RunStatus.FAILED
    assert failed.id == "never-created"


def test_orchestrator_error_when_create_run_skipped(
    orchestrator: _StubOrchestrator,
) -> None:
    """Direct execute on a never-created id surfaces a synthetic FAILED;
    the underlying error type is OrchestratorError."""
    finished = asyncio.run(orchestrator.execute("missing"))
    assert finished.status is RunStatus.FAILED
    assert "OrchestratorError" in (finished.error or "")


# -- verify_safely fallback -------------------------------------------------


def test_verify_safely_returns_fallback_concerns_on_verifier_error(
    runs: InMemoryRunsStore[StubTailored],
    settings: InMemorySettingsStore[BaseRuntimeSettings],
    llm: FakeLLMClient,
    tmp_path: Path,
    patched_pipeline: None,
) -> None:
    class FlakyVerifierOrchestrator(_StubOrchestrator):
        def _verify(self, *args: object, **kwargs: object) -> VerificationResult:
            raise RuntimeError("verifier flake")

    orch = FlakyVerifierOrchestrator(
        calls={},
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )

    run = orch.create_run(TailorRequest(jd_text="paste"))
    finished = asyncio.run(orch.execute(run.id))

    # Pipeline still succeeded -- verifier failure surfaces as CONCERNS.
    assert finished.status is RunStatus.SUCCEEDED
    assert finished.verification is not None
    assert finished.verification.status is VerificationStatus.CONCERNS
    assert "RuntimeError" in finished.verification.issues[0].message


# -- vision verification --------------------------------------------------


def test_vision_verification_default_hook_returns_none(
    orchestrator: _StubOrchestrator,
    patched_pipeline: None,
) -> None:
    """The default ``_verify_visually`` is a no-op; runs succeed with
    ``vision_verification == None`` unless the subclass overrides."""
    run = orchestrator.create_run(TailorRequest(jd_text="paste"))
    finished = asyncio.run(orchestrator.execute(run.id))
    assert finished.vision_verification is None


def test_vision_verification_hook_persists_when_overridden(
    runs: InMemoryRunsStore[StubTailored],
    settings: InMemorySettingsStore[BaseRuntimeSettings],
    llm: FakeLLMClient,
    tmp_path: Path,
    patched_pipeline: None,
) -> None:
    class VisionOrchestrator(_StubOrchestrator):
        def _verify_visually(self, pdf_path: Path) -> VerificationResult | None:
            return VerificationResult(
                status=VerificationStatus.CONCERNS,
                summary="widow line",
                rationale="page 3 starts with three words.",
            )

    orch = VisionOrchestrator(
        calls={},
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )

    run = orch.create_run(TailorRequest(jd_text="paste"))
    finished = asyncio.run(orch.execute(run.id))
    assert finished.vision_verification is not None
    assert finished.vision_verification.summary == "widow line"


# -- _execute_inner direct error path -------------------------------------


def test_execute_inner_raises_orchestrator_error_for_unknown_id(
    orchestrator: _StubOrchestrator,
) -> None:
    with pytest.raises(OrchestratorError, match="unknown run"):
        asyncio.run(orchestrator._execute_inner("missing"))


def test_pdf_path_from_result_url_decodes_spaces() -> None:
    """Regression: ``Path.as_uri()`` percent-encodes spaces; the inverse
    decode must run when we turn the file URL back into a Path or
    pypdfium2 / pypdf can't find the file."""
    from tailor_core.runs.orchestrator import _pdf_path_from_result  # noqa: PLC0415

    result = RenderResult(
        doc_id="run_x",
        doc_url="file:///app/runs/run_x/Cover%20Letter%20-%20Acme%20-%20Engineer.pdf",
    )
    pdf_path = _pdf_path_from_result(result, Path("/app/runs/run_x"))
    assert pdf_path == Path("/app/runs/run_x/Cover Letter - Acme - Engineer.pdf")


def test_pdf_path_from_result_falls_back_to_doc_id_for_non_file_urls(tmp_path: Path) -> None:
    """Defensive: if a renderer ever emits a non-file URL (eg http://),
    fall back to ``<output_dir>/<doc_id>.pdf`` rather than trying to
    fetch it."""
    from tailor_core.runs.orchestrator import _pdf_path_from_result  # noqa: PLC0415

    result = RenderResult(doc_id="run_x", doc_url="https://example.com/x.pdf")
    pdf_path = _pdf_path_from_result(result, tmp_path)
    assert pdf_path == tmp_path / "run_x.pdf"


def test_event_bus_property_exposes_supplied_bus(
    runs: InMemoryRunsStore[StubTailored],
    settings: InMemorySettingsStore[BaseRuntimeSettings],
    llm: FakeLLMClient,
    tmp_path: Path,
) -> None:
    """The ``event_bus`` property exposes the bus the SSE route subscribes
    to. Pinning the accessor so we don't accidentally make a fresh one."""
    from tailor_core.runs.events import RunEventBus  # noqa: PLC0415

    bus = RunEventBus()
    orch = _StubOrchestrator(
        calls={},
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        event_bus=bus,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )
    assert orch.event_bus is bus


def test_llm_property_exposes_supplied_client(
    orchestrator: _StubOrchestrator, llm: FakeLLMClient
) -> None:
    """``llm`` property: subclasses (like coverletterai) reach in for the
    LLM client when forwarding to their agent."""
    assert orchestrator.llm is llm
