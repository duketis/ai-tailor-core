"""BaseOrchestrator -- template-method pipeline skeleton.

Drives the shared phases for every tailoring run:

1. ``FETCHING_JD`` -- fetch the JD URL (or pass through pasted text).
2. ``PARSING_JD`` -- extract a :class:`JobRequirements` via the JD parser.
3. ``LOADING_CONTEXT`` -- load the user-context tree + uploaded files.
4. ``TAILORING`` -- consumer hook :meth:`_tailor` returns the structured
   tailored output.
5. ``RENDERING`` -- consumer hook :meth:`_render` returns a
   :class:`RenderResult` pointing at the on-disk PDF.
6. ``VERIFYING`` -- consumer hook :meth:`_verify` runs the QC pass;
   :meth:`_verify_visually` runs the vision pass best-effort.

Subclasses implement the four hooks and bind ``TailoredT`` to their
concrete output model. Everything else (event publishing, run lifecycle
mutation, error handling, run-id generation) is shared.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
from pydantic import BaseModel

from tailor_core.context.loader import load_user_context
from tailor_core.jd.fetcher import fetch_jd
from tailor_core.jd.parser import parse_jd_text
from tailor_core.runs.events import RunEventBus
from tailor_core.runs.models import RenderResult, Run, RunEvent, RunStatus, TailorRequest
from tailor_core.runs.store import RunsStore, update_run
from tailor_core.verifier.scaffold import VerifierError, fallback_concerns_result

if TYPE_CHECKING:
    from tailor_core.context.models import UserContext
    from tailor_core.context_files.models import ContextFile
    from tailor_core.context_files.store import ContextFileStore
    from tailor_core.jd.models import JobRequirements
    from tailor_core.llm.client import LLMClient
    from tailor_core.settings.models import BaseRuntimeSettings
    from tailor_core.settings.store import SettingsStore
    from tailor_core.verifier.models import VerificationResult


DEFAULT_CONTEXT_ROOT = Path("UserContext")
DEFAULT_RUNS_ROOT = Path("runs")


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator can't even start the pipeline."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def generate_run_id() -> str:
    """Short, URL-safe, sortable-ish run identifier."""
    stamp = _utcnow().strftime("%Y%m%d%H%M%S")
    return f"run_{stamp}_{secrets.token_urlsafe(6)}"


class BaseOrchestrator[TailoredT: BaseModel, SettingsT: BaseRuntimeSettings]:
    """Pipeline driver. Subclass and implement the four abstract hooks.

    Parameterised on the consumer's tailored output model (``TailoredT``)
    and runtime-settings subclass (``SettingsT``) so the stored references
    keep their concrete types.

    A single instance is held on ``app.state.app_state`` and serves every
    concurrent run. The orchestrator:

    - Persists run records via the supplied ``RunsStore`` so reconnecting
      clients see history.
    - Publishes ``RunEvent`` to the in-memory ``RunEventBus`` so SSE
      subscribers see live progress.
    - Runs the pipeline's blocking steps (HTTP fetch, ``claude`` subprocess,
      tectonic compile, vision API call) on a worker thread via
      ``asyncio.to_thread`` so the event loop stays responsive.
    """

    def __init__(
        self,
        *,
        runs_store: RunsStore[TailoredT],
        settings_store: SettingsStore[SettingsT],
        llm_client: LLMClient,
        event_bus: RunEventBus | None = None,
        context_root: Path = DEFAULT_CONTEXT_ROOT,
        http_client: httpx.Client | None = None,
        context_file_store: ContextFileStore | None = None,
        runs_root: Path = DEFAULT_RUNS_ROOT,
    ) -> None:
        self._runs = runs_store
        self._settings = settings_store
        self._llm = llm_client
        self._event_bus = event_bus or RunEventBus()
        self._context_root = context_root
        self._http = http_client
        self._context_files = context_file_store
        self._runs_root = runs_root

    @property
    def event_bus(self) -> RunEventBus:
        return self._event_bus

    @property
    def llm(self) -> LLMClient:
        return self._llm

    def create_run(self, request: TailorRequest) -> Run[TailoredT]:
        """Persist a new pending run and return it."""
        now = _utcnow()
        run: Run[TailoredT] = Run(
            id=generate_run_id(),
            request=request,
            status=RunStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        self._runs.save(run)
        return run

    async def execute(self, run_id: str) -> Run[TailoredT]:
        """Run the pipeline for ``run_id``. Idempotent end-state on failure.

        This is awaited by the API's BackgroundTasks shim so tests can await
        the same coroutine and assert the resulting state.
        """
        try:
            return await self._execute_inner(run_id)
        except Exception as exc:  # noqa: BLE001 -- funnel everything to FAILED
            return await self._mark_failed(run_id, exc)

    # -- abstract hooks ----------------------------------------------------

    def _tailor(
        self,
        requirements: JobRequirements,
        context: UserContext,
        request: TailorRequest,
        context_files: tuple[ContextFile, ...],
    ) -> TailoredT:
        """Run the consumer's tailoring agent. Subclasses must implement."""
        raise NotImplementedError

    def _render(
        self,
        tailored: TailoredT,
        requirements: JobRequirements,
        output_dir: Path,
    ) -> RenderResult:
        """Render the tailored output to a PDF in ``output_dir``.

        Subclasses must implement. Returns a :class:`RenderResult` whose
        ``doc_url`` points at the rendered PDF on disk.
        """
        raise NotImplementedError

    def _verify(
        self,
        requirements: JobRequirements,
        tailored: TailoredT,
        pdf_path: Path,
    ) -> VerificationResult:
        """Run the consumer's text-mode verifier. Subclasses must implement.

        May raise; the base wraps the call in :meth:`_verify_safely` so a
        :class:`VerifierError` (or any other ``RuntimeError`` / ``OSError``)
        is funnelled into a non-blocking ``CONCERNS`` result rather than
        failing the run.
        """
        raise NotImplementedError

    def _verify_visually(self, pdf_path: Path) -> VerificationResult | None:
        """Run the consumer's vision verifier. Best-effort -- return
        ``None`` when the OAuth token / SDK / API isn't available so the
        run still succeeds. Default impl returns ``None`` (consumers that
        don't care about visual QC can leave it alone).
        """
        return None

    # -- pipeline ---------------------------------------------------------

    async def _execute_inner(self, run_id: str) -> Run[TailoredT]:
        current = self._runs.get(run_id)
        if current is None:
            raise OrchestratorError(f"unknown run {run_id!r}")

        request = current.request

        # Step 1: JD text (fetch URL or use supplied text).
        if request.jd_url:
            await self._step(run_id, RunStatus.FETCHING_JD, f"fetching {request.jd_url}")
            fetched = await asyncio.to_thread(fetch_jd, request.jd_url, http_client=self._http)
            jd_text = fetched.cleaned_text
            source_url = fetched.source_url
        else:
            # TailorRequest's validator guarantees jd_text is set when
            # jd_url is not, so the cast is safe.
            jd_text = cast("str", request.jd_text)
            source_url = None

        # Step 2: parse JD into JobRequirements.
        await self._step(run_id, RunStatus.PARSING_JD, "extracting structured requirements")
        requirements = await asyncio.to_thread(
            parse_jd_text, jd_text, llm=self._llm, source_url=source_url
        )
        update_run(self._runs, run_id, requirements=requirements)

        # Step 3: load the user's context tree + uploaded files.
        await self._step(run_id, RunStatus.LOADING_CONTEXT, "loading user context")
        context = await asyncio.to_thread(load_user_context, self._context_root)
        context_files_list = list(self._context_files.list_all()) if self._context_files else []

        # Step 4: tailor (consumer hook).
        await self._step(run_id, RunStatus.TAILORING, "running tailoring agent")
        tailored = await asyncio.to_thread(
            self._tailor,
            requirements,
            context,
            request,
            tuple(context_files_list),
        )
        update_run(self._runs, run_id, tailored=tailored)

        # Step 5: render (consumer hook).
        await self._step(run_id, RunStatus.RENDERING, "rendering tailored PDF")
        output_dir = self._runs_root / run_id
        result = await asyncio.to_thread(self._render, tailored, requirements, output_dir)
        update_run(self._runs, run_id, result=result)

        # Step 6: verify (consumer hook). QC pass never blocks the run --
        # surfaces findings on the run detail page so the user can review.
        # PDF path passes through so the verifier can run page-count checks.
        await self._step(run_id, RunStatus.VERIFYING, "running QC verification")
        rendered_pdf_path = _pdf_path_from_result(result, output_dir)
        verification = await asyncio.to_thread(
            self._verify_safely, requirements, tailored, rendered_pdf_path
        )
        update_run(self._runs, run_id, verification=verification)

        # Step 6b: vision QC (consumer hook). Best-effort -- returns None
        # when the token / SDK / API isn't available. The run still
        # succeeds without it; only the run-detail page's "Visual
        # verification" card is empty.
        vision_verification = await asyncio.to_thread(self._verify_visually, rendered_pdf_path)
        if vision_verification is not None:
            update_run(self._runs, run_id, vision_verification=vision_verification)

        # Done.
        finished = update_run(
            self._runs,
            run_id,
            status=RunStatus.SUCCEEDED,
            detail="render + verification complete",
        )
        await self._publish(run_id, RunStatus.SUCCEEDED, "render + verification complete")
        await self._event_bus.close(run_id)
        return finished

    def _verify_safely(
        self,
        requirements: JobRequirements,
        tailored: TailoredT,
        pdf_path: Path,
    ) -> VerificationResult:
        """Run :meth:`_verify` but never let it block the run -- fall back
        to a synthetic ``CONCERNS`` result on any failure."""
        try:
            return self._verify(requirements, tailored, pdf_path)
        except (VerifierError, OSError, RuntimeError) as exc:
            return fallback_concerns_result(f"{type(exc).__name__}: {exc}")

    async def _mark_failed(self, run_id: str, exc: Exception) -> Run[TailoredT]:
        """Persist FAILED state if the run exists, then close the event bus.

        If the run was never created (eg ``execute`` called with a bad id),
        synthesise a one-off FAILED Run so callers always get a model back.
        """
        message = f"{type(exc).__name__}: {exc}"
        try:
            failed = update_run(
                self._runs,
                run_id,
                status=RunStatus.FAILED,
                detail="pipeline failed",
                error=message,
            )
        except KeyError:
            now = _utcnow()
            failed = Run(
                id=run_id or "unknown",
                request=TailorRequest(jd_text="(unknown — run was never created)"),
                status=RunStatus.FAILED,
                created_at=now,
                updated_at=now,
                detail="pipeline failed",
                error=message,
            )
        await self._publish(failed.id, RunStatus.FAILED, str(exc))
        await self._event_bus.close(run_id)
        return failed

    async def _step(self, run_id: str, status: RunStatus, detail: str) -> None:
        update_run(self._runs, run_id, status=status, detail=detail)
        await self._publish(run_id, status, detail)

    async def _publish(self, run_id: str, status: RunStatus, detail: str) -> None:
        await self._event_bus.publish(
            RunEvent(run_id=run_id, status=status, detail=detail, at=_utcnow())
        )


def _pdf_path_from_result(result: RenderResult, output_dir: Path) -> Path:
    """Derive the on-disk PDF path from a :class:`RenderResult`.

    ``doc_url`` is a ``file://`` URL pointing at the rendered PDF; fall
    back to ``<output_dir>/<doc_id>.pdf`` when the URL isn't parseable
    (defensive -- consumer renderers should always emit a file URL).
    """
    if result.doc_url.startswith("file://"):
        return Path(result.doc_url.removeprefix("file://"))
    return output_dir / f"{result.doc_id}.pdf"
