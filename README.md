# ai-tailor-core

Shared core for my AI tailoring family of apps.

Reads a job description + a candidate's career context, drives a tailoring
agent, renders the structured output to a PDF, and runs a QC pass — but
none of those steps are resume-specific or cover-letter-specific. The
resume-shaped output lives in [resumeai](https://github.com/duketis/resumeai)
and the cover-letter-shaped output lives in
[coverletterai](https://github.com/duketis/coverletterai); both consume
this library so they share one orchestrator skeleton, one storage layer,
one verifier scaffold, and one LLM client instead of copy-pasting them.

This is the *engine room* — interesting to a tech-screen reviewer who
wants to see how the abstractions are factored. If you're looking for the
apps that actually produce a tailored PDF, start at the consumers above.

## What's in here

| Package | Role |
|---|---|
| `tailor_core.llm` | `LLMClient` `Protocol` + `ClaudeCliClient` (subprocess wrapper around `claude --print`, prompts piped via file + stdin to dodge `ARG_MAX`) + `FakeLLMClient`. |
| `tailor_core.jd` | JD fetcher (httpx + selectolax), deterministic regex pass, LLM-driven extractor, `JobRequirements` model. |
| `tailor_core.context` | `UserContext` loader. Reads `resume.yaml` + `work_history/*.md` + `projects/*.md` + `cover_letters/*.md` + `git_audit/*.md` and enriches each project with a local-repo scan. |
| `tailor_core.local_projects` | Recursive scanner for a project folder: READMEs, manifests (`pyproject.toml`, `package.json`, `Dockerfile`, …), code-stats, structure, git log. |
| `tailor_core.context_files` | Uploaded-context-file store (PDF + text extraction). |
| `tailor_core.settings` | `BaseRuntimeSettings` pydantic model + PEP-695 generic SQLite / in-memory stores. Each consumer subclasses to add its own keys. |
| `tailor_core.runs` | `Run[TailoredT: BaseModel]` generic over the consumer's structured output, `RunStatus`, `RunEvent`, `RunsStore[TailoredT]` (SQLite + in-memory), `RunEventBus`, and the **`BaseOrchestrator[TailoredT, SettingsT]` template-method** that drives the shared phases. Subclasses implement `_tailor`, `_render`, `_verify`, `_verify_visually`. |
| `tailor_core.verifier` | Three passes: a text-mode LLM-judge scaffold (consumer-supplied `SYSTEM_PROMPT` + structured input → `VerificationResult`), a vision pass (pypdfium2 rasterise → Anthropic vision API → structured result), and a **deterministic numeric-claim fact-check** (`find_numeric_contradictions` / `verify_numeric_claims`) — pure, no LLM, flags a metric figure only when it both sits by a tracked keyword (tests / coverage / LOC / commits) and appears nowhere in the caller-supplied verified context. All content-agnostic; each consumer plugs in its own rubric. |

## Design notes worth flagging

- **Pydantic v2 + PEP-695 generics**: `Run[TailoredT]`, `RunsStore[TailoredT]`,
  `BaseOrchestrator[TailoredT, SettingsT]` are real generics, not type
  aliases. The store round-trips a consumer's subclass-typed run through
  SQLite with full schema validation.
- **`claude` CLI subprocess, not the per-token API**: the production LLM
  path shells out to Anthropic's Max-subscription `claude --print`. Auth
  flows through a long-lived `CLAUDE_CODE_OAUTH_TOKEN` env var generated
  once via `claude setup-token`. The vision verifier uses the Anthropic
  Python SDK with the same OAuth token, so no API key anywhere.
- **`ARG_MAX`-safe prompt delivery**: a fully-stitched tailoring prompt
  blows past the OS argv limit. The system prompt goes to a temp file
  consumed via `--system-prompt-file`; the user prompt is piped over stdin.
- **Template-method orchestration**: each consumer's pipeline is a
  four-method subclass — `_tailor`, `_render`, `_verify`,
  `_verify_visually`. The base owns the boring stuff: JD fetch + parse,
  context load, event publishing, run-state mutation, error funnelling
  (`_verify_safely` wraps consumer verifier errors into a non-blocking
  `CONCERNS` result so the run never fails on QC infrastructure).
- **Verifier separation of content from scaffold**: the LLM-judge call,
  JSON parse, schema validate, severity-promoting issue merge, fallback
  synthesis, PDF rasterisation, and Anthropic SDK invocation all live
  here generically. Each consumer supplies its own `SYSTEM_PROMPT` +
  structured input shape; the verifier's checklist is content-specific
  but the plumbing is shared.

## How a consumer wires it up

```python
from tailor_core.runs.orchestrator import BaseOrchestrator
from tailor_core.runs.store import SqliteRunsStore
from tailor_core.settings.models import BaseRuntimeSettings
from tailor_core.settings.store import SqliteSettingsStore
from tailor_core.verifier.scaffold import evaluate_judgement, check_pdf_length

class MyOutput(BaseModel): ...
class MySettings(BaseRuntimeSettings):
    template_name: str = "default.tex.j2"

class MyOrchestrator(BaseOrchestrator[MyOutput, MySettings]):
    def _tailor(self, requirements, context, request, context_files) -> MyOutput:
        return call_my_agent(requirements, context, self._llm, ...)

    def _render(self, tailored, requirements, output_dir) -> RenderResult:
        return render_via_latex(tailored, output_dir, stem=stem_for(requirements))

    def _verify(self, requirements, tailored, pdf_path) -> VerificationResult:
        result = evaluate_judgement(system_prompt=MY_PROMPT, ..., llm=self._llm)
        if (issue := check_pdf_length(pdf_path, target_max_pages=2)):
            result = merge_issue(result, issue)
        return result

# usage
store = SqliteRunsStore(tailored_cls=MyOutput, db_path=Path("~/.myapp/myapp.db"))
settings = SqliteSettingsStore(settings_cls=MySettings, db_path=Path("~/.myapp/myapp.db"))
orch = MyOrchestrator(runs_store=store, settings_store=settings, llm_client=ClaudeCliClient())
```

That's roughly what
[`resumeai/runs/orchestrator.py`](https://github.com/duketis/resumeai/blob/main/resumeai/runs/orchestrator.py)
and
[`coverletterai/runs/orchestrator.py`](https://github.com/duketis/coverletterai/blob/main/coverletterai/runs/orchestrator.py)
look like.

## Engineering bar

Python 3.12, `mypy --strict`, `ruff` strict (lint + format), conventional-
commits granularity, GPG-signed. `./Tools/quality-gate.sh` runs the local
check; CI on every push and PR mirrors it.

```bash
./Tools/quality-gate.sh    # ruff check + ruff format --check + mypy + pytest
```

348 tests at the time of writing. Coverage emphasis is on the boundary
modules (LLM client subprocess plumbing, generic SQLite round-trips,
template-method hook dispatch).

## Status

Pre-1.0. The public API surface is liable to change as the two consumer
apps mature. Pinning consumers to a `git+https://...@<sha>` ref rather
than `@main` is the safe default once you start depending on it from
elsewhere.

## License

MIT. See [LICENSE](LICENSE).
