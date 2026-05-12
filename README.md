# ai-tailor-core

Shared building blocks for AI-driven tailoring pipelines.

`ai-tailor-core` is the library underneath
[resumeai](https://github.com/duketis/resumeai) (job-description → tailored
resume PDF) and
[coverletterai](https://github.com/duketis/coverletterai) (job-description +
tailored resume → tailored cover letter PDF). Both apps share the same
ingest, orchestration, and verification scaffolding; the LaTeX template and
agent prompt are the only things that differ between them.

## What's in the box

| Package | Purpose |
|---------|---------|
| `tailor_core.llm`            | `LLMClient` protocol, `ClaudeCliClient` (subprocess wrapper around the `claude` CLI), `FakeLLMClient` for tests. ARG_MAX-safe prompt delivery (system prompt via file, user prompt via stdin). |
| `tailor_core.jd`             | Job-description fetcher (httpx + selectolax), text parser, deterministic + LLM extractor, `JobRequirements` model. |
| `tailor_core.settings`       | `BaseRuntimeSettings` pydantic model + SQLite / in-memory stores. Apps subclass to add their own keys (eg `template_name`). |
| `tailor_core.runs`           | `Run`, `RunStatus`, `RunEvent` models, `RunsStore`, `RunEventBus`, `BaseOrchestrator` skeleton with hooks for `_tailor()` and `_render()`. |
| `tailor_core.verifier`       | LLM-judge scaffolding (call → parse → build `VerificationResult`), `pypdfium2` vision rasteriser + Anthropic vision client, page-count helper. Apps supply their own `SYSTEM_PROMPT` and structured input. |
| `tailor_core.context`        | `UserContext` loader (`resume.yaml`, `work_history/*.md`, `projects/*.md`, `cover_letters/*.md`, `git_audit/*.md`). Entry models for each kind. |
| `tailor_core.local_projects` | Recursive scanner that mines READMEs, manifests (`pyproject.toml`, `package.json`, …), code stats, structure, and git log from local project folders. |
| `tailor_core.context_files`  | Storage + extraction (PDF/text) for user-uploaded supplementary documents. |
| `tailor_core.docker`         | Shared Dockerfile fragment + compose template + `CLAUDE_CODE_OAUTH_TOKEN` loader. Apps overlay. |

## Status

Pre-release. Internal use only at this point — extracted from `resumeai` to
unblock `coverletterai`. The public API is liable to change.

## Engineering notes

- Python 3.12, `mypy --strict`, `ruff` strict, 100% line + branch coverage on
  every module. `./Tools/quality-gate.sh` runs the canonical check locally.
- The LLM path is Anthropic's `claude` CLI subprocess (Max-subscription
  OAuth), **not** the per-token API key. Long-lived OAuth via
  `CLAUDE_CODE_OAUTH_TOKEN`.
- Vision QC uses the Anthropic Python SDK with the same OAuth token.

## License

MIT. See [LICENSE](LICENSE).
