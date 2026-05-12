"""Scan a local project directory into a plain-text summary.

What the scanner returns (in order):

1. **PROJECT** — name + absolute path.
2. **DOCS** — every README / CLAUDE.md / PLAN_*.md / ARCHITECTURE*.md /
   DESIGN*.md found within ``docs_max_depth`` levels, each truncated
   individually to ``per_doc_max_chars`` so a giant top-level README
   can't crowd out a per-service one. Total doc content capped at
   ``docs_total_max_chars`` so a monorepo doesn't blow up the prompt.
3. **MANIFESTS** — ``pyproject.toml`` / ``package.json`` / ``Cargo.toml``
   / ``go.mod`` / ``Gemfile`` / ``requirements*.txt`` / ``Dockerfile``
   found within ``docs_max_depth`` levels, each truncated. Gives the
   agent the REAL tech stack (parsed deps), not whatever the candidate
   handwrote in a .md.
4. **CODE STATS** — file count + line count grouped by extension.
   Concrete numbers ("Python: 18,432 LOC across 231 files") the agent
   can lift into resume bullets verbatim.
5. **STRUCTURE** — non-secret top-level files and immediate subdirs
   (skipping the privacy-sensitive list).
6. **GIT LOG** — when ``.git/`` is present and ``git`` is on PATH, runs
   ``git log`` filtered to ``author_email`` (when supplied) and
   summarises: total commits, date range, recent commit subjects, and
   per-month commit counts. Capped at ``git_max_chars``.

Each section is best-effort and returns its placeholder text on failure
so a single broken scan never blocks the whole context payload.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_README_MAX_CHARS = 8000
DEFAULT_GIT_MAX_CHARS = 12000
DEFAULT_GIT_MAX_COMMITS = 1000
DEFAULT_DOCS_MAX_DEPTH = 2
DEFAULT_PER_DOC_MAX_CHARS = 3000
DEFAULT_DOCS_TOTAL_MAX_CHARS = 30000
DEFAULT_PER_MANIFEST_MAX_CHARS = 2000
DEFAULT_MANIFESTS_TOTAL_MAX_CHARS = 15000
DEFAULT_CODE_STATS_MAX_FILES = 5000  # safety cap to keep the walk bounded

_README_NAMES: tuple[str, ...] = (
    "README.md",
    "README.rst",
    "README.txt",
    "Readme.md",
    "readme.md",
    "README",
)

# Top-level entries we *don't* show in the structure listing OR descend into.
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".cache",
        ".idea",
        ".vscode",
        ".DS_Store",
    }
)

_SKIP_FILE_PREFIXES: tuple[str, ...] = (".env",)
_SKIP_FILE_SUBSTRINGS: tuple[str, ...] = ("credentials", "secret", "client_secret")


class ScanError(RuntimeError):
    """Raised when the path can't be scanned (missing dir, permission, etc.)."""


def scan_project(
    path: str | Path,
    *,
    name: str | None = None,
    author_email: str | None = None,
    git_max_chars: int = DEFAULT_GIT_MAX_CHARS,
    git_max_commits: int = DEFAULT_GIT_MAX_COMMITS,
    docs_max_depth: int = DEFAULT_DOCS_MAX_DEPTH,
    per_doc_max_chars: int = DEFAULT_PER_DOC_MAX_CHARS,
    docs_total_max_chars: int = DEFAULT_DOCS_TOTAL_MAX_CHARS,
    per_manifest_max_chars: int = DEFAULT_PER_MANIFEST_MAX_CHARS,
    manifests_total_max_chars: int = DEFAULT_MANIFESTS_TOTAL_MAX_CHARS,
    code_stats_max_files: int = DEFAULT_CODE_STATS_MAX_FILES,
) -> str:
    """Scan ``path`` and return the assembled plain-text summary.

    Recursively pulls docs, manifests, code stats, structure, and git
    history -- enough surface for an LLM agent to write resume bullets
    with concrete numbers and real tech-stack signals, not paraphrases
    of a hand-written .md.
    """
    project_path = Path(path).expanduser().resolve()
    if not project_path.is_dir():
        raise ScanError(f"{project_path} is not a directory")

    project_name = name or project_path.name
    parts: list[str] = [
        f"PROJECT: {project_name}",
        f"PATH: {project_path}",
        "",
        _docs_section(
            project_path,
            max_depth=docs_max_depth,
            per_doc_max_chars=per_doc_max_chars,
            total_max_chars=docs_total_max_chars,
        ),
        "",
        _manifests_section(
            project_path,
            max_depth=docs_max_depth,
            per_manifest_max_chars=per_manifest_max_chars,
            total_max_chars=manifests_total_max_chars,
        ),
        "",
        _code_stats_section(project_path, max_files=code_stats_max_files),
        "",
        _structure_section(project_path),
        "",
        _git_section(project_path, author_email, git_max_chars, git_max_commits),
    ]
    return "\n".join(parts).strip() + "\n"


# -- Docs (recursive) -------------------------------------------------------


# Filenames the recursive docs collector treats as project documentation.
# Case-insensitive match on the filename only (not the full path).
_DOC_FILENAME_RE = re.compile(
    r"^(README(\.md|\.rst|\.txt)?|CLAUDE\.md|"
    r"PLAN[_-].*\.md|ARCHITECTURE.*\.md|DESIGN.*\.md|"
    r"OVERVIEW.*\.md|ROADMAP.*\.md|CONTRIBUTING\.md)$",
    re.IGNORECASE,
)


def _docs_section(
    project_path: Path,
    *,
    max_depth: int,
    per_doc_max_chars: int,
    total_max_chars: int,
) -> str:
    """Collect README + CLAUDE.md + PLAN/ARCHITECTURE/DESIGN docs recursively.

    Each doc truncated individually so a giant top-level README can't
    crowd out a per-service one; total budget capped so a monorepo with
    50 READMEs doesn't blow up the LLM prompt.
    """
    matches = _find_docs(project_path, max_depth=max_depth)
    if not matches:
        return "## DOCS\n_(no README / CLAUDE.md / PLAN_*.md / ARCHITECTURE found)_"

    chunks: list[str] = ["## DOCS"]
    total_used = 0
    for doc_path in matches:
        rel = doc_path.relative_to(project_path)
        try:
            content = doc_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            chunks.append(f"### {rel}\n_(could not read: {exc})_")
            continue
        if len(content) > per_doc_max_chars:
            content = (
                content[:per_doc_max_chars]
                + f"\n\n…({len(content) - per_doc_max_chars} more chars truncated)"
            )
        # Reserve headroom for the next section even if we're near budget.
        remaining = total_max_chars - total_used
        if remaining <= 200:
            chunks.append(
                f"### …({len(matches) - len(chunks) + 1} more doc(s) skipped — total budget hit)"
            )
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n\n…(truncated to fit overall scan budget)"
        chunks.append(f"### {rel}\n{content}")
        total_used += len(content)
    return "\n\n".join(chunks)


def _find_docs(project_path: Path, *, max_depth: int) -> list[Path]:
    """Breadth-first walk for doc files, stopping at ``max_depth`` levels."""
    found: list[Path] = []
    # (depth, dir) queue.
    queue: list[tuple[int, Path]] = [(0, project_path)]
    while queue:
        depth, current = queue.pop(0)
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in children:
            if _should_skip(child):
                continue
            if child.is_file() and _DOC_FILENAME_RE.match(child.name):
                found.append(child)
            elif child.is_dir() and depth < max_depth:
                queue.append((depth + 1, child))
    # Stable order: top-level docs first (depth 0), then subfolder docs.
    found.sort(key=lambda p: (len(p.relative_to(project_path).parts), str(p).lower()))
    return found


# -- Manifests (recursive) --------------------------------------------------


# Build/dep manifests the scanner pulls verbatim so the agent sees the
# REAL tech stack (deps in pyproject.toml, scripts in package.json, etc.)
# not just whatever the candidate wrote in a .md.
_MANIFEST_FILENAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements.in",
        "Pipfile",
        "poetry.lock",  # only first ~2KB -- top of file lists deps
        "package.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.toml",
        "go.mod",
        "Gemfile",
        "Gemfile.lock",
        "build.gradle",
        "build.gradle.kts",
        "pom.xml",
        "CMakeLists.txt",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
    }
)


def _manifests_section(
    project_path: Path,
    *,
    max_depth: int,
    per_manifest_max_chars: int,
    total_max_chars: int,
) -> str:
    """Pull every dependency / build manifest within ``max_depth`` levels."""
    matches = _find_files_by_name(
        project_path,
        names=_MANIFEST_FILENAMES,
        max_depth=max_depth,
    )
    if not matches:
        return "## MANIFESTS\n_(no dep / build manifests found)_"

    chunks: list[str] = ["## MANIFESTS"]
    total_used = 0
    for manifest_path in matches:
        rel = manifest_path.relative_to(project_path)
        try:
            content = manifest_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            chunks.append(f"### {rel}\n_(could not read: {exc})_")
            continue
        if len(content) > per_manifest_max_chars:
            content = (
                content[:per_manifest_max_chars]
                + f"\n\n…({len(content) - per_manifest_max_chars} more chars truncated)"
            )
        remaining = total_max_chars - total_used
        if remaining <= 200:
            chunks.append(
                f"### …({len(matches) - len(chunks) + 1} more manifest(s) skipped — budget hit)"
            )
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n\n…(truncated to fit overall scan budget)"
        chunks.append(f"### {rel}\n```\n{content}\n```")
        total_used += len(content)
    return "\n\n".join(chunks)


# -- Code stats -------------------------------------------------------------


# Source-file extensions worth counting. Keeps the stats focused on
# code volume rather than asset / data file noise.
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".rb",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".swift",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".scala",
        ".php",
        ".dart",
        ".lua",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".svelte",
    }
)


def _code_stats_section(project_path: Path, *, max_files: int) -> str:
    """File count + line count grouped by extension across the project.

    Walks the whole tree (skipping the standard ``_SKIP_DIR_NAMES`` set)
    up to ``max_files`` files so an enormous monorepo doesn't blow up
    the scanner. Reports descending by line count -- the biggest
    languages first so the agent sees the project's primary stack at
    a glance.
    """
    file_counts: Counter[str] = Counter()
    line_counts: Counter[str] = Counter()
    seen_files = 0
    truncated = False
    for entry in _walk_files(project_path):
        if seen_files >= max_files:
            truncated = True
            break
        seen_files += 1
        ext = entry.suffix.lower()
        if ext not in _CODE_EXTENSIONS:
            continue
        file_counts[ext] += 1
        try:
            with entry.open(encoding="utf-8", errors="replace") as fh:
                line_counts[ext] += sum(1 for _ in fh)
        except OSError:
            continue

    if not file_counts:
        return "## CODE STATS\n_(no recognised source files found)_"

    lines: list[str] = ["## CODE STATS"]
    if truncated:
        lines.append(f"_(walked first {max_files} files; some content not counted)_")
    ranked = sorted(file_counts.items(), key=lambda kv: line_counts[kv[0]], reverse=True)
    for ext, files in ranked:
        loc = line_counts[ext]
        lines.append(f"- {ext}: {loc:,} LOC across {files:,} file(s)")
    return "\n".join(lines)


# -- Generic helpers --------------------------------------------------------


def _find_files_by_name(project_path: Path, *, names: frozenset[str], max_depth: int) -> list[Path]:
    """BFS for files whose ``Path.name`` exactly matches one of ``names``."""
    found: list[Path] = []
    queue: list[tuple[int, Path]] = [(0, project_path)]
    while queue:
        depth, current = queue.pop(0)
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in children:
            if _should_skip(child):
                continue
            if child.is_file() and child.name in names:
                found.append(child)
            elif child.is_dir() and depth < max_depth:
                queue.append((depth + 1, child))
    found.sort(key=lambda p: (len(p.relative_to(project_path).parts), str(p).lower()))
    return found


def _walk_files(project_path: Path):  # type: ignore[no-untyped-def]
    """Yield every non-skipped file under ``project_path``."""
    stack: list[Path] = [project_path]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if _should_skip(child):
                continue
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                yield child


# -- Structure --------------------------------------------------------------


def _structure_section(project_path: Path) -> str:
    """Top-level + one level of subdirs. Skips privacy-sensitive entries."""
    try:
        entries = sorted(project_path.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return f"## STRUCTURE\n_(could not list: {exc})_"

    lines: list[str] = ["## STRUCTURE"]
    for entry in entries:
        if _should_skip(entry):
            continue
        if entry.is_dir():
            lines.append(f"- {entry.name}/")
            try:
                children = sorted(entry.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            shown = 0
            for child in children:
                if _should_skip(child):
                    continue
                suffix = "/" if child.is_dir() else ""
                lines.append(f"  - {child.name}{suffix}")
                shown += 1
                if shown >= 20:
                    lines.append(f"  - …({len(children) - 20} more)")
                    break
        else:
            lines.append(f"- {entry.name}")
    if len(lines) == 1:
        lines.append("_(no non-skipped entries at top level)_")
    return "\n".join(lines)


def _should_skip(entry: Path) -> bool:
    name = entry.name
    if name in _SKIP_DIR_NAMES:
        return True
    lowered = name.lower()
    if any(lowered.startswith(prefix) for prefix in _SKIP_FILE_PREFIXES):
        return True
    return any(token in lowered for token in _SKIP_FILE_SUBSTRINGS)


# -- Git log ---------------------------------------------------------------


def _git_section(
    project_path: Path,
    author_email: str | None,
    max_chars: int,
    max_commits: int,
) -> str:
    if not (project_path / ".git").exists():
        return "## GIT LOG\n_(not a git repository)_"
    git_bin = shutil.which("git")
    if not git_bin:
        return "## GIT LOG\n_(git binary not found on PATH)_"

    raw_log = _run_git_log(git_bin, project_path, author_email, max_commits)
    if raw_log.startswith("## GIT LOG\n_("):
        return raw_log  # error-text placeholder
    lines = [line for line in raw_log.splitlines() if line.strip()]
    if not lines:
        scope = f" by {author_email}" if author_email else " in this repo"
        return f"## GIT LOG\n_(no commits{scope})_"

    timestamps, months, subjects = _aggregate_git_lines(lines)
    if not timestamps:
        scope = f" by {author_email}" if author_email else " in this repo"
        return f"## GIT LOG\n_(no commits{scope})_"

    section = _format_git_summary(timestamps, months, subjects, author_email)
    if len(section) > max_chars:
        section = section[:max_chars] + f"\n\n…({len(section) - max_chars} more chars truncated)"
    return section


def _run_git_log(
    git_bin: str, project_path: Path, author_email: str | None, max_commits: int
) -> str:
    cmd = [
        git_bin,
        "-C",
        str(project_path),
        "log",
        f"--max-count={max_commits}",
        "--pretty=format:%ai|%H|%s",
    ]
    if author_email:
        cmd.append(f"--author={author_email}")
    try:
        proc = subprocess.run(  # noqa: S603 — args are a list, no shell expansion
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"## GIT LOG\n_(git log failed: {exc})_"
    if proc.returncode != 0:
        return f"## GIT LOG\n_(git log returned {proc.returncode}: {proc.stderr.strip()})_"
    return proc.stdout


def _aggregate_git_lines(
    lines: list[str],
) -> tuple[list[datetime], Counter[str], list[str]]:
    timestamps: list[datetime] = []
    months: Counter[str] = Counter()
    subjects: list[str] = []
    for raw in lines:
        try:
            iso, _sha, subject = raw.split("|", 2)
        except ValueError:
            continue
        try:
            ts = datetime.fromisoformat(iso.replace(" ", "T", 1))
        except ValueError:
            continue
        timestamps.append(ts)
        months[ts.strftime("%Y-%m")] += 1
        subjects.append(subject)
    return timestamps, months, subjects


def _format_git_summary(
    timestamps: list[datetime],
    months: Counter[str],
    subjects: list[str],
    author_email: str | None,
) -> str:
    section_lines: list[str] = ["## GIT LOG"]
    scope_label = f" (filter: --author={author_email})" if author_email else ""
    earliest = min(timestamps).date().isoformat()
    latest = max(timestamps).date().isoformat()
    section_lines.append(
        f"Total commits scanned: {len(timestamps)}{scope_label}. Date range: {earliest} → {latest}."
    )
    # ``months`` is always non-empty here — every parsed timestamp adds an
    # entry, and the caller short-circuits on no timestamps before we get
    # here.
    section_lines.append("")
    section_lines.append("**Per-month commit counts (most active months first):**")
    for month, count in months.most_common(12):
        section_lines.append(f"- {month}: {count}")
    section_lines.append("")
    section_lines.append("**Recent commit subjects (newest first):**")
    for subject in subjects[:80]:
        section_lines.append(f"- {subject}")
    return "\n".join(section_lines)
