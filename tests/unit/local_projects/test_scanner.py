"""Local-project scanner tests — README, structure, git log."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tailor_core.local_projects.scanner import ScanError, scan_project

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _git_init(repo_path: Path, *, with_commit: bool = True) -> None:
    """Initialise a git repo at ``repo_path`` for tests."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "alex@example.com"],
        cwd=repo_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Alex Sample"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo_path, check=True)
    if with_commit:
        (repo_path / "hello.txt").write_text("hi")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "initial commit"],
            cwd=repo_path,
            check=True,
        )


# -- error paths -----------------------------------------------------------


def test_scan_raises_when_path_doesnt_exist(tmp_path: Path) -> None:
    with pytest.raises(ScanError, match="not a directory"):
        scan_project(tmp_path / "nope")


def test_scan_raises_when_path_is_file(tmp_path: Path) -> None:
    file_path = tmp_path / "x.txt"
    file_path.write_text("hi")
    with pytest.raises(ScanError, match="not a directory"):
        scan_project(file_path)


# -- DOCS section ----------------------------------------------------------


def test_scan_includes_top_level_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# My Project\n\nSome text.")
    output = scan_project(tmp_path, name="proj")
    assert "## DOCS" in output
    assert "### README.md" in output
    assert "# My Project" in output


def test_scan_truncates_long_per_doc(tmp_path: Path) -> None:
    long = "x" * 10000
    (tmp_path / "README.md").write_text(long)
    output = scan_project(tmp_path, per_doc_max_chars=500)
    assert "more chars truncated" in output


def test_scan_handles_missing_docs(tmp_path: Path) -> None:
    output = scan_project(tmp_path)
    assert "no README / CLAUDE.md / PLAN_*.md / ARCHITECTURE" in output


def test_scan_falls_back_to_alternative_readme_filenames(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("plaintext readme")
    output = scan_project(tmp_path)
    assert "### README.txt" in output


def test_scan_handles_unreadable_doc(tmp_path: Path, mocker: MockerFixture) -> None:
    (tmp_path / "README.md").write_text("ok")
    mocker.patch.object(Path, "read_text", side_effect=OSError("permission denied"))
    output = scan_project(tmp_path)
    assert "could not read" in output


def test_scan_pulls_subfolder_readmes_recursively(tmp_path: Path) -> None:
    """Monorepo-style projects expose per-service READMEs the agent needs."""
    (tmp_path / "README.md").write_text("# Top README")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "README.md").write_text("# Engine docs\n\nbacktrader stuff.")
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "CLAUDE.md").write_text("# Dashboard CLAUDE\n\nclaude agent setup.")
    output = scan_project(tmp_path)
    assert "### README.md" in output
    assert "### engine/README.md" in output
    assert "backtrader stuff" in output
    assert "### dashboard/CLAUDE.md" in output
    assert "claude agent setup" in output


def test_scan_pulls_plan_and_architecture_docs(tmp_path: Path) -> None:
    """``PLAN_*.md`` and ``ARCHITECTURE*.md`` are part of the doc surface
    the agent needs to write meaningful bullets."""
    (tmp_path / "PLAN_OPTIONS.md").write_text("plan for options engine")
    (tmp_path / "ARCHITECTURE.md").write_text("the system shape")
    output = scan_project(tmp_path)
    assert "### PLAN_OPTIONS.md" in output
    assert "plan for options engine" in output
    assert "### ARCHITECTURE.md" in output


def test_scan_respects_docs_max_depth(tmp_path: Path) -> None:
    """A doc nested deeper than ``max_depth`` is NOT pulled."""
    deep_dir = tmp_path / "a" / "b" / "c"
    deep_dir.mkdir(parents=True)
    (deep_dir / "README.md").write_text("too deep")
    output = scan_project(tmp_path, docs_max_depth=1)
    assert "too deep" not in output


def test_scan_caps_docs_total_budget(tmp_path: Path) -> None:
    """When the cumulative doc bytes exceed the total cap, later docs
    get skipped with a budget-hit marker."""
    # Two big docs; first eats the whole budget, second should be skipped.
    (tmp_path / "README.md").write_text("a" * 6000)
    (tmp_path / "PLAN_X.md").write_text("b" * 6000)
    output = scan_project(tmp_path, per_doc_max_chars=6000, docs_total_max_chars=4000)
    # The total cap fires either as the "more doc(s) skipped" marker or as
    # mid-doc truncation depending on which path the budget hits.
    assert "scan budget" in output or "total budget hit" in output


# -- MANIFESTS section -----------------------------------------------------


def test_scan_pulls_pyproject_and_package_json_manifests(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["fastapi", "polars"]\n'
    )
    (tmp_path / "package.json").write_text('{"name": "front", "dependencies": {"react": "18.0.0"}}')
    output = scan_project(tmp_path)
    assert "## MANIFESTS" in output
    assert "### pyproject.toml" in output
    assert "fastapi" in output
    assert "### package.json" in output
    assert "react" in output


def test_scan_pulls_subfolder_manifests_too(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "Dockerfile").write_text("FROM python:3.12\nRUN pip install fastapi")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"name": "fe"}')
    output = scan_project(tmp_path)
    assert "### api/Dockerfile" in output
    assert "FROM python:3.12" in output
    assert "### frontend/package.json" in output


def test_scan_handles_missing_manifests(tmp_path: Path) -> None:
    output = scan_project(tmp_path)
    assert "no dep / build manifests" in output


def test_scan_truncates_long_manifest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("x" * 5000)
    output = scan_project(tmp_path, per_manifest_max_chars=200)
    assert "more chars truncated" in output


def test_scan_handles_unreadable_manifest(tmp_path: Path, mocker: MockerFixture) -> None:
    (tmp_path / "pyproject.toml").write_text("ok")
    real_read_text = Path.read_text

    def selective_boom(self: Path, *a: object, **kw: object) -> str:
        if self.name == "pyproject.toml":
            raise OSError("perm denied")
        return real_read_text(self, *a, **kw)  # type: ignore[arg-type]

    mocker.patch.object(Path, "read_text", selective_boom)
    output = scan_project(tmp_path)
    assert "could not read" in output


def test_scan_caps_manifests_total_budget(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("a" * 6000)
    (tmp_path / "package.json").write_text("b" * 6000)
    output = scan_project(tmp_path, per_manifest_max_chars=6000, manifests_total_max_chars=4000)
    assert "scan budget" in output or "budget hit" in output


# -- CODE STATS section ----------------------------------------------------


def test_scan_reports_code_stats_by_extension(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("a\nb\nc\n")
    (tmp_path / "helper.py").write_text("x\ny\n")
    (tmp_path / "ui.tsx").write_text("const x = 1;\n")
    output = scan_project(tmp_path)
    assert "## CODE STATS" in output
    # Python lines: 5; TSX lines: 1. Python ranked first (more LOC).
    py_idx = output.index("- .py:")
    tsx_idx = output.index("- .tsx:")
    assert py_idx < tsx_idx
    assert "5 LOC across 2 file(s)" in output
    assert "1 LOC across 1 file(s)" in output


def test_scan_walks_into_subfolders_for_code_stats(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a\nb\n")
    (tmp_path / "src" / "deep").mkdir()
    (tmp_path / "src" / "deep" / "b.py").write_text("c\n")
    output = scan_project(tmp_path)
    assert "3 LOC across 2 file(s)" in output


def test_scan_handles_no_code_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("nothing source-like")
    output = scan_project(tmp_path)
    assert "no recognised source files" in output


def test_scan_respects_code_stats_max_files(tmp_path: Path) -> None:
    """A massive monorepo doesn't blow up the scanner -- walks stop at the cap."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(20):
        (src / f"f_{i:02d}.py").write_text("x\n")
    output = scan_project(tmp_path, code_stats_max_files=5)
    assert "walked first 5 files" in output


def test_scan_skips_unreadable_code_file(tmp_path: Path, mocker: MockerFixture) -> None:
    """A file we can't read is silently skipped (no scan-wide failure)."""
    (tmp_path / "good.py").write_text("a\nb\n")
    (tmp_path / "bad.py").write_text("c\n")

    real_open = Path.open

    def selective_boom(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "bad.py":
            raise OSError("perm denied")
        return real_open(self, *args, **kwargs)  # type: ignore[call-overload]

    mocker.patch.object(Path, "open", selective_boom)
    output = scan_project(tmp_path)
    # Good file still counted; total only includes good.py's 2 lines.
    assert "2 LOC across 2 file(s)" in output


# -- Structure section -----------------------------------------------------


def test_scan_lists_top_level_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("...")
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("...")
    output = scan_project(tmp_path)

    assert "## STRUCTURE" in output
    assert "- src/" in output
    assert "- tests/" in output
    assert "- pyproject.toml" in output
    assert "  - main.py" in output


def test_scan_skips_secret_dirs_and_files(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "foo").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".env").write_text("SECRET=x")
    (tmp_path / "credentials.json").write_text("{}")
    (tmp_path / "src").mkdir()
    output = scan_project(tmp_path)

    assert "node_modules" not in output
    assert "__pycache__" not in output
    assert ".env" not in output
    assert "credentials.json" not in output
    assert "- src/" in output


def test_scan_skips_secret_files_inside_subdir(tmp_path: Path) -> None:
    """The skip rules apply to children of subdirs too, not just the top level."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("ok")
    (tmp_path / "src" / ".env").write_text("SECRET=x")
    (tmp_path / "src" / "credentials.json").write_text("{}")
    output = scan_project(tmp_path)
    assert "main.py" in output
    assert ".env" not in output
    assert "credentials.json" not in output


def test_scan_caps_subdir_listing_at_20_entries(tmp_path: Path) -> None:
    sub = tmp_path / "many"
    sub.mkdir()
    for i in range(30):
        (sub / f"file_{i:02d}.txt").write_text(str(i))
    output = scan_project(tmp_path)
    assert "(10 more)" in output


def test_scan_handles_empty_top_level(tmp_path: Path) -> None:
    output = scan_project(tmp_path)
    assert "no non-skipped entries at top level" in output


def test_scan_handles_unreadable_subdir(tmp_path: Path, mocker: MockerFixture) -> None:
    (tmp_path / "sub").mkdir()
    real_iterdir = Path.iterdir

    def boom(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == "sub":
            raise OSError("perm denied")
        return real_iterdir(self)

    mocker.patch.object(Path, "iterdir", boom)
    output = scan_project(tmp_path)
    # Top-level listing tried to descend; the broken subdir is silently skipped
    # rather than crashing the whole scan.
    assert "- sub/" in output


def test_scan_handles_unreadable_top_level(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(Path, "iterdir", side_effect=OSError("perm denied"))
    output = scan_project(tmp_path)
    assert "could not list" in output


# -- Git section -----------------------------------------------------------


def test_scan_reports_non_git_repo(tmp_path: Path) -> None:
    output = scan_project(tmp_path)
    assert "## GIT LOG" in output
    assert "not a git repository" in output


def test_scan_summarises_git_log(tmp_path: Path) -> None:
    _git_init(tmp_path)
    output = scan_project(tmp_path, author_email="alex@example.com")

    assert "## GIT LOG" in output
    assert "Total commits scanned: 1" in output
    assert "filter: --author=alex@example.com" in output
    assert "initial commit" in output
    assert "Per-month commit counts" in output


def test_scan_runs_git_log_without_email_filter(tmp_path: Path) -> None:
    _git_init(tmp_path)
    output = scan_project(tmp_path)
    assert "Total commits scanned: 1" in output
    assert "filter:" not in output


def test_scan_reports_no_commits_when_repo_empty(tmp_path: Path) -> None:
    """Empty git repo (init'd, no commits) — git log exits 128 with a known
    error message; we surface it verbatim rather than silently succeeding."""
    _git_init(tmp_path, with_commit=False)
    output = scan_project(tmp_path, author_email="someone@else.com")
    assert "## GIT LOG" in output
    # Real git surfaces "does not have any commits yet" — we surface that.
    assert "git log returned 128" in output


def test_scan_reports_no_commits_when_filter_matches_nothing(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Author filter that matches no commits returns empty stdout (exit 0)."""
    _git_init(tmp_path)
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )
    output = scan_project(tmp_path, author_email="someone@else.com")
    assert "no commits by someone@else.com" in output


def test_scan_reports_no_commits_without_filter_when_empty(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Empty stdout without author filter."""
    _git_init(tmp_path)
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )
    output = scan_project(tmp_path)
    assert "no commits in this repo" in output


def test_scan_handles_missing_git_binary(tmp_path: Path, mocker: MockerFixture) -> None:
    _git_init(tmp_path)
    mocker.patch("shutil.which", return_value=None)
    output = scan_project(tmp_path)
    assert "git binary not found on PATH" in output


def test_scan_handles_git_log_failure(tmp_path: Path, mocker: MockerFixture) -> None:
    _git_init(tmp_path)
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=128, stdout="", stderr="fatal: bad rev"
        ),
    )
    output = scan_project(tmp_path)
    assert "git log returned 128" in output
    assert "fatal: bad rev" in output


def test_scan_handles_git_log_timeout(tmp_path: Path, mocker: MockerFixture) -> None:
    _git_init(tmp_path)
    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=15),
    )
    output = scan_project(tmp_path)
    assert "git log failed" in output


def test_scan_skips_malformed_log_lines(tmp_path: Path, mocker: MockerFixture) -> None:
    _git_init(tmp_path)
    bad_output = "\n".join(
        [
            "no-pipes-here",
            "wrong-fields|only-one",
            "2026-05-09 10:00:00 +0000|abc123|good commit",
            "not-an-iso-date|abc123|skipped",
        ]
    )
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=bad_output, stderr=""
        ),
    )
    output = scan_project(tmp_path)
    assert "good commit" in output
    assert "Total commits scanned: 1" in output


def test_scan_reports_no_commits_when_all_log_lines_malformed(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Every log line failed to parse — same as having no commits at all."""
    _git_init(tmp_path)
    bad_output = "no-pipes-here\nalso-bad\nstill-bad"
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=bad_output, stderr=""
        ),
    )
    output = scan_project(tmp_path, author_email="alex@example.com")
    assert "no commits by alex@example.com" in output


def test_scan_truncates_long_git_section(tmp_path: Path, mocker: MockerFixture) -> None:
    _git_init(tmp_path)
    massive_output = "\n".join(
        f"2026-05-09 10:00:00 +0000|abc{i:04d}|commit {i}" for i in range(500)
    )
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=massive_output, stderr=""
        ),
    )
    output = scan_project(tmp_path, git_max_chars=300)
    assert "more chars truncated" in output


# -- assembly -------------------------------------------------------------


def test_scan_assembles_all_sections_with_path_and_name(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("readme content")
    (tmp_path / "src").mkdir()
    output = scan_project(tmp_path, name="myproj")
    assert output.startswith("PROJECT: myproj")
    assert f"PATH: {tmp_path}" in output
    assert "## README" in output
    assert "## STRUCTURE" in output
    assert "## GIT LOG" in output


def test_scan_default_name_is_directory_basename(tmp_path: Path) -> None:
    project = tmp_path / "wow"
    project.mkdir()
    output = scan_project(project)
    assert "PROJECT: wow" in output
