"""Loader tests against both ad-hoc tmp_path layouts and the committed sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from tailor_core.context.loader import ContextLoadError, load_user_context

# -- happy path against the committed sample ---------------------------------


def test_loads_committed_sample_context(sample_context_root: Path) -> None:
    ctx = load_user_context(sample_context_root)

    assert ctx.resume is not None
    assert ctx.resume.name == "Alex Sample"
    assert ctx.resume.contact.email == "alex.sample@example.com"
    assert "Python" in ctx.resume.skills
    assert ctx.resume.education[0].institution == "Sample University"
    assert ctx.resume.certifications == ("AWS Solutions Architect Associate",)

    # Two work-history entries, ordered most-recent first by end date.
    titles = [entry.company for entry in ctx.work_history]
    assert titles == ["Acme Corp", "Beta Inc"]
    acme = ctx.work_history[0]
    assert acme.title == "Senior Software Engineer"
    assert "FastAPI" in acme.technologies
    assert acme.summary.startswith("Built and ran the data-ingestion platform")
    assert len(acme.bullets) == 3
    assert any("deduplication" in b for b in acme.bullets)

    # Git audit
    assert len(ctx.git_audit) == 1
    audit = ctx.git_audit[0]
    assert audit.repo == "acme/platform"
    assert audit.role == "Senior Software Engineer"
    assert "1,200 commits" in audit.summary

    # Cover letter
    assert len(ctx.cover_letters) == 1
    cover = ctx.cover_letters[0]
    assert cover.role == "Senior Software Engineer"
    assert cover.company == "Acme Corp"
    assert "high-trust team" in cover.body

    # Project
    assert len(ctx.projects) == 1
    project = ctx.projects[0]
    assert project.slug == "sample-tool"
    assert project.name == "sample-tool"
    assert project.url == "https://github.com/alex/sample-tool"
    assert project.stack is not None and "FastAPI" in project.stack
    assert "code-review showcase" in project.summary
    assert any("FastAPI surface" in b for b in project.bullets)
    assert "Important framing" in project.body


# -- empty / missing tolerance ----------------------------------------------


def test_returns_empty_context_when_root_missing(tmp_path: Path) -> None:
    ctx = load_user_context(tmp_path / "nope")
    assert ctx.is_empty()


def test_returns_empty_context_when_root_is_a_file(tmp_path: Path) -> None:
    f = tmp_path / "actually-a-file"
    f.write_text("hi")
    ctx = load_user_context(f)
    assert ctx.is_empty()


def test_partial_tree_loads_what_exists(tmp_path: Path) -> None:
    """Only resume.yaml present — work_history/, git_audit/, cover_letters/,
    and projects/ are all missing. Loader returns the resume and empty
    tuples."""
    (tmp_path / "resume.yaml").write_text("name: Alex\ncontact:\n  email: a@example.com\n")
    ctx = load_user_context(tmp_path)
    assert ctx.resume is not None
    assert ctx.work_history == ()
    assert ctx.git_audit == ()
    assert ctx.cover_letters == ()
    assert ctx.projects == ()


# -- resume.yaml error handling ---------------------------------------------


def test_raises_when_resume_yaml_top_level_is_not_a_mapping(tmp_path: Path) -> None:
    (tmp_path / "resume.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ContextLoadError, match="YAML mapping"):
        load_user_context(tmp_path)


def test_raises_on_invalid_resume_yaml(tmp_path: Path) -> None:
    (tmp_path / "resume.yaml").write_text("name: [unclosed\n")
    with pytest.raises(ContextLoadError, match="invalid YAML"):
        load_user_context(tmp_path)


def test_raises_on_resume_validation_error(tmp_path: Path) -> None:
    (tmp_path / "resume.yaml").write_text("name: ''\ncontact:\n  email: x\n")
    with pytest.raises(ContextLoadError):
        load_user_context(tmp_path)


# -- work history file handling ---------------------------------------------


def test_work_history_orders_most_recent_first(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "old.md").write_text(
        "---\ntitle: Eng\ncompany: Old\nstart: 2018-01\nend: 2020-01\n---\n\nbody\n"
    )
    (wh / "new.md").write_text(
        "---\ntitle: Eng\ncompany: New\nstart: 2020-02\nend: 2024-01\n---\n\nbody\n"
    )
    (wh / "current.md").write_text(
        "---\ntitle: Eng\ncompany: Current\nstart: 2024-02\n---\n\nbody\n"
    )

    ctx = load_user_context(tmp_path)

    assert [e.company for e in ctx.work_history] == ["Current", "New", "Old"]


def test_work_history_extracts_summary_paragraph(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text(
        "---\ntitle: Eng\ncompany: Acme\n---\n\n"
        "# heading shouldn't count\n\n"
        "Summary paragraph that should be picked up.\n\n"
        "Second paragraph not picked up.\n\n"
        "- bullet\n"
    )
    ctx = load_user_context(tmp_path)
    assert ctx.work_history[0].summary == "Summary paragraph that should be picked up."


def test_work_history_summary_skips_empty_paragraphs(tmp_path: Path) -> None:
    """Bodies with extra blank lines split into empty paragraphs which the
    summary extractor must skip past, not return."""
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\ntitle: Eng\ncompany: Acme\n---\n\n\n\nReal summary here.\n")
    ctx = load_user_context(tmp_path)
    assert ctx.work_history[0].summary == "Real summary here."


def test_work_history_summary_blank_when_only_bullets(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\ntitle: Eng\ncompany: Acme\n---\n\n- bullet 1\n- bullet 2\n")
    ctx = load_user_context(tmp_path)
    assert ctx.work_history[0].summary == ""


def test_work_history_extracts_all_bullet_styles(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text(
        "---\ntitle: Eng\ncompany: Acme\n---\n\n- dash bullet\n* star bullet\n+ plus bullet\n"
    )
    ctx = load_user_context(tmp_path)
    assert ctx.work_history[0].bullets == ("dash bullet", "star bullet", "plus bullet")


def test_work_history_raises_on_missing_required_frontmatter(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\ncompany: Acme\n---\n\nbody\n")
    with pytest.raises(ContextLoadError, match="title"):
        load_user_context(tmp_path)


def test_work_history_raises_on_unclosed_frontmatter(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\ntitle: Eng\ncompany: Acme\nno close ever\n")
    with pytest.raises(ContextLoadError, match="frontmatter opening"):
        load_user_context(tmp_path)


def test_work_history_treats_no_frontmatter_as_empty_metadata(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("Just a body, no frontmatter")
    with pytest.raises(ContextLoadError, match="title"):
        load_user_context(tmp_path)


def test_work_history_raises_on_non_mapping_frontmatter(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\n- not\n- a\n- mapping\n---\n\nbody\n")
    with pytest.raises(ContextLoadError, match="must be a YAML mapping"):
        load_user_context(tmp_path)


def test_work_history_handles_empty_frontmatter_block(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\n\n---\n\nbody\n")
    with pytest.raises(ContextLoadError, match="title"):
        load_user_context(tmp_path)


def test_work_history_raises_on_invalid_frontmatter_yaml(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\ntitle: [unclosed\n---\n\nbody\n")
    with pytest.raises(ContextLoadError, match="invalid frontmatter YAML"):
        load_user_context(tmp_path)


def test_work_history_coerces_non_string_optional_metadata(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text("---\ntitle: Eng\ncompany: Acme\nstart: 2022\n---\n\nbody\n")
    ctx = load_user_context(tmp_path)
    assert ctx.work_history[0].start == "2022"


def test_work_history_handles_missing_technologies_list(tmp_path: Path) -> None:
    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "x.md").write_text(
        "---\ntitle: Eng\ncompany: Acme\ntechnologies: not-a-list\n---\n\nbody\n"
    )
    ctx = load_user_context(tmp_path)
    assert ctx.work_history[0].technologies == ()


# -- git audit + cover letters ----------------------------------------------


def test_git_audit_skips_non_markdown_files(tmp_path: Path) -> None:
    ga = tmp_path / "git_audit"
    ga.mkdir()
    (ga / "x.md").write_text("---\nrepo: acme/x\n---\n\nsummary\n")
    (ga / "ignore.txt").write_text("not a markdown file")
    ctx = load_user_context(tmp_path)
    assert len(ctx.git_audit) == 1
    assert ctx.git_audit[0].repo == "acme/x"


def test_git_audit_raises_on_missing_repo_field(tmp_path: Path) -> None:
    ga = tmp_path / "git_audit"
    ga.mkdir()
    (ga / "x.md").write_text("---\nrole: Engineer\n---\n\nbody\n")
    with pytest.raises(ContextLoadError, match="repo"):
        load_user_context(tmp_path)


def test_cover_letter_body_strips_surrounding_whitespace(tmp_path: Path) -> None:
    cl = tmp_path / "cover_letters"
    cl.mkdir()
    (cl / "acme.md").write_text("---\nrole: Eng\ncompany: Acme\n---\n\n  Hi team,\n\nBody.\n\n")
    ctx = load_user_context(tmp_path)
    assert ctx.cover_letters[0].body.startswith("Hi team,")
    assert ctx.cover_letters[0].body.endswith("Body.")


def test_cover_letter_without_frontmatter_keeps_full_body(tmp_path: Path) -> None:
    cl = tmp_path / "cover_letters"
    cl.mkdir()
    (cl / "x.md").write_text("Just text, no frontmatter")
    ctx = load_user_context(tmp_path)
    assert ctx.cover_letters[0].body == "Just text, no frontmatter"
    assert ctx.cover_letters[0].role is None
    assert ctx.cover_letters[0].company is None


# -- projects ----------------------------------------------------------------


def test_projects_sort_by_slug(tmp_path: Path) -> None:
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "zeta.md").write_text("---\nproject: Zeta\n---\n\nzeta summary\n")
    (pr / "alpha.md").write_text("---\nproject: Alpha\n---\n\nalpha summary\n")
    ctx = load_user_context(tmp_path)
    assert [p.slug for p in ctx.projects] == ["alpha", "zeta"]


def test_projects_extract_summary_bullets_and_body(tmp_path: Path) -> None:
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "x.md").write_text(
        "---\n"
        "project: X\n"
        "url: https://example.com/x\n"
        "status: alpha\n"
        "stack: Python, FastAPI\n"
        "---\n\n"
        "First paragraph is the summary.\n\n"
        "- bullet one\n"
        "- bullet two\n"
    )
    ctx = load_user_context(tmp_path)
    project = ctx.projects[0]
    assert project.name == "X"
    assert project.url == "https://example.com/x"
    assert project.status == "alpha"
    assert project.stack == "Python, FastAPI"
    assert project.summary == "First paragraph is the summary."
    assert project.bullets == ("bullet one", "bullet two")
    assert project.body.startswith("First paragraph")


def test_projects_raises_on_missing_required_project_field(tmp_path: Path) -> None:
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "x.md").write_text("---\nurl: https://example.com\n---\n\nbody\n")
    with pytest.raises(ContextLoadError, match="project"):
        load_user_context(tmp_path)


def test_projects_skips_non_markdown_files(tmp_path: Path) -> None:
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "x.md").write_text("---\nproject: X\n---\n\nbody\n")
    (pr / "ignore.txt").write_text("not a markdown file")
    ctx = load_user_context(tmp_path)
    assert len(ctx.projects) == 1


def test_projects_tolerate_missing_optional_metadata(tmp_path: Path) -> None:
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "x.md").write_text("---\nproject: X\n---\n\nbody\n")
    ctx = load_user_context(tmp_path)
    project = ctx.projects[0]
    assert project.url is None
    assert project.status is None
    assert project.stack is None
    assert project.local_path is None
    assert project.scanned == ""


def test_project_with_local_path_scans_the_referenced_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``local_path`` is set, the loader runs the recursive scanner
    against ``RESUMEAI_PROJECTS_ROOT / local_path`` and stores the output
    on ``ProjectEntry.scanned``."""
    # Stage a fake "projects root" with one project inside it.
    projects_root = tmp_path / "host-personal"
    project_dir = projects_root / "fake-project"
    project_dir.mkdir(parents=True)
    (project_dir / "README.md").write_text("# Fake Project\n\nDoes a thing.\n")
    monkeypatch.setenv("RESUMEAI_PROJECTS_ROOT", str(projects_root))

    # And the resumeai UserContext side.
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "fake.md").write_text(
        "---\nproject: Fake\nlocal_path: fake-project\n---\n\nhand-written summary\n"
    )

    ctx = load_user_context(tmp_path)
    project = ctx.projects[0]
    assert project.local_path == "fake-project"
    assert "Fake Project" in project.scanned
    assert "Does a thing." in project.scanned


def test_project_local_path_pointing_to_missing_folder_degrades_to_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misconfigured ``local_path`` doesn't take down the whole context
    load -- it just yields an empty ``scanned`` and a logged warning."""
    monkeypatch.setenv("RESUMEAI_PROJECTS_ROOT", str(tmp_path / "missing-root"))
    pr = tmp_path / "projects"
    pr.mkdir()
    (pr / "fake.md").write_text("---\nproject: Fake\nlocal_path: nope\n---\n\nbody\n")
    ctx = load_user_context(tmp_path)
    assert ctx.projects[0].scanned == ""


# -- master resume + reference resumes --------------------------------------


def test_master_resume_loaded_from_root_text_file(tmp_path: Path) -> None:
    (tmp_path / "master-resume.txt").write_text("Authoritative master content here.\n")
    ctx = load_user_context(tmp_path)
    assert ctx.master_resume == "Authoritative master content here."


def test_master_resume_empty_when_file_absent(tmp_path: Path) -> None:
    ctx = load_user_context(tmp_path)
    assert ctx.master_resume == ""


def test_reference_resumes_loaded_from_root_tex_files(tmp_path: Path) -> None:
    (tmp_path / "Resume - Acme.tex").write_text("\\documentclass{article}\nAcme\n")
    (tmp_path / "Resume - Globex.tex").write_text("\\documentclass{article}\nGlobex\n")
    (tmp_path / "notes.txt").write_text("ignored")
    ctx = load_user_context(tmp_path)
    assert len(ctx.reference_resumes) == 2
    # Sorted alphabetically by filename.
    assert "Acme" in ctx.reference_resumes[0]
    assert "Globex" in ctx.reference_resumes[1]


def test_reference_resumes_empty_when_no_tex_files(tmp_path: Path) -> None:
    ctx = load_user_context(tmp_path)
    assert ctx.reference_resumes == ()
