"""Tests for the deterministic numeric-claim fact-checker.

Full line + branch coverage. The headline case is jobai run #50: the
cover letter said "705+ tests at 89.5% coverage" while the verified
context records 1,126 backend + 86 frontend tests at 100% coverage.
"""

from __future__ import annotations

from tailor_core.verifier.deterministic import (
    _fmt,
    _nearest_metric,
    _normalise_number,
    find_numeric_contradictions,
    verify_numeric_claims,
)
from tailor_core.verifier.models import IssueSeverity, VerificationStatus

# The real verified snippet shape jobai pins (decomposed + qualitative).
VERIFIED = (
    "VERIFIED jobai project stats: 1,126 backend (pytest) tests and "
    "86 frontend (Vitest) tests at 100% line + branch coverage; "
    "~15,700 lines of code; 540 commits."
)


# --------------------------------------------------------------------------
# _normalise_number
# --------------------------------------------------------------------------


def test_normalise_plain_thousands() -> None:
    assert _normalise_number("1,126") == 1126.0


def test_normalise_tilde_prefix() -> None:
    assert _normalise_number("~15,700") == 15700.0


def test_normalise_plus_suffix() -> None:
    assert _normalise_number("705+") == 705.0


def test_normalise_percent() -> None:
    assert _normalise_number("89.5%") == 89.5


def test_normalise_k_suffix_lower_and_upper() -> None:
    assert _normalise_number("38k") == 38000.0
    assert _normalise_number("38K") == 38000.0


def test_normalise_empty_after_stripping_returns_none() -> None:
    # cleaned becomes "" -> first operand of the guard.
    assert _normalise_number("~+") is None


def test_normalise_nondigit_but_nonempty_returns_none() -> None:
    # cleaned is "." (truthy) with no digit -> second operand of the guard.
    assert _normalise_number(".") is None


# --------------------------------------------------------------------------
# _nearest_metric (bare-number classification)
# --------------------------------------------------------------------------


def test_nearest_metric_keyword_after_number() -> None:
    assert _nearest_metric("1126 tests pass", 0, 4) == "test count"


def test_nearest_metric_keyword_before_number() -> None:
    text = "commits: 540"
    assert _nearest_metric(text, 9, 12) == "commit count"


def test_nearest_metric_none_when_no_keyword_in_window() -> None:
    text = "the lonely number 9999 sits with nothing relevant anywhere near"
    assert _nearest_metric(text, 18, 22) is None


def test_nearest_metric_picks_closest_keyword() -> None:
    # 'commits' is adjacent; 'tests' is far -> commit count wins.
    text = "tests were many but here we count 540 commits in the repo"
    idx = text.index("540")
    assert _nearest_metric(text, idx, idx + 3) == "commit count"


def test_nearest_metric_word_boundary_ignores_loc_in_block() -> None:
    # 'loc' must not match inside 'block'.
    text = "the block held 12 widgets"
    idx = text.index("12")
    assert _nearest_metric(text, idx, idx + 2) is None


# --------------------------------------------------------------------------
# find_numeric_contradictions
# --------------------------------------------------------------------------


def test_no_verified_context_returns_empty() -> None:
    assert find_numeric_contradictions("anything 705 tests", None) == ()


def test_blank_verified_context_returns_empty() -> None:
    assert find_numeric_contradictions("705 tests", "") == ()


def test_whitespace_only_verified_context_returns_empty() -> None:
    assert find_numeric_contradictions("705 tests", "   \n\t ") == ()


def test_run50_regression_flags_both_fabrications() -> None:
    artefact = (
        "jobai is my project; it runs 705+ tests at 89.5% coverage and ships as a single container."
    )
    issues = find_numeric_contradictions(artefact, VERIFIED)
    cats = {i.category for i in issues}
    assert cats == {"fabricated_metric"}
    assert all(i.severity is IssueSeverity.ERROR for i in issues)
    metrics = " ".join(i.message for i in issues)
    assert "test count" in metrics and "705" in metrics
    assert "coverage" in metrics and "89.5" in metrics
    # The verified figures are surfaced for the retry to copy verbatim.
    assert "1126" in metrics and "100" in metrics


def test_correct_decomposed_numbers_not_flagged() -> None:
    # Both components appear verbatim in the verified snapshot.
    artefact = "jobai has 1,126 backend tests and 86 frontend tests at 100% coverage."
    assert find_numeric_contradictions(artefact, VERIFIED) == ()


def test_decomposition_reconciles_to_verified_total() -> None:
    # Verified states only the TOTAL; artefact decomposes it. Sum guard
    # must prevent a false positive (never block a correct artefact).
    verified_total = "The suite is 1,212 tests at 100% coverage."
    artefact = "I wrote 1,126 backend tests and 86 frontend tests."
    assert find_numeric_contradictions(artefact, verified_total) == ()


def test_metric_without_verified_ground_truth_is_not_flagged() -> None:
    # Verified context says nothing about commits -> absence is the LLM
    # judge's call, not the deterministic gate's.
    verified = "jobai has 1,126 tests at 100% coverage."
    artefact = "jobai has 1,126 tests at 100% coverage across 9999 commits."
    assert find_numeric_contradictions(artefact, verified) == ()


def test_all_claimed_corroborated_but_sum_unmatched_not_flagged() -> None:
    # Each claimed test number appears in verified; their sum (1212) does
    # not -> guard is False, bad is empty, no issue.
    verified = "jobai: 1,126 backend tests; 86 frontend tests; 100% coverage."
    artefact = "jobai has 1,126 backend tests and 86 frontend tests."
    assert find_numeric_contradictions(artefact, verified) == ()


def test_stray_number_far_from_any_metric_is_ignored() -> None:
    # "2019" has no tracked keyword within the window -> classified
    # None and skipped; only the real fabrication ("705 tests") flags.
    artefact = (
        "Founded in 2019, a long while before any of the relevant work "
        "ever began around here. jobai runs 705 tests."
    )
    issues = find_numeric_contradictions(artefact, VERIFIED)
    assert len(issues) == 1
    assert "test count" in issues[0].message
    assert "2019" not in issues[0].message


def test_single_fabricated_metric_one_issue() -> None:
    artefact = "jobai runs 540 commits but only 99 tests."
    issues = find_numeric_contradictions(artefact, VERIFIED)
    assert len(issues) == 1
    assert "test count" in issues[0].message
    assert "99" in issues[0].message
    assert "verbatim" in issues[0].suggestion


def test_multiple_fabricated_metrics_sorted() -> None:
    artefact = "jobai runs 705 tests at 12.3% coverage."
    issues = find_numeric_contradictions(artefact, VERIFIED)
    # Two metrics: 'coverage' and 'test count', sorted by metric name.
    assert [i.message.split()[1] for i in issues] == ["coverage", "test"]


# --------------------------------------------------------------------------
# verify_numeric_claims (result wrapper)
# --------------------------------------------------------------------------


def test_verify_numeric_claims_passed_when_clean() -> None:
    result = verify_numeric_claims("nothing numeric to see", VERIFIED)
    assert result.status is VerificationStatus.PASSED
    assert result.issues == ()


def test_verify_numeric_claims_failed_when_contradicted() -> None:
    result = verify_numeric_claims("jobai runs 705 tests", VERIFIED)
    assert result.status is VerificationStatus.FAILED
    assert len(result.issues) == 1
    assert "Deterministic" in result.rationale


# --------------------------------------------------------------------------
# _fmt
# --------------------------------------------------------------------------


def test_fmt_integer_has_no_trailing_zero() -> None:
    assert _fmt(1126.0) == "1126"


def test_fmt_non_integer_keeps_decimal() -> None:
    assert _fmt(89.5) == "89.5"
