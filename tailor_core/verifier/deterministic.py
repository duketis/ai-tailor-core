"""Deterministic numeric-claim fact-checker.

The LLM verifier (``scaffold.evaluate_judgement``) is probabilistic: it
*usually* catches a fabricated stat, but a confidently-recalled number
("jobai runs 705+ tests at 89.5% coverage") can slip past both the
generation prompt and the LLM judge. This module is the deterministic
backstop -- pure string/number arithmetic, no LLM, zero nondeterminism.

It flags a metric number in the artefact **only when all three hold**:

1. the number is the nearest-classified to a tracked project-metric
   keyword (tests / coverage / lines-of-code / commits), AND
2. the VERIFIED CONTEXT carries ground truth for that metric class
   (at least one number classified to the same keyword) -- so we never
   punish a legitimately unsourced metric (that is the LLM judge's
   softer call, not ours; the hard-won "contradiction, not absence"
   invariant), AND
3. that exact number appears **nowhere** in the verified context.

Rule 3 keeps the false-positive rate near zero: a fabricated figure
(705, 89.5) is by definition absent from the verified snapshot, while a
correctly *decomposed* real figure ("1,126 backend + 86 frontend tests"
when the verified line says the same) still has every component present
somewhere in the verified text, so it is not flagged even if the
verified text states only the combined total elsewhere.
"""

from __future__ import annotations

import re

from tailor_core.verifier.models import (
    IssueSeverity,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)

# A tracked metric class -> the keywords that signal it. These are the
# candidate-controlled-repo figures that actually get hallucinated.
_METRIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "test count": ("test", "tests"),
    "coverage": ("coverage",),
    "lines of code": ("lines of code", "loc", "kloc", "sloc"),
    "commit count": ("commit", "commits"),
}

# Max char distance between a number and a metric keyword to associate them.
_WINDOW = 45

# A number token: optional ~, digits with optional thousands separators,
# optional decimal, optional + ("705+"), optional k/K ("38k"), optional %.
_NUMBER_RE = re.compile(r"~?\d[\d,]*(?:\.\d+)?\+?[kK]?%?")


def _normalise_number(token: str) -> float | None:
    """Parse a matched number token into a comparable float.

    ``~15,700`` -> 15700.0, ``705+`` -> 705.0, ``89.5%`` -> 89.5,
    ``38k`` -> 38000.0, ``1,126`` -> 1126.0. Returns ``None`` when the
    token carries no digit after stripping decoration.
    """
    cleaned = token.strip().lstrip("~").rstrip("%").rstrip("+")
    multiplier = 1.0
    if cleaned[-1:] in ("k", "K"):
        multiplier = 1000.0
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace(",", "")
    if not cleaned or not any(ch.isdigit() for ch in cleaned):
        return None
    try:
        return float(cleaned) * multiplier
    except ValueError:  # pragma: no cover - regex guarantees parseability
        return None


def _all_numbers(text: str) -> set[float]:
    """Every normalised number anywhere in ``text``."""
    out: set[float] = set()
    for m in _NUMBER_RE.finditer(text):
        value = _normalise_number(m.group(0))
        if value is None:  # pragma: no cover - regex guarantees a digit
            continue
        out.add(value)
    return out


def _nearest_metric(text_lower: str, start: int, end: int) -> str | None:
    """Classify a number at ``[start, end)`` to its nearest metric keyword.

    Distance is measured edge-to-edge between the number span and a
    whole-word keyword occurrence (so ``loc`` does not match inside
    ``block``). Returns the closest metric class within ``_WINDOW``
    characters, or ``None`` when nothing tracked is near enough.
    """
    best_metric: str | None = None
    best_dist = _WINDOW + 1
    for metric, keywords in _METRIC_KEYWORDS.items():
        for kw in keywords:
            for km in re.finditer(rf"\b{re.escape(kw)}\b", text_lower):
                if km.end() <= start:
                    dist = start - km.end()
                elif km.start() >= end:
                    dist = km.start() - end
                else:  # pragma: no cover - a keyword never overlaps a number
                    dist = 0
                if dist < best_dist:
                    best_dist = dist
                    best_metric = metric
    return best_metric if best_dist <= _WINDOW else None


def _classify(token: str, text_lower: str, start: int, end: int) -> str | None:
    """Classify a number to a metric class.

    A ``%`` in the token is a strong, position-independent signal: in
    these résumés/letters a percentage is always a coverage figure, so
    ``89.5%`` / ``100%`` classify as coverage regardless of a ``tests``
    word that happens to sit a few characters away. Bare numbers fall
    back to nearest-keyword.
    """
    if "%" in token:
        return "coverage"
    return _nearest_metric(text_lower, start, end)


def _metric_numbers(text: str) -> dict[str, set[float]]:
    """Map each tracked metric class to the numbers claimed for it in ``text``."""
    lowered = text.lower()
    found: dict[str, set[float]] = {}
    for m in _NUMBER_RE.finditer(text):
        metric = _classify(m.group(0), lowered, m.start(), m.end())
        if metric is None:
            continue
        value = _normalise_number(m.group(0))
        if value is None:  # pragma: no cover - regex guarantees a digit
            continue
        found.setdefault(metric, set()).add(value)
    return found


def _fmt(value: float) -> str:
    """Render a normalised number without a trailing ``.0`` for integers."""
    return str(int(value)) if value.is_integer() else str(value)


def find_numeric_contradictions(
    artefact_text: str,
    verified_context: str | None,
) -> tuple[VerificationIssue, ...]:
    """Return one ERROR issue per metric class the artefact contradicts.

    Deterministic and pure. Returns an empty tuple when
    ``verified_context`` is absent/blank (no ground truth -> nothing to
    contradict) or when every metric number in the artefact is
    corroborated somewhere in the verified context.
    """
    if not verified_context or not verified_context.strip():
        return ()

    verified_by_metric = _metric_numbers(verified_context)
    verified_all = _all_numbers(verified_context)
    claimed_by_metric = _metric_numbers(artefact_text)

    issues: list[VerificationIssue] = []
    for metric, claimed in sorted(claimed_by_metric.items()):
        if not verified_by_metric.get(metric):
            # No verified figure for this metric class -> not our call.
            continue
        if sum(claimed) in verified_all:
            # The claimed components reconcile to a verified total
            # (eg "1,126 backend + 86 frontend" vs a verified "1,212
            # tests"). Decomposition of a real figure is not a
            # fabrication -- never block a correct artefact.
            continue
        bad = sorted(c for c in claimed if c not in verified_all)
        if not bad:
            continue
        bad_str = ", ".join(_fmt(n) for n in bad)
        truth_str = ", ".join(_fmt(n) for n in sorted(verified_by_metric[metric]))
        issues.append(
            VerificationIssue(
                severity=IssueSeverity.ERROR,
                category="fabricated_metric",
                message=(
                    f"Unverified {metric} claim: the artefact states "
                    f"{bad_str} but the verified context records "
                    f"{truth_str} for {metric}, and {bad_str} appears "
                    f"nowhere in the verified facts."
                ),
                suggestion=(
                    f"Replace the {metric} figure with the verified value "
                    f"({truth_str}) verbatim, or state it qualitatively "
                    f"with no number."
                ),
            )
        )
    return tuple(issues)


def verify_numeric_claims(
    artefact_text: str,
    verified_context: str | None,
) -> VerificationResult:
    """Standalone result wrapper around :func:`find_numeric_contradictions`.

    ``FAILED`` with one issue per contradicted metric, else ``PASSED``.
    Consumers that already hold an LLM-judge result should instead fold
    each issue from :func:`find_numeric_contradictions` in via
    :func:`tailor_core.verifier.scaffold.merge_issue` so the LLM
    summary/rationale is preserved.
    """
    issues = find_numeric_contradictions(artefact_text, verified_context)
    if not issues:
        return VerificationResult(
            status=VerificationStatus.PASSED,
            summary="No numeric claim contradicts the verified context.",
        )
    return VerificationResult(
        status=VerificationStatus.FAILED,
        summary=f"{len(issues)} metric claim(s) contradict the verified context.",
        issues=issues,
        rationale="Deterministic numeric-claim check (no LLM).",
    )


__all__ = [
    "find_numeric_contradictions",
    "verify_numeric_claims",
]
