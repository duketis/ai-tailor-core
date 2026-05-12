"""Fetch a JD page over HTTP, strip the chrome, return the visible text.

Phase 2 ships with a single tier-1 (httpx) fetcher. ATS-specific selectors
and a Playwright fallback are deferred — every public ATS in scope
(Greenhouse, Lever, Ashby, Workable, SmartRecruiters) returns the JD as
plain HTML accessible to httpx with a sensible User-Agent.

The cleaned text removes script/style/nav/footer/aside/header/iframe
content and collapses repeated whitespace. Where ``<main>`` or ``<article>``
exists we prefer it; otherwise we fall back to ``<body>``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from selectolax.parser import HTMLParser, Node

from tailor_core.jd.models import FetchedJD

USER_AGENT = (
    "resumeai/0.x (+https://github.com/duketis/resumeai) AppleWebKit/537.36 (KHTML, like Gecko)"
)
DEFAULT_TIMEOUT = 15.0

_STRIP_TAGS = {
    "script",
    "style",
    "nav",
    "footer",
    "aside",
    "header",
    "iframe",
    "noscript",
    "form",
    "svg",
    "button",
}


class FetchError(RuntimeError):
    """Raised when the URL can't be fetched or returns a non-success status."""


def fetch_jd(url: str, *, http_client: httpx.Client | None = None) -> FetchedJD:
    """Fetch + clean a JD URL.

    A custom ``http_client`` lets tests inject a transport (e.g. respx).
    """
    client_owned_locally = http_client is None
    client = http_client or httpx.Client(
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )
    try:
        try:
            resp = client.get(url)
        except httpx.HTTPError as exc:
            raise FetchError(f"network error fetching {url!r}: {exc}") from exc
        if resp.status_code >= httpx.codes.BAD_REQUEST:
            raise FetchError(f"non-success status {resp.status_code} fetching {url!r}")
        html = resp.text
    finally:
        if client_owned_locally:
            client.close()

    return FetchedJD(
        source_url=url,
        raw_html=html,
        cleaned_text=extract_main_text(html),
        fetched_at=datetime.now(UTC),
        ats=detect_ats(url),
    )


def detect_ats(url: str) -> str:
    """Coarse host-based ATS detection. Returns a stable string label."""
    host = urlparse(url).netloc.lower()
    if "greenhouse.io" in host:
        return "greenhouse"
    if "lever.co" in host:
        return "lever"
    if "ashbyhq.com" in host:
        return "ashby"
    if "workable.com" in host:
        return "workable"
    if "smartrecruiters.com" in host:
        return "smartrecruiters"
    return "unknown"


def extract_main_text(html: str) -> str:
    """Strip chrome and return the visible plain text of a page."""
    if not html.strip():
        return ""

    tree = HTMLParser(html)
    for tag in _STRIP_TAGS:
        for node in tree.css(tag):
            node.decompose()

    container = _pick_container(tree)
    raw = container.text(separator="\n")
    return _normalise_whitespace(raw)


def _pick_container(tree: HTMLParser) -> Node:
    """Prefer ``<main>``, then ``<article>``, then ``<body>``.

    selectolax always synthesises an ``<html><body>...</body></html>``
    skeleton for any non-empty input, so ``body`` is guaranteed; the early
    return in :func:`extract_main_text` handles the empty-input case.
    """
    for selector in ("main", "article"):
        nodes = tree.css(selector)
        if nodes:
            return nodes[0]
    return tree.css("body")[0]


_INLINE_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _normalise_whitespace(text: str) -> str:
    """Collapse inline whitespace runs and runs of blank lines."""
    # Strip per line first so all-whitespace lines become empty strings —
    # otherwise leading whitespace on otherwise-blank lines blocks the
    # blank-line collapse below.
    lines = [_INLINE_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    joined = "\n".join(lines)
    return _MULTI_NEWLINE_RE.sub("\n\n", joined).strip()
