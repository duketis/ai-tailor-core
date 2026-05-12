"""Tests for the JD HTTP fetcher + HTML cleaner."""

from __future__ import annotations

import httpx
import pytest
import respx

from tailor_core.jd.fetcher import (
    FetchError,
    detect_ats,
    extract_main_text,
    fetch_jd,
)

# -- detect_ats --------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://boards.greenhouse.io/acme/jobs/123", "greenhouse"),
        ("https://jobs.lever.co/acme/job-123", "lever"),
        ("https://jobs.ashbyhq.com/acme/role-123", "ashby"),
        ("https://acme.workable.com/jobs/123", "workable"),
        ("https://apply.workable.com/acme/j/123", "workable"),
        ("https://acme.smartrecruiters.com/jobs/123", "smartrecruiters"),
        ("https://example.com/careers/123", "unknown"),
    ],
)
def test_detect_ats(url: str, expected: str) -> None:
    assert detect_ats(url) == expected


# -- extract_main_text ------------------------------------------------------


def test_extract_main_text_strips_chrome_and_returns_main() -> None:
    html = """
    <html>
      <head><style>p { color: red }</style><script>alert(1)</script></head>
      <body>
        <header>SITE NAV</header>
        <nav>nav links</nav>
        <main>
          <h1>Senior Engineer</h1>
          <p>You will <strong>build</strong> things.</p>
        </main>
        <aside>ads</aside>
        <footer>copyright</footer>
      </body>
    </html>
    """

    text = extract_main_text(html)

    assert "Senior Engineer" in text
    assert "build" in text
    assert "things" in text
    for chrome in ("SITE NAV", "nav links", "ads", "copyright", "alert(1)"):
        assert chrome not in text


def test_extract_main_text_falls_back_to_article_then_body() -> None:
    html_article = "<html><body><article>article body</article></body></html>"
    html_body = "<html><body>just body</body></html>"

    assert "article body" in extract_main_text(html_article)
    assert "just body" in extract_main_text(html_body)


def test_extract_main_text_returns_empty_for_empty_html() -> None:
    assert extract_main_text("") == ""
    assert extract_main_text("   ") == ""


def test_extract_main_text_returns_empty_for_html_without_body() -> None:
    # selectolax always synthesises a body — but an html with no body-equivalent
    # element should still degrade gracefully.
    html = "<!doctype html><script>x()</script>"
    text = extract_main_text(html)
    # Either empty or close to it; the script content must have been stripped.
    assert "x()" not in text


def test_extract_main_text_collapses_whitespace() -> None:
    html = "<body><main><p>line one</p>\n\n\n  <p>line   two</p></main></body>"

    text = extract_main_text(html)

    # Inline runs collapsed; blank-line gaps preserved as single blank line.
    assert "line two" in text
    assert "line   two" not in text
    assert "\n\n\n" not in text


# -- fetch_jd ---------------------------------------------------------------


_SAMPLE_HTML = """
<html><body><main>
  <h1>Senior Software Engineer</h1>
  <p>Acme Corp is hiring.</p>
</main></body></html>
"""


@respx.mock
def test_fetch_jd_returns_fetched_jd_with_cleaned_text() -> None:
    url = "https://boards.greenhouse.io/acme/jobs/123"
    respx.get(url).mock(return_value=httpx.Response(200, text=_SAMPLE_HTML))

    fetched = fetch_jd(url)

    assert fetched.source_url == url
    assert fetched.ats == "greenhouse"
    assert "Senior Software Engineer" in fetched.cleaned_text
    assert "Acme Corp is hiring" in fetched.cleaned_text
    assert fetched.raw_html == _SAMPLE_HTML


@respx.mock
def test_fetch_jd_raises_on_4xx() -> None:
    url = "https://example.com/jd"
    respx.get(url).mock(return_value=httpx.Response(404))

    with pytest.raises(FetchError, match="404"):
        fetch_jd(url)


@respx.mock
def test_fetch_jd_raises_on_5xx() -> None:
    url = "https://example.com/jd"
    respx.get(url).mock(return_value=httpx.Response(503))

    with pytest.raises(FetchError, match="503"):
        fetch_jd(url)


@respx.mock
def test_fetch_jd_raises_on_network_error() -> None:
    url = "https://example.com/jd"
    respx.get(url).mock(side_effect=httpx.ConnectError("boom"))

    with pytest.raises(FetchError, match="network error"):
        fetch_jd(url)


@respx.mock
def test_fetch_jd_uses_supplied_http_client_and_does_not_close_it() -> None:
    url = "https://example.com/jd"
    respx.get(url).mock(return_value=httpx.Response(200, text=_SAMPLE_HTML))

    client = httpx.Client(timeout=5.0)
    try:
        fetch_jd(url, http_client=client)
        # The injected client should still be open afterwards.
        assert client.is_closed is False
    finally:
        client.close()
