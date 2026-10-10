"""Static checks on the site's HTML pages: the shared nav, local asset links, and that every element id a page
script looks up exists in its HTML. The behavior is checked in a browser (see context/elo-site.md)."""
import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent / "site"
PAGES = ["index", "models", "plays", "winprob", "fourth", "playcalling"]
NAV_LINKS = ["plays.html", "winprob.html", "fourth.html", "playcalling.html", "models.html"]
SCRIPT = {"index": "app", "models": "models", "plays": "plays", "winprob": "winprob", "fourth": "fourth", "playcalling": "playcalling"}


def read(name: str) -> str:
    return (SITE / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_has_the_model_nav_links(page):
    html = read(f"{page}.html")
    nav = re.search(r"<ul class=\"nav-links\">(.*?)</ul>", html, re.S).group(1)
    for href in NAV_LINKS:
        if href == f"{page}.html":
            continue  # the page itself is reached through its own section links
        assert f'href="{href}"' in nav, f"{page}.html nav is missing {href}"


@pytest.mark.parametrize("page", PAGES)
def test_local_assets_exist(page):
    html = read(f"{page}.html")
    for ref in re.findall(r'(?:src|href)="([^"#:]+\.(?:js|css|html))"', html):
        assert (SITE / ref).exists(), f"{page}.html links to missing {ref}"
    assert f'src="{SCRIPT[page]}.js"' in html


@pytest.mark.parametrize("page", ["plays", "winprob", "fourth", "playcalling"])
def test_script_ids_exist_in_html(page):
    html = read(f"{page}.html")
    js = read(f"{page}.js")
    ids = set(re.findall(r"""(?:done|setText|\$)\(\s*['"]#?([a-z][\w-]*)['"]""", js))
    for line in js.splitlines():
        if "var ALL" in line:
            ids |= set(re.findall(r"'([a-z]+-body)'", line))
    missing = sorted(i for i in ids if f'id="{i}"' not in html and f"id='{i}'" not in html and i not in _built_in_js(js, i))
    assert not missing, f"{page}.js looks up ids that {page}.html does not have: {missing}"


def _built_in_js(js: str, i: str) -> set:
    """Ids that the script creates itself (for example the game picker)."""
    return {i} if re.search(rf"id:\s*'{re.escape(i)}'", js) else set()


@pytest.mark.parametrize("page", ["plays", "winprob", "fourth", "playcalling"])
def test_live_region_id_is_not_reused(page):
    """The shared aria-live region is #live; a section with that id broke the fourth-down page once."""
    html = read(f"{page}.html")
    assert len(re.findall(r'id="live"', html)) == 1


@pytest.mark.parametrize("page", ["plays", "winprob", "fourth", "playcalling"])
def test_no_dashes_or_betting_words_in_page_copy(page):
    text = read(f"{page}.html") + read(f"{page}.js")
    assert "–" not in text and "—" not in text
    lowered = re.sub(r"[^a-z ]", " ", text.lower())
    for word in ("sportsbook", "parlay", "wager", "units"):
        assert word not in lowered.split(), word
