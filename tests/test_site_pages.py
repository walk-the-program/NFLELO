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


@pytest.mark.parametrize("page", PAGES)
def test_brand_stylesheet_and_no_retired_theme_code(page):
    """Soft Form and the dark theme are retired: every page links brand.css, and no page loads theme.js or soft-form.css."""
    html = read(f"{page}.html")
    assert 'href="brand.css"' in html and 'href="styles.css"' in html
    assert "soft-form" not in html and "theme.js" not in html and "data-theme-choice" not in html
    assert not (SITE / "soft-form.css").exists() and not (SITE / "theme.js").exists()


@pytest.mark.parametrize("page", PAGES)
def test_one_primary_action_per_page(page):
    assert read(f"{page}.html").count('class="b-primary"') == 1


def test_no_hex_colors_outside_the_brand_tokens():
    for name in ("styles.css", "common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", read(name)), name


# The navy Raw Brutalist brief (context/brand.md): every hex the site may contain, and nothing else.
BRIEF_HEXES = {
    "#20436b", "#eff8a5", "#ff4d37", "#26221c", "#f9f8f5", "#706d68", "#d9d8d4",
    "#ffffff", "#141414", "#dfe2e4", "#f7f8e8",
}
RETIRED_HEXES = {
    "#a11413", "#d14f34", "#4a3122", "#261c1c", "#f9f5f5", "#706868", "#d9d4d4", "#eedada", "#f3dad6",
}


def test_brand_css_defines_only_the_navy_brief_hexes():
    found = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", read("brand.css"))}
    assert found <= BRIEF_HEXES, found - BRIEF_HEXES
    assert not (found & RETIRED_HEXES)


def test_highlighter_band_and_on_tokens_match_the_brief():
    css = read("brand.css")
    assert re.search(r"--on-secondary:\s*#141414", css) and re.search(r"--on-accent:\s*#141414", css)
    assert re.search(r"\.hl \{[^}]*background: var\(--accent\); color: var\(--on-accent\)", css)


@pytest.mark.parametrize("page", PAGES)
def test_pages_use_only_brief_hexes_and_a_flow_hero(page):
    html = read(f"{page}.html")
    assert {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", html)} <= BRIEF_HEXES
    assert html.count("hero-flow") == 1


@pytest.mark.parametrize("asset", ["pattern-halftone.svg", "pattern-glyphgrid.svg", "hero-flow.svg"])
def test_pattern_assets_are_in_the_brief_colorway(asset):
    svg = (SITE / "assets" / asset).read_text()
    assert {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", svg)} <= BRIEF_HEXES


def test_secondary_is_never_a_line_dot_or_text_color():
    """Pale yellow is about 1.1:1 on the canvas: no stroke, border, text color or marker uses it without an ink edge."""
    css = read("styles.css")
    for decl in re.findall(r"(?:^|[;{\s])((?:stroke|color|border(?:-[a-z]+)?)\s*:[^;}]*var\(--secondary\)[^;}]*)", css):
        assert False, decl
    for name in ("common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        for line in read(name).splitlines():
            if "var(--secondary)" in line:
                assert "fill:" in line and "pale: true" in line, (name, line.strip())
