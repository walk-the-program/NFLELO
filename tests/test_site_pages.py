"""Static checks on the site's HTML pages: the shared nav, local asset links, and that every element id a page
script looks up exists in its HTML, plus the ASCII Art brand brief (context/brand-ascii.md): its hexes, monospace-only
type, one primary CTA, the accent highlight word with its caret, and the generated pattern assets. The behavior is checked
in a browser (see context/elo-site.md)."""
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
    """Soft Form and the theme switcher are retired: every page links brand.css, and no page loads theme.js or soft-form.css."""
    html = read(f"{page}.html")
    assert 'href="brand.css"' in html and 'href="styles.css"' in html
    assert "soft-form" not in html and "theme.js" not in html and "data-theme-choice" not in html
    assert not (SITE / "soft-form.css").exists() and not (SITE / "theme.js").exists()


@pytest.mark.parametrize("page", PAGES)
def test_one_primary_action_per_page(page):
    assert read(f"{page}.html").count('class="b-primary"') == 1


JS_FILES = ("common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js", "ascii.js")


def test_no_hex_colors_outside_the_brand_tokens():
    for name in ("styles.css",) + JS_FILES:
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", read(name)), name


# The ASCII Art brief (context/brand-ascii.md): every hex the site may contain, and nothing else.
BRIEF_HEXES = {
    "#7db7eb", "#f0c5c6", "#f0ec5e", "#1c2126", "#f5f7f9", "#9ea1a5", "#3d4146",
    "#141414", "#28333e", "#3e3b40", "#292e33", "#595d61",
}
RETIRED_HEXES = {
    "#20436b", "#eff8a5", "#ff4d37", "#26221c", "#f9f8f5", "#706d68", "#d9d8d4", "#dfe2e4", "#f7f8e8", "#ffffff",
    "#eaecee", "#64686c", "#cbced0", "#aa810e", "#fbfbfc",
}


def test_brand_css_defines_only_the_ascii_brief_hexes():
    found = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", read("brand.css"))}
    assert found <= BRIEF_HEXES, found - BRIEF_HEXES
    assert BRIEF_HEXES <= found, BRIEF_HEXES - found
    assert not (found & RETIRED_HEXES)


def test_on_tokens_dark_canvas_and_highlight_word_match_the_brief():
    css = read("brand.css")
    for tok in ("--on-primary", "--on-secondary", "--on-accent"):
        assert re.search(tok + r":\s*#141414", css)
    assert re.search(r"--canvas:\s*#1c2126", css) and re.search(r"--ink:\s*#f5f7f9", css) and re.search(r"--accent-ink:\s*#f0ec5e", css)
    # the highlight word is accent text with no fill and no band
    hl = re.search(r"\.hl \{([^}]*)\}", css).group(1)
    assert "color: var(--accent-ink)" in hl and "background" not in hl
    assert "color-scheme: dark" in css


def test_carets_are_static_under_reduced_motion():
    css = read("brand.css")
    assert "@keyframes caret-blink" in css
    assert re.search(r"@media \(prefers-reduced-motion: reduce\) \{ \.caret \{ animation: none", css)


def test_monospace_only_no_proportional_faces():
    """JetBrains Mono for the headline, labels, interface and art; IBM Plex Mono for reading text; nothing proportional."""
    css = read("brand.css") + read("styles.css")
    assert "JetBrains Mono" in css and "IBM Plex Mono" in css
    assert not re.search(r"Figtree|Nunito|Inter\b|Helvetica|Arial|sans-serif|system-ui|-apple-system|Georgia|Times", css)
    for decl in re.findall(r"font-family:\s*([^;}]+)", css):
        assert "monospace" in decl or decl.strip().startswith("var(--"), decl
    for page in PAGES:
        html = read(f"{page}.html")
        assert "family=IBM+Plex+Mono" in html and "family=JetBrains+Mono" in html
        assert "Figtree" not in html and "Nunito" not in html


@pytest.mark.parametrize("page", PAGES)
def test_pages_use_only_brief_hexes_and_the_ascii_hero(page):
    html = read(f"{page}.html")
    assert {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", html)} <= BRIEF_HEXES
    assert html.count('id="ascii"') == 1 and 'src="ascii.js"' in html
    assert "hero-flow" not in html and "tex-glyph" not in html
    # the headline panel and the art are separate: the figure is not inside the copy panel
    copy = re.search(r'<section class="panel hero-copy".*?</section>', html, re.S).group(0)
    assert "ascii-pre" not in copy
    # exactly one blinking-caret highlight word in the h1: the last word, in accent text, then the caret
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S).group(1)
    assert h1.count('class="hl"') == 1 and re.search(r'</span><span class="caret hl-caret" aria-hidden="true">', h1)
    # one section band and at most one textured panel per page; the CTA is a text bar with a caret
    assert html.count('class="band"') == 1 and html.count("tex-halftone") <= 1
    cta = re.search(r'<a class="b-primary"[^>]*>(.*?)</a>', html, re.S).group(1)
    assert cta.lstrip().startswith("<span>&gt; ") or cta.lstrip().startswith("&gt; ")
    assert 'class="caret"' in cta
    assert '<span class="acc-block"' in html


@pytest.mark.parametrize("asset", ["tile-halftone.svg", "band-gradpanes.svg"])
def test_pattern_assets_are_in_the_brief_colorway(asset):
    svg = (SITE / "assets" / asset).read_text()
    assert {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", svg)} <= BRIEF_HEXES
    if asset.startswith("band"):
        assert 'preserveAspectRatio="xMidYMid slice"' in svg


def test_ascii_hero_field_is_a_digit_grid_on_the_density_ramp():
    import json
    d = json.loads((SITE / "assets" / "ascii-hero.json").read_text())
    assert d["ramp"] == " .:-=+*#%@"
    assert len(d["field"]) == d["rows"] and all(len(r) == d["cols"] and set(r) <= set("0123456789") for r in d["field"])
    assert {c for r in d["field"] for c in r} == set("0123456789")   # the whole ramp is used
    for gone in ("hero-flow.svg", "pattern-glyphgrid.svg", "pattern-halftone.svg", "hero-aura.png", "hero-terrain.png"):
        assert not (SITE / "assets" / gone).exists()


def test_secondary_never_sets_body_text():
    """Hard rule 2: secondary is for prompts, progress, status brackets and chart marks, never body text."""
    css = read("styles.css")
    body_selectors = (".deck", ".hero-deck", ".method p", ".note", ".fnote", ".find p", ".mc-what", ".mc-read", ".tile .sub", "body")
    for sel in body_selectors:
        for rule in re.findall(re.escape(sel) + r"\s*\{([^}]*)\}", css):
            assert not re.search(r"(?<![-a-z])color:\s*var\(--secondary\)", rule), sel
    assert not re.search(r"\.(?:b-primary|cta)[^{]*\{[^}]*background:\s*var\(--(?:secondary|accent)\)", css + read("brand.css"))
