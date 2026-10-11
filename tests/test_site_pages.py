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


@pytest.mark.parametrize("page", PAGES)
def test_no_dashes_or_betting_words_in_page_copy(page):
    text = read(f"{page}.html") + read(f"{SCRIPT[page]}.js")
    assert "\u2013" not in text and "\u2014" not in text
    lowered = re.sub(r"[^a-z ]", " ", text.lower())
    for word in ("sportsbook", "parlay", "wager", "units"):
        assert word not in lowered.split(), word


def test_no_dashes_in_stylesheets_and_common_script():
    for name in ("brand.css", "styles.css", "common.js"):
        text = read(name)
        assert "\u2013" not in text and "\u2014" not in text, name


BRIEF = Path(__file__).resolve().parent.parent / "context" / "brand.md"
HEX = re.compile(r"(?<![&\w])#[0-9a-fA-F]{3,8}\b")


@pytest.mark.parametrize("page", PAGES)
def test_brand_stylesheet_fonts_and_no_retired_theme_code(page):
    """Editorial Serif: every page links brand.css and styles.css, loads the three brief fonts (not Figtree),
    and carries no code from the retired Soft Form, dark theme, or Raw Brutalist looks."""
    html = read(f"{page}.html")
    assert 'href="brand.css"' in html and 'href="styles.css"' in html
    assert "Instrument+Serif" in html and "Newsreader" in html and "Nunito+Sans" in html and "Figtree" not in html
    assert "soft-form" not in html and "theme.js" not in html and "data-theme-choice" not in html
    assert "splash" not in html and "tagline" not in html
    assert not (SITE / "soft-form.css").exists() and not (SITE / "theme.js").exists()


@pytest.mark.parametrize("page", PAGES)
def test_one_primary_action_per_page(page):
    assert read(f"{page}.html").count('class="b-primary"') == 1


@pytest.mark.parametrize("page", PAGES)
def test_one_masthead_rule_and_no_second_double_rule(page):
    html = read(f"{page}.html")
    assert html.count('class="masthead-rule"') == 1
    assert read("brand.css").count("border-top: 3px solid") == 1  # the thick-over-thin rule is defined once
    assert "3px solid var(--ab-hair-strong)" not in read("styles.css")


@pytest.mark.parametrize("page", PAGES)
def test_headline_highlight_word_is_the_last_word_of_the_h1(page):
    """The brief's rule 8: once per view, the last word of the headline is an italic em in accent ink."""
    html = read(f"{page}.html")
    assert len(re.findall(r'class="hl"', html)) == 1
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S).group(1)
    assert re.search(r'<em class="hl">[A-Za-z]+</em>$', h1.strip()), h1
    css = read("brand.css")
    assert re.search(r"\.hl \{[^}]*font-style: italic[^}]*color: var\(--accent-ink\)", css)
    assert re.search(r"\.hl \{[^}]*font-weight: 400", css) and "background: none" in css


def test_headings_are_never_bold_and_ui_text_is_not_monospace():
    css = read("styles.css") + read("brand.css")
    assert re.search(r"h1, h2, h3 \{[^}]*font-weight: 400", css)
    assert "monospace" not in css and "--font-mono" not in css
    assert not re.search(r"font:\s*[6-9]00\s+[^;]*var\(--font-head\)", css)


def test_corners_are_square_and_nothing_is_shadowed():
    css = read("styles.css") + read("brand.css")
    assert not re.search(r"border-radius:\s*(?!0)[0-9]", css)
    assert "--shadow" not in css and "backdrop-filter" not in css and "text-shadow" not in css
    # the only box-shadows are the focus ring and the selected-state underline, never an offset drop shadow
    for decl in re.findall(r"box-shadow:\s*([^;]+);", css):
        assert decl.strip() in ("var(--ring)", "none"), decl


def test_one_gradient_only_and_it_is_the_duotone_plate():
    css = read("styles.css") + read("brand.css")
    grads = re.findall(r"[a-z-]*gradient\([^;]*", css)
    assert len(grads) == 1 and "--ab-eds-plate" in grads[0], grads


def test_hexes_are_only_the_briefs():
    """Hex colors live only in brand.css, and every one of them is in the brief; the other CSS/JS use tokens."""
    brief = {h.lower() for h in HEX.findall(BRIEF.read_text(encoding="utf-8"))}
    for name in ("styles.css", "common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        assert not HEX.search(read(name)), name
    used = {h.lower() for h in HEX.findall(read("brand.css"))}
    assert used and used <= brief, sorted(used - brief)
    for page in PAGES:
        assert {h.lower() for h in HEX.findall(read(f"{page}.html"))} <= brief


def test_pattern_assets_come_from_the_engine_and_are_used_once_each():
    for name, size in (("pattern-halftone.svg", 160), ("pattern-glyphgrid.svg", 160), ("hero-flow.svg", 1600)):
        svg = (SITE / "assets" / name).read_text(encoding="utf-8")
        assert f'viewBox="0 0 {size}' in svg
        assert {h.lower() for h in HEX.findall(svg)} <= {"#f7f2ee", "#261c1c", "#a11413", "#d14f34", "#4a3122"}
    assert len((SITE / "assets" / "hero-flow.svg").read_bytes()) < 400_000
    css = read("brand.css")
    for name in ("pattern-halftone.svg", "pattern-glyphgrid.svg", "hero-flow.svg"):
        assert f"assets/{name}" in css


@pytest.mark.parametrize("page", PAGES)
def test_at_most_one_textured_card_and_one_flow_hero_per_page(page):
    html = read(f"{page}.html")
    assert len(re.findall(r'class="[^"]*\btexcard\b', html)) <= 1
    assert len(re.findall(r'class="[^"]*\bflow-bg\b', html)) == 1


def test_index_cover_plate_and_contents_rows():
    html = read("index.html")
    assert html.count("b-plate") == 1 and "Fig. 1." in html and 'class="cover-n"' in html
    assert html.count('class="contents-row"') == 12
    for href in ("plays.html", "winprob.html", "fourth.html", "playcalling.html", "models.html"):
        assert f'class="contents-row" href="{href}"' in html
