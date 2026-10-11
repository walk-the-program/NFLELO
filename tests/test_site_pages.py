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
    """Exactly one solid primary action tile (the "01" tile) per page, and only the page markup may carry it."""
    html = read(f"{page}.html")
    assert html.count('class="b-primary"') == 1
    assert html.count("b-primary__n") == 1 and html.count("b-primary__go") == 1 and html.count("b-primary__label") == 1
    for name in ("common.js", f"{SCRIPT[page]}.js"):
        assert "b-primary" not in read(name)


def test_no_hex_colors_outside_the_brand_tokens():
    for name in ("styles.css", "common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", read(name)), name


# The final Bento Grid brief (context/brand.md, Walker 2026-10-10): every hex the site may contain, and nothing else.
BRIEF_HEXES = {
    "#7db7eb", "#ffffff", "#fbffa0", "#1c2126", "#f5f7f9", "#eaecee", "#64686c", "#cbced0", "#141414", "#a38400",
    "#dde6ee", "#edeff1", "#fbfbfc", "#fcfcfd", "#e9f1fa", "#f1f1f2", "#8dc0ed", "#6796c1",
}
RETIRED_HEXES = {
    "#20436b", "#eff8a5", "#ff4d37", "#26221c", "#f9f8f5", "#706d68", "#d9d8d4", "#dfe2e4", "#f7f8e8",
    "#f0c5c6", "#f0ec5e", "#aa810e", "#ebe6e8", "#f9f0f1",
}


def test_brand_css_defines_only_the_bento_brief_hexes():
    found = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", read("brand.css"))}
    assert found <= BRIEF_HEXES, found - BRIEF_HEXES
    assert not (found & RETIRED_HEXES)


def test_highlight_word_is_the_yellow_accent_with_an_ink_outline():
    css = read("brand.css")
    assert re.search(r"--on-primary:\s*#141414", css) and re.search(r"--on-secondary:\s*#141414", css) and re.search(r"--on-accent:\s*#141414", css)
    assert re.search(r"--accent:\s*#fbffa0", css) and re.search(r"--secondary:\s*#ffffff", css)
    assert re.search(r"--hl-fill:\s*var\(--accent\)", css) and re.search(r"--hl-stroke:\s*var\(--ink\)", css)
    assert re.search(r"\.hl \{[^}]*color: var\(--hl-fill\)[^}]*-webkit-text-stroke", css)
    assert "var(--accent-ink)" not in css + read("styles.css")   # Walker: the darker accent-ink is not used


def test_bento_materials_match_the_brief():
    css = read("brand.css")
    assert "--r-tile: 28px" in css and "--r-sub: 18px" in css and "--gap: 14px" in css
    assert "outline: 3px solid var(--primary)" in css and "0 0 0 4px var(--ink)" in css


def test_light_only_no_monospace_and_the_two_brand_fonts():
    for name in ("brand.css", "styles.css", "common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        text = read(name).lower()
        assert "monospace" not in text and "--font-mono" not in text and "ui-monospace" not in text, name
    css = read("brand.css")
    assert "Figtree" in css and "Nunito Sans" in css
    assert "prefers-color-scheme: dark" not in css + read("styles.css")


@pytest.mark.parametrize("page", PAGES)
def test_pages_use_only_brief_hexes_and_one_pattern_hero(page):
    html = read(f"{page}.html")
    assert {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", html)} <= BRIEF_HEXES
    heroes = re.findall(r'class="[^"]*hero-art (pat-[a-z]+)', html)
    assert heroes == ["pat-terrain"]                         # posterised terrain hero, one per page
    assert html.count("pat-pebbles") == 1                    # one pebble-field textured tile
    for gone in ("pat-aura", "pat-band", "pat-halftone"):
        assert gone not in html
    assert html.count('class="b-tag"') <= 1


def test_raster_pattern_assets_exist_at_the_brief_sizes():
    import struct
    for name, size in {"hero-terrain.png": (2560, 1440), "tex-pebbles.png": (1280, 720)}.items():
        head = (SITE / "assets" / name).read_bytes()[:24]
        assert head[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", head[16:24]) == size, name


def test_retired_brutalist_assets_are_gone_and_all_assets_are_used():
    for old in ("hero-flow.svg", "pattern-glyphgrid.svg", "pattern-halftone.svg", "hero-aura.png", "band-gradpanes.svg", "tile-halftone.svg"):
        assert not (SITE / "assets" / old).exists()
    css = read("brand.css")
    for asset in (SITE / "assets").iterdir():
        assert f"assets/{asset.name}" in css, f"{asset.name} is not used"


def test_secondary_is_never_a_line_dot_or_text_color():
    """Pale pink is about 1.1:1 on the surface: no stroke, text color or marker uses it. The brief's secondary button edge
    is allowed only in a rule that also carries an ink inset ring, so the control still reads."""
    for name in ("styles.css", "brand.css"):
        css = read(name)
        for sel, block in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            for decl in re.findall(r"(?:^|[;\s])((?:stroke|color)\s*:[^;}]*var\(--secondary\)[^;}]*)", block):
                assert False, (name, sel.strip(), decl)
            if re.search(r"border(?:-[a-z]+)?\s*:[^;}]*var\(--secondary\)", block):
                assert "inset 0 0 0 1px var(--ink)" in block, (name, sel.strip())
    for name in ("common.js", "app.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        for line in read(name).splitlines():
            if "var(--secondary)" in line:
                assert "fill:" in line and "pale: true" in line, (name, line.strip())
