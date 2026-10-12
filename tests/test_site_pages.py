"""Static checks on the site's HTML pages: the shared nav, local asset links, and that every element id a page
script looks up exists in its HTML. The behavior is checked in a browser (see context/elo-site.md)."""
import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent / "site"
PAGES = ["index", "elo", "models", "plays", "winprob", "fourth", "playcalling"]
NAV_LINKS = ["index.html", "elo.html", "plays.html", "winprob.html", "fourth.html", "playcalling.html", "models.html"]   # Home, Elo, then the model pages, in this order
SCRIPT = {"index": "app", "elo": "elo", "models": "models", "plays": "plays", "winprob": "winprob", "fourth": "fourth", "playcalling": "playcalling"}


def read(name: str) -> str:
    return (SITE / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_has_the_model_nav_links(page):
    html = read(f"{page}.html")
    nav = re.search(r"<ul class=\"nav-links\">(.*?)</ul>", html, re.S).group(1)
    hrefs = re.findall(r'<a href="([a-z]+\.html)"', nav)
    assert hrefs == NAV_LINKS, f"{page}.html nav links are {hrefs}"


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
    for name in ("styles.css", "common.js", "elo-lib.js", "app.js", "elo.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
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
    for name in ("brand.css", "styles.css", "common.js", "elo-lib.js", "app.js", "elo.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
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
    for gone in ("pat-aura", "pat-band", "pat-halftone", "pat-pebbles", "band-wrap", "One tile, one idea"):
        assert gone not in html
    assert html.count('class="b-tag"') <= 1


def test_raster_pattern_assets_exist_at_the_brief_sizes():
    import struct
    for name, size in {"hero-terrain.png": (2560, 1440)}.items():
        head = (SITE / "assets" / name).read_bytes()[:24]
        assert head[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", head[16:24]) == size, name


def test_retired_brutalist_assets_are_gone_and_all_assets_are_used():
    for old in ("hero-flow.svg", "pattern-glyphgrid.svg", "pattern-halftone.svg", "hero-aura.png", "band-gradpanes.svg", "tile-halftone.svg", "tex-pebbles.png"):
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
    for name in ("common.js", "elo-lib.js", "app.js", "elo.js", "models.js", "plays.js", "winprob.js", "fourth.js", "playcalling.js"):
        for line in read(name).splitlines():
            if "var(--secondary)" in line:
                assert "fill:" in line and "pale: true" in line, (name, line.strip())


# ---- Walker 2026-10-11: no inline sources or receipts; the data credit appears once, in the footer ----

SA_PAGES = ["models", "plays", "fourth", "playcalling"]   # pages whose numbers come from participation data


@pytest.mark.parametrize("page", PAGES)
def test_footer_credit_appears_once_per_page(page):
    html = read(f"{page}.html")
    assert html.count('id="credit"') == 1
    foot = re.search(r"<footer.*?</footer>", html, re.S).group(0)
    credit = re.search(r'<p id="credit">(.*?)</p>', foot, re.S).group(1)
    assert "nflverse" in credit and "CC BY" in credit
    if page in SA_PAGES:
        assert "CC BY-SA 4.0" in credit
    if page in ("index", "models"):
        assert "FiveThirtyEight" in credit
    # nothing else on the page restates the credit (no second credit element, no per-section credit lines)
    assert "how-credit" not in html and "foot-credit" not in html
    assert html.count("Data: nflverse") == 1


@pytest.mark.parametrize("page", PAGES)
def test_no_source_or_receipt_clutter_renders(page):
    text = read(f"{page}.html") + read(f"{SCRIPT[page]}.js") + read("common.js")
    for gone in ("experiments/runs", "/commit/", "Rules ' +", "Result ' +", "Rules commit", "Result commit", "how-rcpt", "receiptLinks",
                 "Run registry", "run registry", "Receipts", "receipts", "Source: ", "Run record", "Data contract"):
        assert gone not in text, (page, gone)


def test_models_page_keeps_one_method_link_per_card_and_one_how_we_test_tile():
    html, js = read("models.html"), read("models.js")
    assert html.count('id="how"') == 1 and html.count("How we test") >= 2   # nav link and the section
    assert "github.com/walk-the-program/NFLELO" in html
    sec = re.search(r'<section class="sec" id="how".*?</section>', html, re.S).group(0)
    assert len(re.findall(r"<a ", sec)) == 1 and sec.count('class="deck"') == 2
    assert "text: 'Method'" in js and js.count("class: 'mc-links'") == 1


def test_models_scoreboard_has_a_summary_tile_that_fills_the_grid():
    js, css = read("models.js"), read("styles.css")
    assert "summaryCard(c)" in js and "mc-sum" in js
    assert "development only" in js and "running live" in js
    # 12 tiles divide evenly into the 2, 3 and 4 column layouts
    cols = [int(n) for n in re.findall(r"\.mcards \{ grid-template-columns: repeat\((\d+), 1fr\)", css)]
    assert cols == [2, 3, 4] and all(12 % n == 0 for n in cols)
    # tiles stacked in a loose body (the card grid, then the table-view tile) sit the standard 14px apart
    assert re.search(r"\.body\.loose \{[^}]*display: grid[^}]*gap: var\(--gap\)", css)


# ---- Walker 2026-10-11: cut Records, the heatmap and the team view; one name per model; honest units ----

def test_cut_sections_and_their_data_are_gone():
    index_html, pc_html = read("index.html"), read("playcalling.html")
    assert 'id="records"' not in index_html and "#records" not in index_html
    assert "heat" not in read("app.js").lower() and "tapestry" not in read("app.js")
    assert 'id="teams"' not in pc_html and "team_view" not in read("playcalling.js")
    assert not (SITE / "data" / "records.json").exists() and not (SITE / "data" / "tapestry.json").exists()
    assert "team_view_2025" not in (SITE / "data" / "playcalling.json").read_text()


@pytest.mark.parametrize("page", PAGES)
def test_section_eyebrows_have_no_gaps(page):
    nums = [int(n) for n in re.findall(r'<span class="kn">(\d+)</span>', read(f"{page}.html"))]
    assert nums == list(range(1, len(nums) + 1)), (page, nums)


@pytest.mark.parametrize("page", PAGES)
def test_no_internal_model_codes_in_visible_text(page):
    text = read(f"{page}.html") + read(f"{SCRIPT[page]}.js") + read("common.js")
    for code in ("A4s", "Model v1", "Elo v2", "C2d", "2025 model"):
        assert code not in text, (page, code)


def test_win_probability_gaps_read_pp_not_pts():
    js = read("app.js")
    assert "signed(gapPts, 1) + ' pp'" in js and "' pts', gap: true" not in js
    assert "Gap ' + f1(Math.abs(mg.gap)) + ' pp'" in js
    assert "win probability, %" in js and 'id="mx-axis"' in read("index.html")


def test_fourth_down_chart_uses_three_action_colors_and_a_hatch_for_leans():
    css, js, html = read("styles.css"), read("fourth.js"), read("fourth.html")
    assert re.search(r"--cat-go:\s*var\(--primary\)", css) and re.search(r"--cat-fg:\s*var\(--ink\)", css) and re.search(r"--cat-punt:\s*var\(--muted\)", css)
    assert "url(#hatch-go)" in css and "'hatch-' + k" in js and ".hm .hc.lv-0 { fill: var(--ab-sub); }" in css
    assert "situations are toss-ups" in js and "First model failed its calibration test" in js
    assert 'id="status"' not in html


def test_no_pts_label_on_a_win_probability_difference():
    """Win-probability differences are percentage points ("pp"); "pts" is for spreads and Elo only."""
    for name in ("fourth.js", "winprob.js", "plays.js"):
        js = read(name)
        assert "' pts'" not in js and " pts," not in js and "(points)" not in js, name
    assert "' pts', gap: true" not in read("app.js")


# ---- Walker 2026-10-11: the site is called NFELO; home is concise; the Elo page holds the Elo detail ----

def _without_repo_urls(text: str) -> str:
    return text.replace("walk-the-program/NFLELO", "")


def test_site_name_is_nfelo_everywhere_visible():
    """No visible "NFLELO" in the pages, scripts or published data; only the GitHub repo URL keeps the old spelling."""
    for f in sorted(SITE.glob("*.html")) + sorted(SITE.glob("*.js")) + sorted((SITE / "data").glob("*.json")):
        assert "NFLELO" not in _without_repo_urls(f.read_text(encoding="utf-8")), f.name
    for page in PAGES:
        html = read(f"{page}.html")
        assert re.search(r"<title>NFELO: ", html) and html.count('>NFELO</a>') == 1, page
    assert re.search(r'<h1 class="wordmark">NF<span class="hl">ELO</span></h1>', read("index.html"))


def test_elo_page_exists_with_the_moved_sections_and_a_method_tile():
    html = read("elo.html")
    for sec in ("explorer", "luck", "history", "scorecard", "method"):
        assert f'id="{sec}"' in html, sec
    assert 'id="method-body"' in html and 'class="method how"' in html
    js = read("elo.js")
    for fn in ("renderExplorer", "renderLuck", "renderHistory", "renderScorecard", "liveBlock", "renderMethod"):
        assert f"function {fn}" in js, fn
    assert "specList" in js and "'Expansion teams'" in js and "'Margin of victory'" in js


def test_elo_page_renders_only_existing_data_files_and_known_ids():
    js, html = read("elo.js"), read("elo.html")
    files = re.search(r"var FILES = \[(.*?)\]", js).group(1)
    for name in re.findall(r"'([a-z]+)'", files):
        assert (SITE / "data" / f"{name}.json").exists(), name
    ids = set(re.findall(r"\$\('#([a-z][\w-]*)'\)", js))
    for i in ids:
        assert f'id="{i}"' in html, f"elo.js looks up #{i}, which elo.html does not have"
    assert "explorer-body" in html and "scorecard-body" in html and "history-body" in html and "luck-body" in html


def test_home_keeps_only_the_concise_sections():
    html, js = read("index.html"), read("app.js")
    assert re.findall(r'<section class="sec" id="([a-z]+)"', html) == ["ladder", "week"]   # projected wins and odds are inserted by app.js
    for gone in ("explorer", "luck", "history", "scorecard"):
        assert f'id="{gone}' not in html, gone
        assert f"function render{gone.capitalize()}" not in js, gone
    assert "function renderRest" in js and "function renderOdds" in js
    for name in re.findall(r"'([a-z]+)'", re.search(r"var FILES = \[(.*?)\]", js).group(1)):
        assert (SITE / "data" / f"{name}.json").exists(), name
    assert "lineChart" not in js     # the chart code lives in elo-lib.js, shared with the Elo page
    assert "function lineChart" in read("elo-lib.js")


@pytest.mark.parametrize("page", ["plays", "winprob", "fourth", "playcalling"])
def test_model_pages_have_one_specification_tile(page):
    html, js = read(f"{page}.html"), read(f"{page}.js")
    assert "<span class=\"kt\">Specification</span>" in html and 'id="how-dyn"' in html
    assert js.count("N.specList(") == 1
    for label in ("'Model'", "'Inputs'", "'Training'", "'Test'", "'Metric'", "'Result'", "'Calibration'", "'License'"):
        assert label in js, (page, label)


def test_fourth_down_page_drops_the_failed_engine_audit():
    js = read("fourth.js")
    assert "holdout_audit_v1" not in js and "audit" not in js.lower()
    assert "wp_lost_by_choice" not in js


def test_models_cards_carry_a_one_line_technical_spec():
    import json
    d = json.loads((SITE / "data" / "research.json").read_text())
    assert all(m.get("spec") for m in d["models"])
    assert "mc-spec" in read("models.js")
