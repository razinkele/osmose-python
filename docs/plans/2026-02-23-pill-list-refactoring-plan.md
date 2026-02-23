# Pill List Navigation Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the crowded horizontal `page_navbar` with a grouped left-side `navset_pill_list` layout using bslib components.

**Architecture:** The 10 top-level tabs move from a horizontal navbar into a vertical pill list with 4 section headers (Configure, Execute, Optimize, Manage). Each page's `*_ui()` function drops its redundant `page_fluid()` wrapper. A new `ui/styles.py` consolidates scattered inline style strings.

**Tech Stack:** Shiny for Python 1.5.1, shinyswatch 0.9.0 (superhero theme), bslib via `shiny.ui`

---

### Task 1: Strip `page_fluid` Wrappers from All Page UI Functions

Each of the 10 page modules currently wraps its content in `ui.page_fluid(...)`. Inside `navset_pill_list`, this is redundant — the pill list already provides a layout container. Each `*_ui()` must return its inner content directly.

**Files:**
- Modify: `ui/pages/setup.py` (line 25)
- Modify: `ui/pages/grid.py` (line 76)
- Modify: `ui/pages/forcing.py` (line 22)
- Modify: `ui/pages/fishing.py` (line 15)
- Modify: `ui/pages/movement.py` (line 15)
- Modify: `ui/pages/run.py` (line 32)
- Modify: `ui/pages/results.py` (line 93)
- Modify: `ui/pages/calibration.py` (line 143)
- Modify: `ui/pages/scenarios.py` (line 11)
- Modify: `ui/pages/advanced.py` (line 28)

**Step 1: Run baseline tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 235 passed

**Step 2: Modify pages that return a single `layout_columns`**

These 7 pages return `ui.page_fluid(ui.layout_columns(...))` — change each to return just `ui.layout_columns(...)`:

`ui/pages/setup.py` — change:
```python
def setup_ui():
    return ui.page_fluid(
        ui.layout_columns(
            ...
        ),
    )
```
to:
```python
def setup_ui():
    return ui.layout_columns(
        ...
    )
```

Apply the same pattern to:
- `ui/pages/grid.py` — `grid_ui()`: remove `page_fluid(`, remove trailing `)` and `,`
- `ui/pages/forcing.py` — `forcing_ui()`: same
- `ui/pages/fishing.py` — `fishing_ui()`: same
- `ui/pages/movement.py` — `movement_ui()`: same
- `ui/pages/run.py` — `run_ui()`: same
- `ui/pages/calibration.py` — `calibration_ui()`: same

**Step 3: Modify pages that return multiple top-level elements**

These 3 pages have multiple `layout_columns` or `layout_columns + card` inside `page_fluid`. Replace `page_fluid(...)` with `ui.div(...)`:

`ui/pages/results.py` — change:
```python
def results_ui():
    return ui.page_fluid(
        ui.layout_columns(...),
        ui.layout_columns(...),
    )
```
to:
```python
def results_ui():
    return ui.div(
        ui.layout_columns(...),
        ui.layout_columns(...),
    )
```

`ui/pages/scenarios.py` — change:
```python
def scenarios_ui():
    return ui.page_fluid(
        ui.layout_columns(...),
        ui.layout_columns(...),
    )
```
to:
```python
def scenarios_ui():
    return ui.div(
        ui.layout_columns(...),
        ui.layout_columns(...),
    )
```

`ui/pages/advanced.py` — change:
```python
def advanced_ui():
    return ui.page_fluid(
        ui.layout_columns(...),
        ui.card(...),
    )
```
to:
```python
def advanced_ui():
    return ui.div(
        ui.layout_columns(...),
        ui.card(...),
    )
```

**Step 4: Run tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 235 passed

**Step 5: Lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format osmose/ ui/ tests/`
Expected: Clean

**Step 6: Commit**

```bash
git add ui/pages/setup.py ui/pages/grid.py ui/pages/forcing.py ui/pages/fishing.py ui/pages/movement.py ui/pages/run.py ui/pages/results.py ui/pages/calibration.py ui/pages/scenarios.py ui/pages/advanced.py
git commit -m "refactor: remove redundant page_fluid wrappers from page UI functions"
```

---

### Task 2: Rewrite `app.py` to Use `navset_pill_list`

Replace `page_navbar` with `page_fillable` + `navset_pill_list`. Add grouped section headers. Add a custom app header bar.

**Files:**
- Modify: `app.py`

**Step 1: Rewrite `app_ui`**

Replace the entire `app_ui` definition (lines 18-31) with:

```python
app_ui = ui.page_fillable(
    # ── App header ──────────────────────────────────────────────
    ui.div(
        ui.h4("OSMOSE", ui.tags.small(" | Python Interface", style="color: #999;")),
        style="padding: 12px 20px; border-bottom: 1px solid #444;",
    ),
    # ── Left pill navigation with grouped sections ──────────────
    ui.navset_pill_list(
        # Configure
        "Configure",
        ui.nav_panel("Setup", setup_ui(), value="setup"),
        ui.nav_panel("Grid & Maps", grid_ui(), value="grid"),
        ui.nav_panel("Forcing", forcing_ui(), value="forcing"),
        ui.nav_panel("Fishing", fishing_ui(), value="fishing"),
        ui.nav_panel("Movement", movement_ui(), value="movement"),
        # Execute
        "Execute",
        ui.nav_panel("Run", run_ui(), value="run"),
        ui.nav_panel("Results", results_ui(), value="results"),
        # Optimize
        "Optimize",
        ui.nav_panel("Calibration", calibration_ui(), value="calibration"),
        # Manage
        "Manage",
        ui.nav_panel("Scenarios", scenarios_ui(), value="scenarios"),
        ui.nav_panel("Advanced", advanced_ui(), value="advanced"),
        id="main_nav",
        selected="setup",
        widths=(2, 10),
        well=False,
    ),
    theme=THEME,
)
```

Key details:
- `page_fillable` makes the app use full viewport height
- String arguments to `navset_pill_list` render as section headers (gray text dividers)
- `widths=(2, 10)` gives the pill list 2/12 of the width (narrow sidebar)
- `well=False` removes the default gray background behind pills (cleaner with dark theme)
- `id="main_nav"` enables programmatic tab switching via `ui.update_navset("main_nav", selected="run")`
- `value=` on each `nav_panel` provides stable identifiers

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 235 passed

**Step 3: Verify app can be imported and run**

Run: `.venv/bin/python -c "from app import app; print(type(app))"`
Expected: `<class 'shiny.App'>`

**Step 4: Lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format osmose/ ui/ tests/`
Expected: Clean

**Step 5: Commit**

```bash
git add app.py
git commit -m "feat: switch navigation to left-side pill list with grouped sections"
```

---

### Task 3: Extract Inline Styles to `ui/styles.py`

Scattered inline style strings across 10+ files make theming fragile. Extract them into named constants in a single module.

**Files:**
- Create: `ui/styles.py`
- Modify: `ui/pages/run.py` (console styles)
- Modify: `ui/pages/advanced.py` (table, diff colors)
- Modify: `ui/pages/scenarios.py` (diff highlight)
- Modify: `ui/components/param_form.py` (constraint hint)

**Step 1: Create `ui/styles.py`**

```python
"""Reusable style constants for the OSMOSE UI."""

# Text colors
COLOR_MUTED = "color: #999;"
COLOR_SUCCESS = "color: #2ecc71;"
COLOR_DANGER = "color: #e74c3c;"

# Console terminal
STYLE_CONSOLE = (
    "background: #111; color: #0f0; height: 500px; overflow-y: auto; "
    "padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; "
    "font-size: 13px; white-space: pre-wrap;"
)

# Monospace key display
STYLE_MONO_KEY = "font-family: monospace; font-size: 12px;"

# Constraint hint text below inputs
STYLE_HINT = "color: #888; font-size: 11px; margin-top: -8px;"

# Empty state placeholder
STYLE_EMPTY = "padding: 20px; text-align: center; color: #999;"

# Scrollable table container
STYLE_SCROLL_TABLE = "max-height: 600px; overflow-y: auto;"

# Diff row highlight (orange tint)
STYLE_DIFF_ROW = "background: rgba(255, 165, 0, 0.15);"
```

**Step 2: Replace inline styles in `ui/pages/run.py`**

Change (line 87-89):
```python
style="background: #111; color: #0f0; height: 500px; overflow-y: auto; "
"padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; "
"font-size: 13px; white-space: pre-wrap;",
```
to:
```python
from ui.styles import STYLE_CONSOLE
...
style=STYLE_CONSOLE,
```

**Step 3: Replace inline styles in `ui/components/param_form.py`**

Change (lines 61, 77):
```python
ui.tags.small(hint, style="color: #888; font-size: 11px; margin-top: -8px;"),
```
to:
```python
from ui.styles import STYLE_HINT
...
ui.tags.small(hint, style=STYLE_HINT),
```

**Step 4: Replace inline styles in `ui/pages/advanced.py`**

Replace occurrences:
- `style="font-family: monospace; font-size: 12px;"` → `STYLE_MONO_KEY`
- `style="color: #e74c3c;"...` → `COLOR_DANGER`
- `style="color: #2ecc71;"` → `COLOR_SUCCESS`
- `style="color: #999;"` → `COLOR_MUTED`
- `style="max-height: 200px; overflow-y: auto;"` — leave as-is (unique to this context)
- `style="max-height: 600px; overflow-y: auto;"` → `STYLE_SCROLL_TABLE`

Add import at top:
```python
from ui.styles import COLOR_DANGER, COLOR_MUTED, COLOR_SUCCESS, STYLE_MONO_KEY, STYLE_SCROLL_TABLE
```

**Step 5: Replace inline styles in `ui/pages/scenarios.py`**

Change (line 186):
```python
style="background: rgba(255, 165, 0, 0.15);",
```
to:
```python
from ui.styles import STYLE_DIFF_ROW, STYLE_EMPTY
...
style=STYLE_DIFF_ROW,
```

Also replace the `"padding: 20px; text-align: center; color: #999;"` occurrences with `STYLE_EMPTY`.

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 235 passed

**Step 7: Lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format osmose/ ui/ tests/`
Expected: Clean

**Step 8: Commit**

```bash
git add ui/styles.py ui/pages/run.py ui/pages/advanced.py ui/pages/scenarios.py ui/components/param_form.py
git commit -m "refactor: extract inline styles to ui/styles.py constants"
```

---

### Task 4: Add Test for App Structure

Add a test that verifies the app imports correctly and the pill list navigation structure is intact.

**Files:**
- Create: `tests/test_app_structure.py`

**Step 1: Write the test**

```python
"""Test app module structure and navigation layout."""

from shiny import App


def test_app_imports():
    """App module can be imported without error."""
    from app import app
    assert isinstance(app, App)


def test_app_ui_is_page_fillable():
    """Top-level UI uses page_fillable (not page_navbar)."""
    from app import app_ui
    # page_fillable returns a Tag; check it renders without error
    html = str(app_ui)
    assert "nav-pills" in html or "pill" in html.lower()


def test_nav_sections_present():
    """All 10 nav panels are present in the rendered HTML."""
    from app import app_ui
    html = str(app_ui)
    expected_labels = [
        "Setup", "Grid", "Forcing", "Fishing", "Movement",
        "Run", "Results", "Calibration", "Scenarios", "Advanced",
    ]
    for label in expected_labels:
        assert label in html, f"Missing nav panel: {label}"


def test_section_headers_present():
    """Grouped section headers appear in the navigation."""
    from app import app_ui
    html = str(app_ui)
    for header in ["Configure", "Execute", "Optimize", "Manage"]:
        assert header in html, f"Missing section header: {header}"
```

**Step 2: Run the new test**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py -v`
Expected: 4 passed

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 239 passed

**Step 4: Commit**

```bash
git add tests/test_app_structure.py
git commit -m "test: add app structure tests for pill list navigation"
```

---

### Task 5: Full Verification and Push

Run complete test suite, lint, format, verify app runs, and push to master.

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: 239 passed, 0 failures

**Step 2: Run lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: Clean

**Step 3: Verify app instantiation**

Run: `.venv/bin/python -c "from app import app; print('OK:', type(app))"`
Expected: `OK: <class 'shiny.App'>`

**Step 4: Push all commits**

```bash
git push origin master
```

**Step 5: Verify CI passes**

Run: `gh run list --limit 1`
Expected: Green check
