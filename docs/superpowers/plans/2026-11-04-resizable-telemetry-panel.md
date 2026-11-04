# Resizable Telemetry Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The right-hand telemetry panel can be resized (docked: left-edge handle; floating: every edge and corner) and its contents scaled independently of its size.

**Architecture:** All in `backend/app/static/index.html`. One small module `PanelSize` (exposed as `window.PanelSize`) owns the user scale and the docked width, both persisted in localStorage. `applyPanelScale()` stops deriving the scale from the floating width and uses `PanelSize.scale()` in both modes. New DOM: a Size row under the tabs, a docked splitter, seven floating edge grips next to the existing bottom-left grip.

**Tech Stack:** vanilla JS/CSS in the single-file UI; Playwright browser tests (`backend/tests/ui/`, existing fixtures).

**Spec:** `docs/superpowers/specs/2026-11-03-keep-laps-and-lap-comparison-design.md` §6 (this plan implements only §6; §4 and §5 get their own plans). Visual reference: branch `spike/compare-mockup` (worktree `..\lmu-coach-compare-mockup`), search for `PanelSize` and `floatEdges` there. Reference only, do not merge or copy it wholesale.

## Global Constraints

- PowerShell: chain with `;`, never `&&`. Never modify `.env`, `lmu-coach.config.json`, `data/sessions/`.
- UI changes stay inside `backend/app/static/index.html`. No new libraries. Canvas colour literals stay in sync with `:root` tokens (DESIGN.md). WCAG AA, keyboard operable, sentence case.
- Existing panel behaviour that this plan does not change stays: float/dock buttons, title-bar drag, the bottom-left grip, the keyboard nudge on the title bar and grip, the Balance/Coach/Record tabs.
- Scale range 50-200 %: buttons step 10 %, Ctrl+scroll step 5 %, stored value rounded to 0.05. Default 100 %.
- Docked width: minimum 300 px; maximum `min(1200, window.innerWidth - 520)` (the map keeps at least ~520 px); default = the CSS `--panel-w` (no inline width). Floating frame minimum 260 × 260 px.
- localStorage keys: `lmu-coach.panel.w` (docked width, px, integer) and `lmu-coach.panel.scale` (number). Missing, non-numeric or out-of-range values fall back to the defaults.
- At the `max-width:820px` breakpoint (stacked layout, panel `width:auto`) the splitter is hidden and no inline width is applied.
- Any module the existing code reaches through `window.X` must be assigned to `window` explicitly (a top-level `const` is not a window property; the mockup shipped a dead A-/A+ because of this).
- Comment the why, in the file's existing style (`PREVIOUSLY … Fix:` for changed behaviour).
- Commit: `git -c user.name=hoang -c user.email=hoangnhatvn2002@gmail.com commit -F <msgfile>`, message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- UI tests (repo root): `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic --with "uvicorn[standard]" --with playwright python -m pytest backend/tests/ui -v` (first run: `uv run --no-project --with playwright python -m playwright install chromium`). Backend suite unchanged: same command without the last two `--with` on `backend/tests`.

## Review Focus

1. Window made smaller after the docked panel was widened: the stored width is clamped to the new maximum on load and on resize, so the map never goes below ~520 px. (Task 2 test `test_docked_width_clamped_to_window`)
2. Narrow window (stacked layout, ≤ 820 px): no splitter, no inline width, the stored width is ignored, not overwritten. (Task 2 test `test_stacked_layout_ignores_width`)
3. Garbage in localStorage (`"abc"`, `"7"`, `"-1"`): defaults, no errors. (Task 1 test `test_bad_stored_values_fall_back`)
4. Undock, resize the floating frame, dock again: the docked width the user chose comes back; the floating size does not leak into it. (Task 3 test `test_redock_restores_docked_width`)
5. After a scale or size change the traces are redrawn at the new size (not stretched bitmaps), also mid-playback. (Task 1 test `test_scale_redraws_traces`, Task 3 test `test_floating_resize_redraws`)

---

### Task 1: Scale separate from size; the Size row

**Files:**
- Modify: `backend/app/static/index.html` (CSS block near `.panel-tabs`; markup right after `.panel-tabs`; `applyPanelScale()` ~l.1055; new `PanelSize` module after the panel drag/resize IIFE)
- Create: `backend/tests/ui/test_panel_size.py`

**Interfaces:**
- Produces: `window.PanelSize = { scale(): number, setScale(v: number): void, step(dir: -1|1): void, reset(): void }` (Task 2 adds `setWidth`, `restoreWidth`). Markup: `.panel-sizebar` containing buttons with `aria-label` "Smaller text and graphs", "Larger text and graphs", "Reset panel size", and `#panel-scale-val` showing e.g. `120%`. `--panel-scale` on `#right-panel` equals `PanelSize.scale()` in both modes.

- [ ] **Step 1: Write the failing tests** in `backend/tests/ui/test_panel_size.py` (copy `_open_session` from `test_chase_view.py`; use `HILLY`):

```python
def _scale(page):
    return float(page.evaluate("getComputedStyle(document.getElementById('right-panel')).getPropertyValue('--panel-scale')") or 1)

def test_size_buttons_scale_contents(page, server):
    _open_session(page, server, HILLY)
    page.get_by_role("button", name="Larger text and graphs").click()
    page.get_by_role("button", name="Larger text and graphs").click()
    assert abs(_scale(page) - 1.2) < 1e-6
    assert page.inner_text("#panel-scale-val") == "120%"
    page.get_by_role("button", name="Reset panel size").click()
    assert abs(_scale(page) - 1.0) < 1e-6
    assert page.errors == []

def test_scale_limits(page, server):
    _open_session(page, server, HILLY)
    for _ in range(20): page.get_by_role("button", name="Smaller text and graphs").click()
    assert abs(_scale(page) - 0.5) < 1e-6
    for _ in range(30): page.get_by_role("button", name="Larger text and graphs").click()
    assert abs(_scale(page) - 2.0) < 1e-6

def test_ctrl_wheel_scales(page, server):
    _open_session(page, server, HILLY)
    page.hover("#body-balance")
    page.keyboard.down("Control"); page.mouse.wheel(0, -100); page.mouse.wheel(0, -100); page.keyboard.up("Control")
    assert abs(_scale(page) - 1.1) < 1e-6

def test_scale_persists(page, server):
    _open_session(page, server, HILLY)
    page.get_by_role("button", name="Larger text and graphs").click()
    page.reload(); page.wait_for_function("() => window.PanelSize")
    assert abs(_scale(page) - 1.1) < 1e-6

def test_bad_stored_values_fall_back(page, server):
    page.goto(server + "/")
    for v in ["abc", "7", "-1"]:
        page.evaluate(f"localStorage.setItem('lmu-coach.panel.scale', {v!r}); localStorage.setItem('lmu-coach.panel.w', {v!r})")
        page.reload(); page.wait_for_function("() => window.PanelSize")
        assert abs(_scale(page) - 1.0) < 1e-6
        assert page.evaluate("document.getElementById('right-panel').style.width") == ""
    assert page.errors == []

def test_scale_redraws_traces(page, server):
    _open_session(page, server, HILLY)
    w0 = page.evaluate("document.getElementById('spd-canvas').width")
    page.get_by_role("button", name="Larger text and graphs").click()
    w1 = page.evaluate("document.getElementById('spd-canvas').width")
    assert w1 != w0      # bitmap re-sized to the new on-screen width, not stretched
```

- [ ] **Step 2: Run** `... python -m pytest backend/tests/ui/test_panel_size.py -v` → FAIL (no Size row / `PanelSize`).

- [ ] **Step 3: Implement.** Markup `.panel-sizebar` (one row: "Size", A-, value, A+, Reset; a right-aligned hint "drag the left edge to resize", hidden while floating). `PanelSize` module per the interface, reading/validating `lmu-coach.panel.scale` on load; `setScale` clamps, rounds to 0.05, stores, updates `#panel-scale-val`, calls `applyPanelScale()`. Ctrl+wheel listener on `#right-panel` (`passive:false`, only when `e.ctrlKey`; plain wheel keeps scrolling). In `applyPanelScale()` replace the width-derived floating scale with `PanelSize.scale()` (comment: PREVIOUSLY the floating scale followed `w / BASE_PANEL_WIDTH`, so a bigger window gave bigger text and never more room). Keep its redraw of `updateBalance()` + `updateTraceDots()`.

- [ ] **Step 4: Run** the same command → PASS; then the whole `backend/tests/ui` suite → all PASS (chase view tests unaffected).

- [ ] **Step 5: Commit** `index.html` + `test_panel_size.py`: "Telemetry panel: Size control, scale separate from size".

---

### Task 2: Docked width handle

**Files:**
- Modify: `backend/app/static/index.html` (splitter element inside `#right-panel`; CSS; `PanelSize` gains width; `dockPanel()` restores it)
- Modify: `backend/tests/ui/test_panel_size.py`

**Interfaces:**
- Consumes: `PanelSize` from Task 1.
- Produces: `PanelSize.setWidth(px: number, persist = true): void` (no-op while floating or in the stacked layout), `PanelSize.restoreWidth(): void`. `PanelSize.reset()` now also clears the docked width (inline style and `lmu-coach.panel.w`); add `assert _panel_w(page) == w0` after a Reset click to `test_docked_drag_resizes_and_persists`. Markup `#panel-splitter` with `role="separator"`, `aria-orientation="vertical"`, `aria-valuemin="300"`, `aria-valuemax` (current max), `aria-valuenow`, `tabindex="0"`, label "Telemetry panel width".

- [ ] **Step 1: Write the failing tests:**

```python
def _panel_w(page):
    return page.evaluate("Math.round(document.getElementById('right-panel').getBoundingClientRect().width)")

def _drag(page, sel, dx, dy=0):
    b = page.locator(sel).bounding_box(); x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    page.mouse.move(x, y); page.mouse.down(); page.mouse.move(x + dx, y + dy, steps=8); page.mouse.up()

def test_docked_drag_resizes_and_persists(page, server):
    _open_session(page, server, HILLY)
    w0 = _panel_w(page); map0 = page.evaluate("document.getElementById('map-canvas').getBoundingClientRect().width")
    _drag(page, "#panel-splitter", -200)
    assert abs(_panel_w(page) - (w0 + 200)) <= 2
    assert page.evaluate("document.getElementById('map-canvas').getBoundingClientRect().width") < map0 - 150
    page.reload(); page.wait_for_function("() => window.PanelSize")
    assert abs(_panel_w(page) - (w0 + 200)) <= 2

def test_docked_width_limits(page, server):
    _open_session(page, server, HILLY)
    _drag(page, "#panel-splitter", 2000);  assert _panel_w(page) == 300
    _drag(page, "#panel-splitter", -3000); assert _panel_w(page) == min(1200, 1600 - 520)

def test_docked_keyboard_and_reset(page, server):
    _open_session(page, server, HILLY)
    w0 = _panel_w(page)
    page.focus("#panel-splitter"); page.keyboard.press("ArrowLeft"); page.keyboard.press("Shift+ArrowLeft")
    assert _panel_w(page) == w0 + 80
    assert int(page.get_attribute("#panel-splitter", "aria-valuenow")) == w0 + 80
    page.dblclick("#panel-splitter")
    assert _panel_w(page) == w0 and page.evaluate("localStorage.getItem('lmu-coach.panel.w')") is None

def test_docked_width_clamped_to_window(page, server):
    _open_session(page, server, HILLY)
    _drag(page, "#panel-splitter", -3000)            # 1080 at 1600 wide
    page.set_viewport_size({"width": 1200, "height": 1000})
    page.wait_for_timeout(100)
    assert _panel_w(page) <= 1200 - 520

def test_stacked_layout_ignores_width(page, server):
    page.goto(server + "/"); page.evaluate("localStorage.setItem('lmu-coach.panel.w','700')")
    page.set_viewport_size({"width": 800, "height": 1000}); page.reload(); page.wait_for_function("() => window.PanelSize")
    assert page.evaluate("document.getElementById('right-panel').style.width") == ""
    assert not page.is_visible("#panel-splitter")
    assert page.evaluate("localStorage.getItem('lmu-coach.panel.w')") == "700"
```

- [ ] **Step 2: Run** → FAIL (no `#panel-splitter`).

- [ ] **Step 3: Implement.** `#right-panel` gets `position:relative` (floating keeps `position:fixed`). Splitter: absolutely positioned over the left edge (~9 px wide, `cursor:col-resize`, accent line on hover/focus/drag), hidden while floating and at ≤ 820 px. Pointer drag with `setPointerCapture`; width persisted on pointer-up (not every move). Arrow keys 20 px, Shift 60 px; double-click clears the inline width and the stored key. `restoreWidth()` on load (after validation), on `window.resize` re-clamp the current inline width without changing the stored value, and in `dockPanel()` after it clears the floating inline styles. The stacked-layout check uses `matchMedia('(max-width:820px)')`. The map resizes through its existing `ResizeObserver`.

- [ ] **Step 4: Run** the file → PASS; whole UI suite → PASS.

- [ ] **Step 5: Commit**: "Telemetry panel: docked width handle".

---

### Task 3: Floating frame resizable from every edge and corner; docs

**Files:**
- Modify: `backend/app/static/index.html` (seven grips `.pf-edge.pf-n/s/e/w/ne/nw/se` appended to `#right-panel`; CSS; pointer handlers)
- Modify: `backend/tests/ui/test_panel_size.py`
- Modify: `DevNotes.md` (new `### 11/4/2026` heading, newest last), `docs/HANDOFF.md` (§9 `index.html` line: mention the panel sizing)

**Interfaces:**
- Consumes: `PanelSize.scale()`, `applyPanelScale()`, `panelState.floating`.
- Produces: nothing new for other plans.

- [ ] **Step 1: Write the failing tests:**

```python
def _rect(page):
    return page.evaluate("(()=>{const r=document.getElementById('right-panel').getBoundingClientRect();"
                         "return [Math.round(r.left),Math.round(r.top),Math.round(r.width),Math.round(r.height)]})()")

def test_floating_edges_resize_without_scaling(page, server):
    _open_session(page, server, HILLY)
    page.click("#panel-float-btn")
    l, t, w, h = _rect(page)
    _drag(page, ".pf-w", -150); assert _rect(page)[0] == l - 150 and _rect(page)[2] == w + 150
    _drag(page, ".pf-e", 40);   assert _rect(page)[2] == w + 190
    _drag(page, ".pf-n", 0, 30); r = _rect(page); assert r[1] == t + 30 and r[3] == h - 30
    _drag(page, ".pf-s", 0, -20); assert _rect(page)[3] == h - 50
    _drag(page, ".pf-nw", 20, 20); r = _rect(page); assert r[0] == l - 130 and r[1] == t + 50
    assert abs(_scale(page) - 1.0) < 1e-6           # size never changes the scale
    assert page.errors == []

def test_floating_min_size(page, server):
    _open_session(page, server, HILLY)
    page.click("#panel-float-btn")
    _drag(page, ".pf-se", -2000, -2000)
    r = _rect(page); assert r[2] == 260 and r[3] == 260

def test_floating_scale_is_independent(page, server):
    _open_session(page, server, HILLY)
    page.click("#panel-float-btn")
    page.get_by_role("button", name="Larger text and graphs").click()
    _drag(page, ".pf-e", 200)
    assert abs(_scale(page) - 1.1) < 1e-6

def test_redock_restores_docked_width(page, server):
    _open_session(page, server, HILLY)
    _drag(page, "#panel-splitter", -100); docked = _panel_w(page)
    page.click("#panel-float-btn"); _drag(page, ".pf-w", -300)
    page.click("#panel-dock-btn")
    assert _panel_w(page) == docked

def test_floating_resize_redraws(page, server):
    _open_session(page, server, HILLY)
    page.click("#panel-float-btn")
    w0 = page.evaluate("document.getElementById('spd-canvas').width")
    _drag(page, ".pf-w", -200)
    assert page.evaluate("document.getElementById('spd-canvas').width") > w0
```

- [ ] **Step 2: Run** → FAIL (no `.pf-*`).

- [ ] **Step 3: Implement.** Grips visible only while floating (thin bars on edges, small squares on corners; the existing bottom-left grip stays as the `sw` corner; the `s`/`w` bars leave room for it). Each grip moves only the edges in its name; opposite edges stay put; clamp to the minimum size and to the window (a grip can never push the frame off-screen). Redraw (`applyPanelScale()`) throttled to one per animation frame during the drag and once on pointer-up. Keyboard: the existing Shift+arrow nudge on the title bar/grip is the keyboard path; leave it.

- [ ] **Step 4: Run** the file → PASS; whole UI suite → PASS; backend suite → PASS (unchanged).

- [ ] **Step 5: Manual look.** Start the backend on the real recordings exactly as the mockup did (`uv run --no-project --with fastapi --with uvicorn --with duckdb --with numpy --with scipy --with pydantic --with httpx python backend\tests\ui\serve_for_ui.py C:\Users\hoang\Projects\lmu-coach\data\sessions <port> C:\Users\hoang\Projects\lmu-coach\data\track_reference`), screenshot docked at 300 px / wide / 150 %, and floating resized at 80 %; check nothing overlaps or clips (tabs, Size row, traces, Coach follow-up box). Fix anything that does.

- [ ] **Step 6: DevNotes + HANDOFF**, then **commit**: "Telemetry panel: floating frame resizable from every edge; notes".
