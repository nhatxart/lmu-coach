# LMU Coach — design reference

**Direction:** "pit-wall console". Data-dense dashboard, dark, calm chrome so the telemetry colours do the talking.
**Type:** Fira Sans (interface) + Fira Code (every number, tabular). Bundled in `backend/app/static/fonts/` (SIL OFL), so it works offline.
**Icons:** Lucide (ISC), inlined as an SVG sprite at the top of `index.html`; no emoji or glyph icons.

## Layout
`header` (brand · backend status · ⚙ Settings) → three columns: **Sessions** (filter, grouped, two-line cards) · **Stage** (overlay select + legend, map, driver-input HUD, traction circle + tyre gauges, zoom controls, playback + lap chips) · **Telemetry panel** (Balance / Coach / Record; can be undocked, moved and resized, by mouse or keyboard).

## Tokens (all in `:root` of `index.html`)
| Group | Tokens |
|---|---|
| Surfaces | `--bg #070B14` · `--surface-1 #0D1322` · `--surface-2 #131B2F` · `--surface-3 #1B2540` · `--surface-4 #243152` · `--line #222D48` |
| Text | `--text #E8ECF6` · `--text-2 #B4BED3` · `--text-3 #8E9AB5` (all ≥ 4.5:1 on every surface) |
| Brand / state | `--accent #2EE6A0` · `--ring #7DD3FC` (focus) · `--danger #FF6B6B` · `--warn #FBBF24` |
| Data series | speed `#38BDF8` · brake `#FF6B6B` · throttle `#2DD4A7` · steering `#94A3B8` · ABS `#C4A1FF` · TC `#FBBF24` · yaw `#FB923C` · understeer `#60A5FA` · oversteer `#FB923C` (all ≥ 3:1) |
| Scale | 4px spacing grid · radii 8/10/14/18 · type 13/17/20/24/31px (the larger scale you asked for; nothing below 12px) |
| Motion | 120/200/320ms, `cubic-bezier(.2,.8,.2,1)`; everything collapses under `prefers-reduced-motion` |

Brake = coral and throttle = teal on purpose: they stay distinguishable for common colour-vision deficiencies where red/green would not. Every series is also labelled (heading, legend or in-chart label), so colour is never the only cue.

## Accessibility (verified)
* axe-core: **0 violations** on the loaded lap, Coach, Record, Settings-open and undocked states.
* Landmarks + one `h1`, skip link, ARIA tabs with arrow/Home/End keys, visible 2px focus ring, 44px touch targets on the main controls.
* No mouse needed: the floating panel moves with arrow keys and resizes with Shift+arrows; zoom has +/− buttons; sessions can be filtered and opened from the keyboard.

## Responsive
≥1500px full layout (HUD 1.5×) · ≤1500 · ≤1280 · ≤1040 progressively narrower columns and smaller overlays · ≤820 stacks into one scrolling column. Stat tiles adapt to the *panel's* width (container query).

## Changing things
* Colours / spacing / type: edit `:root` — **except the canvases.** Map, HUD, traces and gauges are drawn in JavaScript with colour literals that match the tokens above (search `rgba(` / `rgb(` in the script). If you change a series colour, change it in both places.
* Canvas text uses `fontData()` (Fira Code at 1.3× the original sizes).
* Fonts: `@font-face` rules at the top of the stylesheet. Licenses: `backend/app/static/fonts/LICENSES.txt`.
