# Keep laps, lap comparison (ghosts) and a resizable telemetry panel - design

Date: 2026-11-03. Status: draft for review.
Mockup: branch `spike/compare-mockup` (worktree `..\lmu-coach-compare-mockup`),
a throwaway copy of `index.html` on real recordings. It is the visual reference
for this design, not code to merge.

## 1. Why

The driver wants to find out **how much time a different line or different
inputs cost or gain**, corner by corner, and see it: a ghost car to chase and
overlaid traces. Recordings are large (15-130 MB each) and a session holds many
laps nobody will look at again, so the driver also wants to **keep the laps
worth keeping and clear out the rest**, for disk space and for a shorter lap
list.

Three features, one release:

1. **Keep laps**: copy chosen laps out of a recording into small files of their
   own in a separate kept-laps folder; afterwards the app may offer to delete
   the original recording.
2. **Lap comparison**: ghost cars (top-down and chase view), a time-delta trace
   and overlaid input traces, across files, same track layout and same car
   class only.
3. **Resizable telemetry panel**: the right-hand panel can be resized and its
   contents scaled, docked and floating.

## 2. Goals, non-goals, anti-goals

Goals
- Keep one or more laps of any recording as stand-alone files that every part
  of the app treats like a recording (map, traces, chase view, Coach, ghosts).
- Clearing out `data/sessions` must be safe once the wanted laps are kept.
- Compare your lap against up to 4 other laps at once, from the same file or
  other files, kept laps included.
- See where time is gained or lost: a delta trace over lap distance, the line
  coloured by gain/loss, and the ghosts on the map and in the chase view.
- Compare inputs: speed, throttle, brake, steering and line position overlaid,
  one colour per lap, zoomable in sync.

Non-goals
- Cross-class comparison (a Hypercar is not a target for a GT3). Refused, not
  hidden behind a toggle.
- Cross-layout comparison (e.g. the two Fuji layouts).
- Sharing laps with other people, uploading anywhere, editing lap data.
- Asking the LLM coach about a comparison (possible later; not now).
- Keeping laps from LMU's native export files (see 4.6).

Anti-goals (would make it a failure even if it "works")
- **Anything is deleted before a verified copy exists.** The only delete path
  is the explicit offer after every kept lap passed its check, and it asks
  twice.
- **Kept laps silently count twice** in grip profiles or track topography (the
  same lap from the original and from its kept copy).
- **A misaligned comparison is shown as a real time delta**: ghosts from
  another layout, another class, or offset by a few metres between files.
- Rewriting what already works: the existing panel, map, chase view and
  traces keep their behaviour unless this document changes it.

## 3. Constraints (from the project)

- UI changes stay in `backend/app/static/index.html` (CLAUDE.md). Canvas colour
  literals stay in sync with the CSS tokens (`DESIGN.md`).
- The backend runs in Docker and mounts `data/sessions` and
  `data/reference_laps` **read-only** (`docker-compose.yml`). Writing and
  deleting files is done by the host agent (`recorder/recorder_control.py`,
  port 8420), which already owns host-side actions and has the GuardMiddleware
  (localhost Host header, allowed Origins).
- Never modify a recording in place. Kept files are new files; the original is
  only ever deleted whole, on explicit confirmation.
- Recorder files are the reference format (see HANDOFF §4.6): scalar tables
  `(value)` at the rate in `channelsList`, wheel tables `(value1..4)`, event
  tables `(ts, value)`, `Other Cars` with `ts`, plus `metadata`,
  `channelsList`, `eventsList`. Per-sample tables have no timestamp; the
  backend maps their rows evenly over the session.
- Backend Python 3.12 in Docker; recorder/host agent Python 3.13 with uv.
- Windows: a file the backend still has open cannot be deleted.

## 4. Keep laps

### 4.1 What the driver does
1. On the lap bar, **Keep laps…** switches the chips to selection mode
   (checkboxes); any number of laps can be ticked. The button becomes
   **Keep N laps** (or **Cancel** with none ticked).
2. A dialog lists, per lap, the file it will write and says the recording is
   not changed. Laps already kept are marked and skipped.
3. The host agent writes and checks each file. The dialog shows the result per
   lap (written, checked, lap time matches).
4. Only if every lap passed: **Delete the original recording?** with the file
   name, what is lost (the other laps, and their contribution to future
   grip-profile rebuilds), and **Keep the recording** as the default button.
   Delete asks once more, then the host agent deletes the file.
5. The lap list and session list refresh; kept laps appear under **Kept laps**
   in the session list (the current "Reference laps" group, renamed).

If the host agent is not running, **Keep laps…** is disabled with the same hint
the Record tab uses.

### 4.2 The kept file
- One file per kept lap, in the kept-laps folder: the host-side folder that is
  mounted as `/data/reference_laps` (`stack_paths.reference_laps_dir`, default
  `data/reference_laps`).
- Name (decided in the conversation, readable in Explorer):
  `<Track> - <Car> - <m>m<ss.sss> - L<n> - <YYYY-MM-DD>.duckdb`, made
  filesystem-safe. The session list parses it like a recording name.
- Same schema as a recording, so the backend reads it unchanged. Contents: the
  lap plus a margin of a few seconds before and after (filters run on the whole
  file before the lap is cut, and the lap needs its timing-line crossings).
- `metadata` is copied, plus: `KeptFromFile` (original file name), `KeptLap`
  (original lap number), `KeptLapTime` (official lap time), `KeptAt` (UTC).
  The file must show **one lap** in the lap list: the margin fragments before
  and after are not offered as laps (use the metadata; how is the builder's
  call).
- Writing goes through the recorder's existing safe-write pattern (temp file,
  atomic rename), never a half-written file under the final name.
- **Check** (required before anything else is offered): the new file opens, the
  backend reads exactly one lap from it, and its lap time equals the original's
  within 1 ms (the official lap time, not a re-derived one).
- Keeping the same lap twice does not create a second file (match on
  `KeptFromFile` + `KeptLap`).

### 4.3 Side effects that must be handled
- **Grip profiles.** Profiles are keyed per file (`tire_files["<file>"]`) and
  `/analyze` folds every analyzed lap into the profile. A kept lap is the same
  lap under a new file name: analyzing it, or a profile rebuild that reads both
  folders, would count it twice. Rule: a kept lap is attributed to its origin
  (`KeptFromFile`, `KeptLap`); if that origin lap is already in the profile it
  is not added again. After the original is deleted, the kept copy is the
  lap's only source and must still be counted.
- **A kept lap must be worth as much to the tyre model as it was inside the
  recording**, also after the original is deleted (that is the point of
  keeping it: "if I clean out the sessions folder it's fine"). The tyre model
  takes per-recording context from the WHOLE file, not from the lap:
  `tire_model.file_load_reference` (median axle load on low-speed straight
  samples, which flying laps do not have; on Fuji only the out-lap and pit
  lane do), the slow-lap cutoff (`fastest * TIRE_LAP_MAX_RATIO`) and the
  deflection-to-force calibration. A one-lap file cannot recompute these.
  Rule: the keep step computes that file-level context from the original and
  stores it in the kept file's metadata; ingest and `/analyze` use the stored
  values for kept files instead of recomputing them. Test: a lap ingested
  from its kept file gives the same `LapBins` as the same lap ingested from
  the original recording.
- **De-duplication scope**: the tyre-model data (`tire_files`, per lap) and
  the topography fit. The ABS/TC/slip baselines (`update_from_grip_report`)
  have no per-lap key and already count a lap again every time it is
  re-analyzed; that is existing behaviour, unchanged here (log it in
  DevNotes as a known issue).
- **Rebuild tools** (`backend/tools/ingest_sessions.py`,
  `build_track_reference.py`) read the kept-laps folder too, with the same
  de-duplication, so a cleanout does not lose what was learned from kept laps.
- **Topography** fit: same rule (no double weight for a lap present twice).
- **Delete on Windows** can fail while the backend has the file open. Report
  that plainly ("in use, try again"); never retry by force.

### 4.4a Amendment (11/9/2026, user decision): the backend does it, no host agent
"This should run on the backend with no LLM involvement" and "It should be
able to remove it after a deliberate delete button and a warning". So the
backend writes kept files and deletes originals itself; the host agent is not
involved. `docker-compose.yml` mounts `/data/reference_laps` and
`/data/sessions` read-write (comment why). The delete endpoint is the only
code that may remove anything from `/data/sessions`, only after the user
pressed a separate Delete button and confirmed a warning naming the file, and
only when every kept lap from that action passed its check. 4.4 below is
superseded where it names the agent; its checks and refusals still apply.

### 4.4 Host agent API (shape; names are the builder's call)
- `POST` keep: `{source, file, laps:[n...]}` -> per lap `{lap, kept_file, ok,
  lap_time_original, lap_time_kept, error}`. Paths are resolved against the
  configured folders only (no client-supplied paths, no `..`).
- `POST` delete: `{source:"sessions", file}`, only for files in the sessions
  folder, only when every lap the client kept from it in this action passed
  its check (the agent re-checks that the kept files exist).
- Both behind GuardMiddleware like the existing endpoints.
- **The check uses the backend**, so there is one lap reader (`telemetry.py`),
  not a second copy in the agent (whose environment has no numpy/scipy and a
  different Python). The agent writes the file, then asks the backend for the
  kept file's laps over HTTP. Consequences, all required:
  - Keeping needs both the host agent and the backend running; otherwise
    **Keep laps…** is disabled with the reason.
  - The agent must write into the folder Docker mounts as
    `/data/reference_laps`. The agent's `stack_paths.reference_laps_dir` is
    empty by default while Docker uses `REFERENCE_LAPS_PATH` from `.env`, so
    before the first keep the agent proves they are the same folder (e.g. it
    writes a uniquely named probe file and the backend must list it), and
    refuses with a clear message if not. Same proof for the sessions folder
    before any delete.
  - The extra metadata of 4.3 is computed by the backend (it owns the tyre
    model), returned to the agent and written into the kept file, or the
    backend exposes an endpoint the agent calls; the builder chooses, the
    rule is one implementation.

### 4.5 Also in scope: the "best lap" bug
The **BEST** chip and the lap auto-selected when a session opens are the
shortest lap time, which is often a partial lap (Fuji 10/19: lap 13, 36 s). Fix
it as part of this work: partial laps are not "best". Comparison's "best lap"
helpers depend on the same rule.

### 4.6 Native LMU export files
Keep laps applies to recorder files only. Native files have a different table
set; the Keep button is disabled for them with a short reason. Comparison
works with native files where their channels allow it.

## 5. Lap comparison

### 5.1 What the driver sees (as in the mockup)
- **Compare tab** in the right panel (fourth tab):
  - "Laps on screen": You (your lap, white) plus up to 4 ghosts, each with a
    colour, lap number, lap time, car model, date, kept/this-session tag, the
    live delta at the playhead (+ red = you are slower, - teal = faster) and a
    remove button.
  - Add a ghost: file picker (same track and class only; "This session"
    first) + lap picker, or the quick buttons **Best lap of this session** and
    **Best other file, same track**. The header states the filter, e.g.
    "LMGT3 only · 15 files on this track".
  - **Ghost position**: Same time (ghost where it was at your elapsed lap
    time; the gap is visible) / Same distance (ghost alongside you; line and
    angle differences visible). Applies to the map and the chase view.
- **Traces over lap distance**, one colour per lap: Time delta (one line per
  ghost), Ground speed, Throttle, Brake, Steering, Line offset to you (metres,
  + = ghost to your right). A dot per lap at the playhead.
- **Zoom on those traces, synced across all of them**: scroll wheel zooms
  around the cursor, drag pans, double-click returns to the whole lap, a click
  without dragging moves the playhead there. The y range of each trace fits
  the visible part. A label shows the visible distance range.
- **Top-down map**: ghost lines (dashed, ghost colour) under your line, ghost
  markers under your chevron, labelled G1..G4; a delta pill over the map (one
  value per ghost).
- **New map overlay "Time vs ghost 1"**: your line coloured by where you gain
  (teal) or lose (coral) against the first ghost; legend Gaining / Losing.
  The chase view's coloured line follows it like the other overlays.
- **Chase view**: each ghost as a see-through car of its own model class,
  resting on the track surface, plus its driven line on the surface; the delta
  pill stays visible and does not overlap the chase view's own notices.

### 5.2 What may be compared
- **Same track layout.** Same track name is not enough (Fuji files contain two
  layouts). A lap is another layout if its length differs from yours by more
  than ~3 % (the mockup's rule; the builder may use something better, e.g. the
  game's lap distance at the line). Such laps are refused with the reason, and
  the best-other search skips them ("Skipped 5 faster laps on another
  layout").
- **Same car class** (`car_identity.identify_car`, `car_class`). Other classes
  are not offered and are refused if requested. If your lap's class is
  `unknown`, files of unknown class on the same track are allowed, labelled as
  unverified.
- **Not partial laps** (see 4.5): best-lap helpers only consider complete laps.
- When your lap changes to another track, all ghosts go; to another class, the
  ghosts of the old class go.

### 5.3 Alignment and accuracy
- **Distance**: align laps by the game's `Lap Dist` where both files have it
  (recorder files do); fall back to distance driven along the line, scaled to
  the lap. Time delta at a point = your elapsed time there minus the ghost's
  elapsed time at the same distance.
- **One coordinate frame per track.** Today each file is projected around its
  own session's bounding box, so the same corner is a few metres off between
  files. Laps from different files must land in one frame (e.g. a fixed origin
  per track from the track reference); the mockup's in-browser mean-offset
  correction is a stopgap, not the design. Target: the same corner overlays
  within about half a metre between files.
- Line offset: perpendicular distance from your line to the ghost's line at
  the same distance, signed by your direction of travel.
- Everything in 5.3 is an estimate built from 10 Hz position; say so where the
  numbers are shown (tooltip), as the body-slip readout already does.

### 5.4 Finding laps quickly
"Best other file" must not open every recording (the mockup took several
seconds on Fuji). The backend provides a metadata-only index per file (track,
car model, class, and the file's lap times), cached by file modification time,
read-only. The mockup's `/sessions/index` is the starting point.

### 5.5 Session-only state
Ghosts, alignment mode and zoom are not saved between visits.

## 6. Resizable telemetry panel

- **Docked**: a drag handle on the panel's left edge sets its width (minimum
  300 px; the map keeps at least ~520 px). Arrow keys on the handle resize;
  double-click resets. The map redraws to the new size.
- **Floating**: resizable from every edge and corner (the existing bottom-left
  grip stays). Move by the title bar as today.
- **Scale** is separate from size in both modes: a **Size** row under the tabs
  (A-, percentage, A+, Reset) and Ctrl+scroll over the panel, 50-200 %.
  PREVIOUSLY the floating panel's scale followed its width; that coupling is
  removed (a bigger window shows more, not bigger text).
- Docked width and scale persist (localStorage); canvases redraw crisp at the
  new size and scale.

## 7. Decided in the conversation
- Kept laps live in their own folder so the sessions folder can be cleared.
- Multiple laps can be kept at once; one file per lap.
- The app offers to delete the original after keeping; it is never automatic.
- Comparison across files, several ghosts at once, ghost in the chase view.
- Same class only; no cross-class comparison.
- Synced scroll-wheel zoom on the comparison traces.
- Docked and floating panel both resizable and scalable.

## 8. Calls made without asking (check these)
- Kept file naming and the extra metadata keys (4.2).
- The "Reference laps" group is renamed **Kept laps**.
- Delete asks twice; "Keep the recording" is the default button.
- Up to 4 ghosts; palette pink, lime, orange, violet (your lap white).
- Layout tolerance ~3 %; unknown class allowed only against unknown class.
- Kept laps from native LMU files are out of scope.
- Comparison state is not persisted; panel width/scale are.
- The partial-lap "best" bug is fixed as part of this work.
- Kept files carry the original recording's file-level tyre-model context, so
  a kept lap counts the same after the original is deleted (4.3).
- De-duplication covers tyre data and topography only; the baselines'
  existing double counting on re-analysis is left as is and logged (4.3).
- The keep check goes through the backend (one lap reader); keeping needs
  Docker up, and the agent proves it writes where Docker reads (4.4).

## 9. Left to the builder
- Code structure, endpoint names, how the kept file is cut (row mapping per
  table rate, margin length within "a few seconds"), how the lap list hides the
  margin fragments.
- Exact alignment method within 5.3, and the per-track coordinate origin.
- Ghost car rendering details (material, opacity), colours within the design
  tokens.
- Tests: backend pytest (cut and check on a synthetic recorder file -
  `tests/synthetic_track.py` exists; index endpoint; grip-profile
  de-duplication), host agent tests for path safety and the delete
  preconditions, Playwright UI tests next to `backend/tests/ui/` for the
  compare tab, zoom sync, keep flow (with the agent mocked) and panel sizing.
- DevNotes entry and HANDOFF update when it ships.
