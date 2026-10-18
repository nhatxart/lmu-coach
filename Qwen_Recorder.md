You’ve got a small but important pipeline here:

```text
LMU shared memory
      |
      v
recorder.py  (long-running sampler / DuckDB writer)
      |
      v
data/sessions/*.duckdb
      |
      v
backend / web UI

Separately:

Web UI Recording tab
      |
      v
recorder_control.py  (local HTTP supervisor on localhost:8420)
      |
      v
starts/stops recorder.py as a subprocess
```

Below is a file-by-file digest, then I’ll focus on the specific issue you mentioned: **`START_RECORDER_CONTROL.ps1` not stopping recording when prompted to**.

---

# 1. High-level summary of the codebase

This is an **LMU recording system** designed to:

1. Capture telemetry from `Live for Speed` / `lmu` shared memory.
2. Save each session into a DuckDB file.
3. Keep the saved schema compatible with LMU’s native `.duckdb` export so existing backend endpoints keep working.
4. Provide a local HTTP service so a web UI can start/stop recording and stream logs.

The project is split into four pieces:

- **`recorder.py`**
  - The actual recorder engine.
  - Reads LMU shared memory.
  - Samples telemetry at 100 Hz.
  - Samples slower geometry/thermal channels at 10 Hz.
  - Records events (lap, gear, ABS, TC, pits) on change/crossing.
  - Writes one DuckDB file per session.

- **`recorder_control.py`**
  - A small FastAPI service.
  - Runs locally on the Windows host.
  - Manages `recorder.py` as a subprocess.
  - Exposes:
    - `GET /status`
    - `POST /start`
    - `POST /stop`
    - `GET /logs/stream`
  - This is what the web UI talks to for the Recording tab.

- **`START_RECORDER.ps1`**
  - Launcher for the standalone recorder.
  - Runs `uv sync`
  - Runs `uv run recorder.py`

- **`START_RECORDER_CONTROL.ps1`**
  - Launcher for the control service.
  - Runs `uv sync`
  - Runs `uv run recorder_control.py`

---

# 2. What each file does in detail

---

## `recorder.py`

### Purpose
This is the **core data-capture process**.

It does the following:

- Connects to LMU’s shared memory.
- Finds the player vehicle.
- Samples telemetry repeatedly.
- Detects session boundaries.
- Buffers session data in memory.
- Writes a complete DuckDB session file when the session ends.

---

### Main concepts

#### 1. Master clock
The recorder intentionally uses:

```python
current_et = float(telem.mElapsedTime)
```

instead of relying on the slower scoring clock.

Why?

- Telemetry updates more frequently.
- This gives better timestamp precision.
- Lap durations can be computed accurately from differences in this clock.

This is one of the main correctness fixes in the code.

---

#### 2. Sampling strategy
The recorder separates channels into three groups:

- **100 Hz channels**
  - position
  - speed
  - inputs
  - G-forces
  - yaw
  - wheel data
  - RPM

- **10 Hz channels**
  - pseudo-GPS lat/lon
  - path lateral
  - track edge
  - lap distance
  - tyre temperature
  - tyre pressure

- **Event channels**
  - lap
  - lap time
  - gear
  - ABS
  - TC
  - in pits

This matches the intent described in the module docstring.

---

### `ConnectionState`

This class watches whether the session is still “alive”.

It detects:

- **reset / session change**
  - elapsed time goes backward
  - car enters garage
  - vehicle slot changes

- **freeze / idle**
  - `mElapsedTime` stops advancing for longer than `FREEZE_TIMEOUT_S`

This is important because it lets the recorder flush the session when:

- the player quits to menu
- the game pauses
- the session becomes idle
- the game closes cleanly

Without this, data could sit in memory until an unclean process kill.

---

### `SessionBuffer`

This accumulates everything for one driving session.

It stores:

- continuous channels in a dictionary of lists
- event tables for lap/gear/ABS/TC/pits
- metadata such as car/track/driver names
- origin for pseudo-GPS projection

It also handles:

- lap crossing registration
- lap time calculation
- per-wheel channel storage
- event de-duplication (write only on change)

In short: **this is the in-memory “session” object before it is written to disk.**

---

### `write_session()`

This is the DuckDB writer.

Its job is to write a **schema-compatible session file**.

Important responsibilities:

- create metadata table
- create `channelsList`
- create `eventsList`
- create one table per channel/event
- insert channel data
- write event tables

#### Important design choices

##### Atomic writes
It writes to a temporary file first:

```python
tmp_out = output_dir / f".{out.name}.tmp"
```

Then only after a successful commit does it do:

```python
os.replace(tmp_out, out)
```

Why?

So that readers never see a half-written session file.

This directly addresses the earlier “silently truncated file” problem.

---

##### Fast bulk inserts
Instead of inserting row-by-row, it uses patterns like:

```python
INSERT INTO "channel" SELECT unnest(?)
```

and for wheel channels:

```python
INSERT INTO "channel" SELECT unnest(?), unnest(?), unnest(?), unnest(?)
```

Why?

Because row-by-row Python inserts were too slow and caused long shutdown windows where interruptions could ruin the file.

---

##### Incomplete final lap removal
When flushing, if the session ends while a lap is still in progress, the recorder drops the last incomplete lap event:

```python
if len(buf.lap_events) > 1:
    ...
```

Why?

Because otherwise you get a fake short lap at the end of the session.

---

### `run_recorder()`

This is the main loop.

It does:

- connects to LMU shared memory
- reconnects if the connection drops
- finds player vehicle
- samples telemetry
- detects session resets/car/track changes
- detects idle/freeze
- creates a new `SessionBuffer`
- records lap crossings
- saves the session on shutdown

It also handles:

- `Ctrl+C`
- `SIGTERM`
- `SIGBREAK`
- stop file polling

---

### Stop mechanism in `recorder.py`

The recorder will stop if:

- it receives `SIGINT`
- it receives `SIGTERM`
- it receives `SIGBREAK`
- it sees the stop file:

```python
STOP_FILE = Path(__file__).parent / ".stop_requested"
```

So there are **two stop paths**:

1. **Signal-based stop**
2. **File-based stop**

The file-based stop exists because on Windows, killing a process may not give you clean signal behavior.

---

## `recorder_control.py`

### Purpose
This is a **supervisor service** for the recorder.

It exists because:

- the web UI needs something to talk to
- the recorder itself is not an HTTP server
- the Docker backend cannot directly manage the Windows-host recorder process
- you want to see logs and start/stop recording from the UI

---

### Main component: `RecorderState`

This object owns:

- the recorder subprocess
- a recent log buffer
- subscriber queues for live log streaming
- stopping state

Key methods:

- `start()`
- `stop()`
- `_read_output()`
- `_await_exit_or_kill()`
- `subscribe()` / `unsubscribe()`

---

### `start()`

When `/start` is called:

- it ensures no recorder is already running
- clears stale stop file
- launches:

```python
subprocess.Popen(
    [sys.executable, "-u", str(RECORDER_SCRIPT)],
    ...
)
```

So the control service directly starts the recorder Python process.

This is intentional, to avoid the old problem where `uv run` became an extra wrapper layer in the process tree.

---

### `stop()`

When `/stop` is called:

- it sets `stopping = True`
- writes the stop file:

```python
STOP_FILE.write_text("stop")
```

- starts a background task to wait for exit or force-kill

Important: it returns immediately. It does **not** wait for the whole session save to complete.

That’s a good design, because saving can take a while.

---

### `_await_exit_or_kill()`

This waits for the recorder to exit.

If it does not exit within:

```python
GRACEFUL_STOP_TIMEOUT_S = 300.0
```

then it force-kills the process tree.

So if you request a stop, you may have to wait up to 5 minutes before a force-kill happens.

---

### Log streaming

The `/logs/stream` endpoint uses Server-Sent Events.

It:

- replays recent buffered log lines
- then streams new lines live

This lets the web UI show recorder output without polling.

---

### Endpoints

- `GET /status`
  - running/stopping state
  - start time
  - recent logs

- `POST /start`
  - start the recorder subprocess

- `POST /stop`
  - request graceful stop

- `GET /logs/stream`
  - live log stream

---

## `START_RECORDER.ps1`

This just runs the recorder directly:

```powershell
cd $PSScriptRoot
uv sync
uv run recorder.py
```

Use this if you want:

- standalone recording
- no web UI supervisor

---

## `START_RECORDER_CONTROL.ps1`

This runs the control service:

```powershell
cd $PSScriptRoot
uv sync
uv run recorder_control.py
```

Use this if you want:

- the web UI Recording tab to work
- start/stop from the browser
- log streaming

It does **not** run the recorder directly. It starts the service that *manages* the recorder.

---

# 3. Why `START_RECORDER_CONTROL.ps1` may not stop recording when prompted to

This is the important part.

There are a few likely explanations, and they usually come down to one of these:

## A. You are stopping the *control service*, not cleanly stopping the *recorder*

`START_RECORDER_CONTROL.ps1` starts the **control service**.

The actual recorder is a **separate child process**.

So there are three different “stop” actions:

1. **Stop the recorder via the web UI / HTTP stop**
   - this writes `.stop_requested`
   - recorder flushes and exits cleanly

2. **Stop the control service gracefully**
   - its shutdown hook should call `state.stop()`
   - which writes `.stop_requested`
   - then waits for the recorder to finish

3. **Kill the control service window / process**
   - this may bypass the graceful stop path entirely

If you are doing #3, especially by closing the PowerShell window, the recorder may not get a clean stop.

---

## B. Closing the PowerShell window is not a graceful stop

This is probably the biggest practical issue.

On Windows:

- `Ctrl+C` can be handled cleanly
- closing the console window is much harsher

When you close the window, Windows may give the process only a short amount of time before it is terminated.

Your code even says this explicitly:

> Windows gives a console only a few seconds after the window's X button is clicked, so use Ctrl+C or the Stop button rather than closing the window while recording.

So if you close `START_RECORDER_CONTROL.ps1`’s window while a session is saving, the control service may die before it can:

- finish the shutdown hook
- wait for the recorder subprocess
- let the DuckDB save complete

Result:
- recorder may be orphaned
- or killed before saving
- or left in a half-stopped state

---

## C. `uv run` can make Ctrl+C / shutdown behavior less reliable

Your launcher uses:

```powershell
uv run recorder_control.py
```

That means the console is running `uv`, and `uv` is running the Python control service.

This extra wrapper layer can make shutdown less predictable, especially on Windows.

When you press `Ctrl+C`, the signal handling path can become:

```text
console -> uv -> python -> uvicorn -> shutdown hook -> recorder stop
```

If any layer exits too quickly, the graceful stop chain can break.

So even though `recorder_control.py` has a shutdown handler, the actual reliability of that handler depends on how the process is launched and stopped.

---

## D. The stop request may be working, but you are not waiting long enough

The control service uses a very long graceful timeout:

```python
GRACEFUL_STOP_TIMEOUT_S = 300.0
```

So if you press stop, and the recorder is in the middle of saving a long session, the control service may remain in a “stopping” state for up to 5 minutes.

If you assume it is broken immediately, it may actually just be:

- writing the DuckDB file
- committing tables
- waiting for the child process to exit

This is less “not stopping” and more “stopping slowly”.

---

## E. The recorder may be stuck writing, not stuck stopping

Another possibility is that the recorder is actually stopping, but the save step is slow or blocked.

Things that can make this look like “not stopping”:

- very long session
- slow disk
- antivirus scanning the `.duckdb` file
- another process holding the file
- DuckDB commit taking longer than expected
- writer blocked on I/O

In that case the stop logic is fine; the disk/write step is the bottleneck.

---

# 4. What is actually happening when you ask it to stop

Here is the intended stop chain for a normal HTTP stop:

```text
Web UI / manual POST /stop
        |
        v
recorder_control.py state.stop()
        |
        v
writes .stop_requested
        |
        v
recorder.py main loop sees stop file
        |
        v
flush current session
        |
        v
write_session() saves DuckDB file
        |
        v
process exits
```

That is the clean path.

If you kill the control service instead, the chain becomes fragile:

```text
close window / kill uv / kill control service
        |
        v
maybe shutdown hook runs
        |
        v
maybe writes stop file
        |
        v
recorder may or may not get time to save
```

That is where your problem most likely comes from.

---

# 5. Most likely root cause

Based on the code and the symptoms, the most likely explanation is:

**You are stopping the control service in a way that does not give the recorder subprocess enough time to finish its clean shutdown.**

In other words:

- the stop request may be happening at the wrong layer
- or it is being interrupted before the recorder finishes saving
- or `uv run` / console close is breaking the graceful shutdown path

The system is designed to stop cleanly if you use:

- the **Stop button in the web UI**
- or a direct `POST /stop`
- or a graceful shutdown of the control service that is allowed to finish

It is **not** designed to survive you closing the PowerShell window instantly while a save is in progress.

---

# 6. How to confirm what is happening

You can diagnose this very quickly by checking a few things.

---

## Check if the recorder subprocess is still running

In PowerShell:

```powershell
Get-Process python | Select-Object Id, ProcessName, Path
```

Or more specifically:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Select-Object ProcessId, CommandLine
```

If you still see a `recorder.py` process, then the recorder was not stopped.

---

## Check if the stop file exists

Look in the `recorder/` directory for:

```text
.stop_requested
```

Interpretation:

- If `.stop_requested` is present and the recorder is still running:
  - the recorder saw the stop request but may be stuck saving
  - or it is taking a long time

- If `.stop_requested` is absent but the recorder is still running:
  - the control service probably never issued a clean stop
  - or it was killed before writing the stop file

---

## Check recent control-service logs

If you have the web UI open, look at the Recording tab log stream.

You should see lines like:

```text
[control] stop requested — recorder will save the session, then exit
```

and then later:

```text
[recorder process exited, code 0]
```

If you do not see the second line, the recorder has not exited.

---

## Check whether the session file was actually saved

Look in:

```text
../data/sessions
```

or the configured output directory.

If you see:

- a final `.duckdb` file
- no leftover `.tmp` file

then the stop/save eventually succeeded.

If you see:

- a leftover `.tmp` file
- no final `.duckdb`

then the write was interrupted.

---

# 7. Best practical fix: stop via the HTTP endpoint, not by killing the launcher

The safest way to stop recording is:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8420/stop"
```

or use the web UI Stop button.

Then watch:

```powershell
Invoke-RestMethod -Uri "http://localhost:8420/status"
```

until `running` becomes false.

This gives the whole pipeline the cleanest chance to finish.

---

# 8. If you want `Ctrl+C` in the control window to work better

If you want stopping the control window to be reliable, you have a few options.

---

## Option 1: Press `Ctrl+C` in the control service window and wait

Do **not**:
- close the window
- press `Ctrl+C` multiple times
- kill the process from Task Manager

Instead:
- press `Ctrl+C` once
- wait for the shutdown to finish

This gives the FastAPI/uvicorn shutdown hook the best chance to run.

---

## Option 2: Avoid the extra `uv run` wrapper when stopping matters

`uv run` is convenient, but it adds another process layer.

For the control service, a more direct launch is more predictable.

For example:

```powershell
cd $PSScriptRoot
uv sync
& (uv run -- python -c "import sys; print(sys.executable)") -u recorder_control.py
```

That is awkward.

A cleaner approach is to activate the environment and run Python directly:

```powershell
cd $PSScriptRoot
uv sync
.\.venv\Scripts\Activate.ps1
python -u recorder_control.py
```

Then `Ctrl+C` goes more directly to the Python control service.

This often behaves more predictably on Windows.

---

## Option 3: Make the shutdown hook more aggressive

Right now, the shutdown hook waits up to 300 seconds before force-killing.

That is good for clean saves, but bad if you want the control service itself to stop promptly.

You could split the behavior:

- long timeout for UI stop requests
- shorter timeout when the control service is shutting down

For example:

```python
SERVICE_SHUTDOWN_TIMEOUT_S = 30.0
```

and in the shutdown hook, use that shorter timeout.

This reduces the chance that the control service hangs for 5 minutes when you are trying to exit.

But it also means you are more likely to force-kill a save if it is slow.

---

## Option 4: Add a clearer shutdown path for Windows console close

If you specifically care about closing the PowerShell window, you need to handle the Windows console-close event better.

That is more advanced than the current code does.

Options include:

- register a console control handler for `CTRL_CLOSE_EVENT`
- run the control service as a background service / scheduled task
- or make the recorder itself robust to the control process dying

The last option is probably the best long-term architecture:

- if the control process dies, the recorder should detect that and flush/exit

Right now, the recorder only knows to stop via:
- signals
- stop file

It does **not** know that the control service disappeared.

---

# 9. A small but useful code-level observation

There is a minor bug in `ConnectionState.update()`:

```python
return reset, just_froze
return reset
```

The second `return reset` is unreachable.

It does not affect behavior, but it is dead code and will confuse readers.

You can safely remove:

```python
return reset
```

---

# 10. Recommended way to run this system

To get the most reliable behavior:

## For normal use
1. Run `START_RECORDER_CONTROL.ps1`
2. Leave that window open
3. Start recording from the web UI
4. Stop recording from the web UI
5. Wait for the status to show not running

Do **not** close the window while a save is happening.

---

## If you need to stop the whole control service
1. Stop recording first from the UI
2. Wait for the recorder subprocess to exit
3. Then stop the control service

That ordering gives you the cleanest shutdown.

---

## If you must close the control window
Expect the recorder to be at risk.

The safest thing is:
- stop recording first
- wait for the save
- then close the window

---

# 11. Bottom line

### What each file does

- **`recorder.py`**
  - the real recorder
  - reads LMU shared memory
  - builds sessions in memory
  - writes DuckDB files

- **`recorder_control.py`**
  - local HTTP supervisor
  - starts/stops `recorder.py`
  - streams logs
  - lets the web UI control recording

- **`START_RECORDER.ps1`**
  - runs the recorder directly

- **`START_RECORDER_CONTROL.ps1`**
  - runs the control service, not the recorder directly

---

### Why stop is failing

The most likely reason is:

**You are stopping the launcher/control service in a way that does not give the recorder subprocess a clean shutdown window.**

In practice, that usually means:

- closing the PowerShell window
- killing the `uv`/control process
- or not waiting long enough for the save

The cleanest stop path is:

```text
POST http://localhost:8420/stop
```

or the web UI Stop button.

If you want `Ctrl+C` / window close to behave better, you should either:

- run the control service directly with Python instead of through `uv run`
- or improve the shutdown handling so the control service and recorder survive console-close more gracefully

---

If you want, I can also do the next step and give you a **specific patch** for `recorder_control.py` and the PowerShell script to make stopping more reliable on Windows.

49.34 t/s, 4.5 min gen time, 13474 tokens for a 12874 prompt token, 744.77 t/s prompt processing speed for a 53 kB (4 files. 2 Py 2 PS1) input

0.11.506.014 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.11.506.064 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.13.100.808 I slot print_timing: id  0 | task 0 | prompt eval time =     887.88 ms /    34 tokens (   26.11 ms per token,    38.29 tokens per second)
0.13.100.815 I slot print_timing: id  0 | task 0 |        eval time =     706.82 ms /    44 tokens (   16.44 ms per token,    60.84 tokens per second)
0.13.100.816 I slot print_timing: id  0 | task 0 |       total time =    1594.70 ms /    78 tokens
0.13.100.816 I slot print_timing: id  0 | task 0 |    graphs reused =         23
0.13.100.819 I slot print_timing: id  0 | task 0 | draft acceptance = 0.91304 (   21 accepted /    23 generated), mean len =  1.91
0.13.100.858 I slot      release: id  0 | task 0 | stop processing: n_tokens = 78, truncated = 0
3.21.544.875 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 13086669
3.21.606.293 I slot launch_slot_: id  0 | task 26 | processing task, is_child = 0
3.24.941.632 I slot print_timing: id  0 | task 26 | n_gen =    151, tg =  49.90 t/s, tg_3s =  50.24 t/s
3.27.942.882 I slot print_timing: id  0 | task 26 | n_gen =    311, tg =  51.61 t/s, tg_3s =  53.31 t/s
3.29.776.387 I slot print_timing: id  0 | task 26 | prompt eval time =     329.54 ms /    64 tokens (    5.15 ms per token,   194.21 tokens per second)
3.29.776.396 I slot print_timing: id  0 | task 26 |        eval time =    7840.49 ms /   409 tokens (   19.22 ms per token,    52.04 tokens per second)
3.29.776.397 I slot print_timing: id  0 | task 26 |       total time =    8170.04 ms /   473 tokens
3.29.776.398 I slot print_timing: id  0 | task 26 |    graphs reused =        270
3.29.776.402 I slot print_timing: id  0 | task 26 | draft acceptance = 0.64257 (  160 accepted /   249 generated), mean len =  1.64
3.29.776.437 I slot      release: id  0 | task 26 | stop processing: n_tokens = 473, truncated = 0
4.06.742.307 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 209762248
4.06.846.723 I slot launch_slot_: id  0 | task 279 | processing task, is_child = 0
4.11.787.606 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =   4097, progress = 0.33, t =   4.27 s / 959.97 tokens per second
4.14.479.408 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =   6145, progress = 0.48, t =   6.85 s / 896.56 tokens per second
4.17.317.449 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =   8193, progress = 0.63, t =   9.70 s / 844.81 tokens per second
4.20.296.025 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =  10241, progress = 0.78, t =  12.64 s / 810.25 tokens per second
4.23.372.642 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =  12289, progress = 0.93, t =  15.67 s / 784.35 tokens per second
4.24.032.974 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =  12716, progress = 0.96, t =  16.54 s / 768.65 tokens per second
4.24.882.493 I slot print_timing: id  0 | task 279 | prompt processing, n_tokens =  13228, progress = 1.00, t =  17.27 s / 765.92 tokens per second
4.28.012.387 I slot print_timing: id  0 | task 279 | n_gen =    151, tg =  49.65 t/s, tg_3s =  49.98 t/s
4.31.027.845 I slot print_timing: id  0 | task 279 | n_gen =    297, tg =  49.04 t/s, tg_3s =  48.42 t/s
4.34.038.274 I slot print_timing: id  0 | task 279 | n_gen =    447, tg =  49.30 t/s, tg_3s =  49.83 t/s
4.35.386.600 W srv          stop: cancel task, id_task = 279
4.35.419.667 I slot      release: id  0 | task 279 | stop processing: n_tokens = 14217, truncated = 0
5.32.273.599 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 275405483
5.32.748.623 I slot launch_slot_: id  0 | task 606 | processing task, is_child = 0
5.37.766.803 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =   4113, progress = 0.32, t =   4.35 s / 945.55 tokens per second
5.40.365.343 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =   6161, progress = 0.48, t =   6.89 s / 894.05 tokens per second
5.43.114.125 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =   8209, progress = 0.64, t =   9.61 s / 854.14 tokens per second
5.45.984.771 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =  10257, progress = 0.80, t =  12.44 s / 824.28 tokens per second
5.48.989.172 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =  12305, progress = 0.96, t =  15.41 s / 798.38 tokens per second
5.49.141.023 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =  12358, progress = 0.96, t =  16.25 s / 760.63 tokens per second
5.49.940.696 I slot print_timing: id  0 | task 606 | prompt processing, n_tokens =  12870, progress = 1.00, t =  16.44 s / 782.73 tokens per second
5.53.061.641 I slot print_timing: id  0 | task 606 | n_gen =    168, tg =  55.17 t/s, tg_3s =  55.50 t/s
5.56.073.426 I slot print_timing: id  0 | task 606 | n_gen =    312, tg =  51.50 t/s, tg_3s =  47.81 t/s
5.59.105.784 I slot print_timing: id  0 | task 606 | n_gen =    466, tg =  51.26 t/s, tg_3s =  50.79 t/s
6.02.124.323 I slot print_timing: id  0 | task 606 | n_gen =    612, tg =  50.54 t/s, tg_3s =  48.37 t/s
6.05.125.647 I slot print_timing: id  0 | task 606 | n_gen =    753, tg =  49.83 t/s, tg_3s =  46.98 t/s
6.08.157.063 I slot print_timing: id  0 | task 606 | n_gen =    906, tg =  49.94 t/s, tg_3s =  50.47 t/s
6.11.157.815 I slot print_timing: id  0 | task 606 | n_gen =   1051, tg =  49.71 t/s, tg_3s =  48.32 t/s
6.14.175.760 I slot print_timing: id  0 | task 606 | n_gen =   1195, tg =  49.46 t/s, tg_3s =  47.71 t/s
6.17.176.728 I slot print_timing: id  0 | task 606 | n_gen =   1338, tg =  49.26 t/s, tg_3s =  47.65 t/s
6.20.177.158 I slot print_timing: id  0 | task 606 | n_gen =   1485, tg =  49.23 t/s, tg_3s =  48.99 t/s
6.23.207.371 I slot print_timing: id  0 | task 606 | n_gen =   1633, tg =  49.20 t/s, tg_3s =  48.84 t/s
6.26.236.321 I slot print_timing: id  0 | task 606 | n_gen =   1774, tg =  48.98 t/s, tg_3s =  46.55 t/s
6.29.241.120 I slot print_timing: id  0 | task 606 | n_gen =   1911, tg =  48.72 t/s, tg_3s =  45.59 t/s
6.32.244.107 I slot print_timing: id  0 | task 606 | n_gen =   2063, tg =  48.85 t/s, tg_3s =  50.62 t/s
6.35.248.699 I slot print_timing: id  0 | task 606 | n_gen =   2207, tg =  48.79 t/s, tg_3s =  47.93 t/s
6.38.259.224 I slot print_timing: id  0 | task 606 | n_gen =   2350, tg =  48.71 t/s, tg_3s =  47.50 t/s
6.41.292.363 I slot print_timing: id  0 | task 606 | n_gen =   2493, tg =  48.62 t/s, tg_3s =  47.15 t/s
6.44.302.660 I slot print_timing: id  0 | task 606 | n_gen =   2637, tg =  48.57 t/s, tg_3s =  47.84 t/s
6.47.327.518 I slot print_timing: id  0 | task 606 | n_gen =   2787, tg =  48.63 t/s, tg_3s =  49.59 t/s
6.50.331.786 I slot print_timing: id  0 | task 606 | n_gen =   2940, tg =  48.74 t/s, tg_3s =  50.93 t/s
6.53.359.783 I slot print_timing: id  0 | task 606 | n_gen =   3078, tg =  48.59 t/s, tg_3s =  45.57 t/s
6.56.386.504 I slot print_timing: id  0 | task 606 | n_gen =   3232, tg =  48.69 t/s, tg_3s =  50.88 t/s
6.59.405.360 I slot print_timing: id  0 | task 606 | n_gen =   3385, tg =  48.78 t/s, tg_3s =  50.68 t/s
7.02.407.829 I slot print_timing: id  0 | task 606 | n_gen =   3522, tg =  48.65 t/s, tg_3s =  45.63 t/s
7.05.408.505 I slot print_timing: id  0 | task 606 | n_gen =   3676, tg =  48.76 t/s, tg_3s =  51.32 t/s
7.08.416.546 I slot print_timing: id  0 | task 606 | n_gen =   3824, tg =  48.77 t/s, tg_3s =  49.20 t/s
7.11.438.461 I slot print_timing: id  0 | task 606 | n_gen =   3972, tg =  48.78 t/s, tg_3s =  48.98 t/s
7.14.471.608 I slot print_timing: id  0 | task 606 | n_gen =   4124, tg =  48.83 t/s, tg_3s =  50.11 t/s
7.17.495.949 I slot print_timing: id  0 | task 606 | n_gen =   4261, tg =  48.71 t/s, tg_3s =  45.30 t/s
7.20.522.418 I slot print_timing: id  0 | task 606 | n_gen =   4404, tg =  48.66 t/s, tg_3s =  47.25 t/s
7.23.537.129 I slot print_timing: id  0 | task 606 | n_gen =   4551, tg =  48.66 t/s, tg_3s =  48.76 t/s
7.26.567.943 I slot print_timing: id  0 | task 606 | n_gen =   4698, tg =  48.66 t/s, tg_3s =  48.50 t/s
7.29.571.571 I slot print_timing: id  0 | task 606 | n_gen =   4839, tg =  48.61 t/s, tg_3s =  46.94 t/s
7.32.588.081 I slot print_timing: id  0 | task 606 | n_gen =   4981, tg =  48.56 t/s, tg_3s =  47.07 t/s
7.35.608.320 I slot print_timing: id  0 | task 606 | n_gen =   5124, tg =  48.53 t/s, tg_3s =  47.35 t/s
7.38.622.592 I slot print_timing: id  0 | task 606 | n_gen =   5269, tg =  48.51 t/s, tg_3s =  48.10 t/s
7.41.643.301 I slot print_timing: id  0 | task 606 | n_gen =   5412, tg =  48.48 t/s, tg_3s =  47.34 t/s
7.44.654.412 I slot print_timing: id  0 | task 606 | n_gen =   5557, tg =  48.47 t/s, tg_3s =  48.15 t/s
7.47.668.378 I slot print_timing: id  0 | task 606 | n_gen =   5706, tg =  48.50 t/s, tg_3s =  49.44 t/s
7.50.688.683 I slot print_timing: id  0 | task 606 | n_gen =   5852, tg =  48.49 t/s, tg_3s =  48.34 t/s
7.53.718.044 I slot print_timing: id  0 | task 606 | n_gen =   5993, tg =  48.45 t/s, tg_3s =  46.54 t/s
7.56.743.803 I slot print_timing: id  0 | task 606 | n_gen =   6139, tg =  48.44 t/s, tg_3s =  48.25 t/s
7.59.746.928 I slot print_timing: id  0 | task 606 | n_gen =   6287, tg =  48.46 t/s, tg_3s =  49.28 t/s
8.02.765.805 I slot print_timing: id  0 | task 606 | n_gen =   6429, tg =  48.43 t/s, tg_3s =  47.04 t/s
8.05.779.384 I slot print_timing: id  0 | task 606 | n_gen =   6588, tg =  48.52 t/s, tg_3s =  52.76 t/s
8.08.784.557 I slot print_timing: id  0 | task 606 | n_gen =   6738, tg =  48.55 t/s, tg_3s =  49.91 t/s
8.11.811.079 I slot print_timing: id  0 | task 606 | n_gen =   6873, tg =  48.47 t/s, tg_3s =  44.61 t/s
8.14.829.282 I slot print_timing: id  0 | task 606 | n_gen =   7023, tg =  48.50 t/s, tg_3s =  49.70 t/s
8.17.836.511 I slot print_timing: id  0 | task 606 | n_gen =   7169, tg =  48.50 t/s, tg_3s =  48.55 t/s
8.20.871.726 I slot print_timing: id  0 | task 606 | n_gen =   7311, tg =  48.46 t/s, tg_3s =  46.78 t/s
8.23.901.997 I slot print_timing: id  0 | task 606 | n_gen =   7461, tg =  48.48 t/s, tg_3s =  49.50 t/s
8.26.911.611 I slot print_timing: id  0 | task 606 | n_gen =   7601, tg =  48.45 t/s, tg_3s =  46.52 t/s
8.29.920.718 I slot print_timing: id  0 | task 606 | n_gen =   7743, tg =  48.42 t/s, tg_3s =  47.19 t/s
8.32.953.436 I slot print_timing: id  0 | task 606 | n_gen =   7884, tg =  48.39 t/s, tg_3s =  46.49 t/s
8.35.962.276 I slot print_timing: id  0 | task 606 | n_gen =   8025, tg =  48.36 t/s, tg_3s =  46.86 t/s
8.38.966.103 I slot print_timing: id  0 | task 606 | n_gen =   8161, tg =  48.30 t/s, tg_3s =  45.28 t/s
8.41.969.969 I slot print_timing: id  0 | task 606 | n_gen =   8316, tg =  48.36 t/s, tg_3s =  51.60 t/s
8.44.973.508 I slot print_timing: id  0 | task 606 | n_gen =   8457, tg =  48.34 t/s, tg_3s =  46.94 t/s
8.47.977.664 I slot print_timing: id  0 | task 606 | n_gen =   8620, tg =  48.44 t/s, tg_3s =  54.26 t/s
8.50.997.741 I slot print_timing: id  0 | task 606 | n_gen =   8779, tg =  48.51 t/s, tg_3s =  52.65 t/s
8.54.031.634 I slot print_timing: id  0 | task 606 | n_gen =   8922, tg =  48.48 t/s, tg_3s =  47.13 t/s
8.57.061.601 I slot print_timing: id  0 | task 606 | n_gen =   9081, tg =  48.55 t/s, tg_3s =  52.48 t/s
9.00.085.944 I slot print_timing: id  0 | task 606 | n_gen =   9235, tg =  48.59 t/s, tg_3s =  50.92 t/s
9.03.105.637 I slot print_timing: id  0 | task 606 | n_gen =   9383, tg =  48.59 t/s, tg_3s =  49.01 t/s
9.06.116.817 I slot print_timing: id  0 | task 606 | n_gen =   9540, tg =  48.65 t/s, tg_3s =  52.14 t/s
9.09.118.202 I slot print_timing: id  0 | task 606 | n_gen =   9699, tg =  48.71 t/s, tg_3s =  52.98 t/s
9.12.119.146 I slot print_timing: id  0 | task 606 | n_gen =   9858, tg =  48.78 t/s, tg_3s =  52.98 t/s
9.15.134.075 I slot print_timing: id  0 | task 606 | n_gen =  10012, tg =  48.81 t/s, tg_3s =  51.08 t/s
9.18.161.989 I slot print_timing: id  0 | task 606 | n_gen =  10165, tg =  48.84 t/s, tg_3s =  50.53 t/s
9.21.179.161 I slot print_timing: id  0 | task 606 | n_gen =  10322, tg =  48.88 t/s, tg_3s =  52.04 t/s
9.24.212.782 I slot print_timing: id  0 | task 606 | n_gen =  10478, tg =  48.92 t/s, tg_3s =  51.42 t/s
9.27.225.992 I slot print_timing: id  0 | task 606 | n_gen =  10637, tg =  48.97 t/s, tg_3s =  52.77 t/s
9.30.233.950 I slot print_timing: id  0 | task 606 | n_gen =  10788, tg =  48.99 t/s, tg_3s =  50.20 t/s
9.33.265.082 I slot print_timing: id  0 | task 606 | n_gen =  10938, tg =  48.99 t/s, tg_3s =  49.49 t/s
9.36.296.678 I slot print_timing: id  0 | task 606 | n_gen =  11084, tg =  48.98 t/s, tg_3s =  48.16 t/s
9.39.307.864 I slot print_timing: id  0 | task 606 | n_gen =  11238, tg =  49.01 t/s, tg_3s =  51.14 t/s
9.42.319.949 I slot print_timing: id  0 | task 606 | n_gen =  11386, tg =  49.01 t/s, tg_3s =  49.14 t/s
9.45.344.273 I slot print_timing: id  0 | task 606 | n_gen =  11546, tg =  49.06 t/s, tg_3s =  52.90 t/s
9.48.353.242 I slot print_timing: id  0 | task 606 | n_gen =  11703, tg =  49.10 t/s, tg_3s =  52.18 t/s
9.51.365.676 I slot print_timing: id  0 | task 606 | n_gen =  11857, tg =  49.13 t/s, tg_3s =  51.12 t/s
9.54.386.971 I slot print_timing: id  0 | task 606 | n_gen =  12011, tg =  49.15 t/s, tg_3s =  50.97 t/s
9.57.394.871 I slot print_timing: id  0 | task 606 | n_gen =  12163, tg =  49.17 t/s, tg_3s =  50.53 t/s
10.00.397.080 I slot print_timing: id  0 | task 606 | n_gen =  12313, tg =  49.18 t/s, tg_3s =  49.96 t/s
10.03.426.861 I slot print_timing: id  0 | task 606 | n_gen =  12470, tg =  49.21 t/s, tg_3s =  51.82 t/s
10.06.440.692 I slot print_timing: id  0 | task 606 | n_gen =  12627, tg =  49.24 t/s, tg_3s =  52.09 t/s
10.09.450.423 I slot print_timing: id  0 | task 606 | n_gen =  12772, tg =  49.23 t/s, tg_3s =  48.18 t/s
10.12.453.044 I slot print_timing: id  0 | task 606 | n_gen =  12931, tg =  49.27 t/s, tg_3s =  52.95 t/s
10.15.454.543 I slot print_timing: id  0 | task 606 | n_gen =  13080, tg =  49.28 t/s, tg_3s =  49.64 t/s
10.18.455.090 I slot print_timing: id  0 | task 606 | n_gen =  13236, tg =  49.31 t/s, tg_3s =  51.99 t/s
10.21.486.874 I slot print_timing: id  0 | task 606 | n_gen =  13392, tg =  49.33 t/s, tg_3s =  51.45 t/s
10.23.096.107 I slot print_timing: id  0 | task 606 | prompt eval time =   17285.82 ms / 12874 tokens (    1.34 ms per token,   744.77 tokens per second)
10.23.096.112 I slot print_timing: id  0 | task 606 |        eval time =  273061.60 ms / 13474 tokens (   20.27 ms per token,    49.34 tokens per second)
10.23.096.114 I slot print_timing: id  0 | task 606 |       total time =  290347.42 ms / 26348 tokens
10.23.096.115 I slot print_timing: id  0 | task 606 |    graphs reused =       8541
10.23.096.118 I slot print_timing: id  0 | task 606 | draft acceptance = 0.68152 ( 5461 accepted /  8013 generated), mean len =  1.68
10.23.096.507 I slot      release: id  0 | task 606 | stop processing: n_tokens = 26390, truncated = 0