# Backseat — an ambient screen-watching assistant

A prototype "always-on co-working partner." It periodically looks at a screen or
window you choose, and speaks up **only when it has something genuinely useful** —
a question, a suggestion, a connection, a possible mistake. Most of the time it
stays quiet, the way a good colleague sitting next to you would.

This is a general-purpose companion, not tied to any particular content.

## How it works

```
capture a monitor/window  ->  cheap change detection  ->  wait for the screen to
settle  ->  (respecting a cooldown)  ->  send the screenshot to a vision LLM  ->
model replies with a comment or "PASS"  ->  non-PASS comments appear in the
Suggestions pane
```

The "act like a co-worker, don't narrate the obvious, PASS when there's nothing
to add" behavior lives entirely in `SYSTEM_PROMPT` in `backends.py`.
That's the first thing to tweak.

## Setup (uv)

```bash
uv sync            # creates the venv and installs everything from pyproject.toml
```

### Backend A — Claude API (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run main.py
```

Set the model string in the GUI to whatever you have access to.

### Backend B — local via llama.cpp (managed for you)

The app launches and manages `llama-server` itself. You only need llama.cpp
installed with its binary on your `PATH`:

```bash
# build from source, or install a prebuilt release — then confirm:
llama-server --version
```

On startup the app launches the **default model** (Qwen3.8-27B, Q4) with:

```
llama-server -hf unsloth/Qwen3.8-27B-GGUF:Q4_K_M --port 8080 \
  --n-gpu-layers 99 --flash-attn on --jinja --ctx-size 8192
```

Note: this model needs a recent llama.cpp build (older CUDA builds mis-run its
DeltaNet layers).

The **Local model server** panel shows load status and lets you switch models:
pick another entry in the dropdown and click **Change model** (only enabled when
the selection differs from what's running). Changing shuts the current server
down and brings the new one up, waiting for it to be ready before you can watch
with it. The app also shuts the server down cleanly on exit, so it never leaks
VRAM.

First launch of a model downloads its weights (the 27B is ~17 GB), so the very
first "loading…" can take a while — the panel shows download and load progress. Edit the model list — including the startup
default and whether it autostarts — in `models.py` and `config.py`.

Then just:

```bash
uv run main.py     # "llamacpp" is the default backend
```

## Using it

1. Pick what to watch (a monitor, or a window if `wmctrl` is installed).
2. Choose backend + model; adjust poll interval and the minimum gap between
   calls if you want it more or less talkative.
3. **Start watching.** Comments appear in the Suggestions pane.
4. **Ask now** forces an immediate comment, bypassing change detection.

## Key knobs (`config.py`)

| Setting | What it does |
|---|---|
| `poll_interval` | how often it samples the screen |
| `min_seconds_between_calls` | hard floor on API call frequency (cost / noise) |
| `change_threshold` | how much the screen must change to trigger interest |
| `stabilize_frames` | frames of calm required before it fires (avoids mid-scroll) |
| `capture_scale` | downscale images before sending (cheaper / faster) |
| `history_turns` | recent comments fed back so it doesn't repeat itself |

## Caveats

- **Don't capture the app itself.** Put the Backseat window on a different
  monitor than the one you're watching, or capture a specific window — otherwise
  the model can start reacting to its own comments.
- **X11 vs Wayland.** `mss` screen capture and `wmctrl` window discovery work on
  X11. Under Wayland, capture is often blocked by the compositor.
- **Window capture grabs a screen region,** so if the target window is occluded
  you'll capture whatever is on top of it. Monitor capture is the reliable path.
- **Cost.** With the Claude backend, every meaningful screen change can be an API
  call. The cooldown and change threshold are your main levers; a small/cheap
  model or the llama.cpp backend is a good fit for long sessions.

## Layout

```
main.py             entry point
config.py           all tunable settings
models.py           local VLM presets shown in the dropdown
server.py           launches/stops/swaps llama-server, waits for ready
capture.py          screen/window capture + change detection
backends.py         Claude + llama.cpp backends, the co-work system prompt
worker.py           the capture->detect->call loop (runs off the UI thread)
gui.py              the single app window (server, watch, suggestions, preview)
```
