"""Screen capture + lightweight change detection.

We use `mss` for fast pixel grabs. Note: an `mss.mss()` instance is NOT safe to
share across threads, so we create it lazily inside whatever thread calls
`grab()` (the watch worker runs in its own QThread).

Two ways to choose what to watch:
  * a whole monitor (always available, most reliable)
  * a specific window, by region, discovered via `wmctrl` if it's installed
    (X11 only; on Wayland this generally won't work).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import mss
except Exception:  # pragma: no cover - import guarded so tests can run headless
    mss = None


@dataclass
class Source:
    """A rectangular capture target."""
    label: str
    region: dict  # {"left","top","width","height"} as mss expects


class Capture:
    """Thin wrapper around an mss instance, created lazily per-thread."""

    def __init__(self) -> None:
        self._sct = None

    def _ensure(self):
        if mss is None:
            raise RuntimeError(
                "The 'mss' package is not installed. Run: uv add mss"
            )
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def grab(self, region: dict) -> Image.Image:
        raw = self._ensure().grab(region)
        # mss returns BGRA; convert to a clean RGB PIL image.
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def list_monitors() -> list[Source]:
    """Return one Source per physical monitor (index 0 is the virtual union)."""
    if mss is None:
        return []
    with mss.mss() as sct:
        out: list[Source] = []
        for i, m in enumerate(sct.monitors):
            if i == 0:
                label = f"All monitors ({m['width']}x{m['height']})"
            else:
                label = f"Monitor {i} ({m['width']}x{m['height']})"
            out.append(Source(label=label, region=dict(m)))
        return out


def list_windows() -> list[Source]:
    """Discover open windows via `wmctrl -lG`. Returns [] if wmctrl is absent."""
    if shutil.which("wmctrl") is None:
        return []
    try:
        raw = subprocess.check_output(["wmctrl", "-lG"], text=True)
    except Exception:
        return []
    sources: list[Source] = []
    for line in raw.splitlines():
        parts = line.split(None, 7)  # winid desktop x y w h host title...
        if len(parts) < 8:
            continue
        _, _, x, y, w, h, _, title = parts
        try:
            region = {
                "left": int(x), "top": int(y),
                "width": int(w), "height": int(h),
            }
        except ValueError:
            continue
        if region["width"] <= 0 or region["height"] <= 0:
            continue
        sources.append(Source(label=f"Window: {title[:50]}", region=region))
    return sources


# --- change detection ------------------------------------------------------

def signature(img: Image.Image, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Tiny grayscale fingerprint used for cheap frame-to-frame comparison."""
    small = img.convert("L").resize(size)
    return np.asarray(small, dtype=np.float32)


def frame_diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean absolute difference between two signatures (0-255). inf if unknown."""
    if a is None or b is None:
        return float("inf")
    return float(np.mean(np.abs(a - b)))
