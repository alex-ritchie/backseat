"""The watch loop, as a QObject worker that runs in its own QThread.

Flow each tick:
  1. sample the screen
  2. compare to the previous frame (cheap signature diff)
  3. if it changed, wait for it to stabilize (avoid mid-scroll captures)
  4. once stable AND past the cooldown, send to the LLM
  5. emit the comment if the model didn't PASS

A manual "nudge" bypasses change detection and the cooldown so you can ask
"anything to say right now?" on demand.
"""
from __future__ import annotations

import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

from backends import Backend, is_pass
from capture import Capture, frame_diff, signature
from config import Config


class WatchWorker(QObject):
    comment = pyqtSignal(str)   # a new suggestion to show the user
    frame = pyqtSignal(object)  # PIL image of the screenshot sent to the model
    sent = pyqtSignal()         # a message was sent to the model (debug)
    replied = pyqtSignal(str)   # raw model reply, PASS included (debug)
    status = pyqtSignal(str)    # short status line ("watching", "thinking", ...)
    error = pyqtSignal(str)     # non-fatal error text
    finished = pyqtSignal()     # loop has exited

    def __init__(self, cfg: Config, region: dict, backend: Backend) -> None:
        super().__init__()
        self.cfg = cfg
        self.region = region
        self.backend = backend
        self._stop = threading.Event()
        self._nudge = threading.Event()
        self._recent: list[str] = []

    # -- control (called from the GUI thread) --
    def stop(self) -> None:
        self._stop.set()

    def nudge(self) -> None:
        self._nudge.set()

    # -- the loop (runs in the worker thread) --
    def run(self) -> None:
        cap = Capture()  # created here so mss lives in this thread
        prev_sig = None
        last_call = 0.0
        pending_change = False
        stable = 0

        self.status.emit("watching")
        while not self._stop.is_set():
            forced = self._nudge.is_set()
            if self._nudge.is_set():
                self._nudge.clear()

            try:
                img = cap.grab(self.region)
            except Exception as e:
                self.error.emit(f"capture failed: {e}")
                self._sleep(self.cfg.poll_interval)
                continue

            sig = signature(img)
            diff = frame_diff(prev_sig, sig)
            prev_sig = sig
            now = time.monotonic()

            should_call = False
            if forced:
                should_call = True
            elif diff > self.cfg.change_threshold:
                # screen is actively changing; wait until it settles
                pending_change = True
                stable = 0
            elif pending_change:
                stable += 1
                if stable >= self.cfg.stabilize_frames:
                    pending_change = False
                    stable = 0
                    if now - last_call >= self.cfg.min_seconds_between_calls:
                        should_call = True

            if should_call:
                last_call = now
                self._call(img)
                self.status.emit("watching")

            self._sleep(self.cfg.poll_interval)

        self.finished.emit()

    # -- helpers --
    def _call(self, img) -> None:
        self.status.emit("thinking…")
        self.frame.emit(img)
        self.sent.emit()
        try:
            text = self.backend.comment(img, self._recent)
        except Exception as e:
            self.error.emit(f"model call failed: {e}")
            return
        self.replied.emit(text)
        if is_pass(text):
            self.status.emit("nothing to add")
            return
        self._recent.append(text)
        if len(self._recent) > self.cfg.history_turns:
            self._recent = self._recent[-self.cfg.history_turns:]
        self.comment.emit(text)

    def _sleep(self, seconds: float) -> None:
        """Sleep in small slices so stop/nudge stay responsive."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stop.is_set() or self._nudge.is_set():
                return
            time.sleep(0.05)
