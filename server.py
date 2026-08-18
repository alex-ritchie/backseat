"""Manages the llama-server subprocess lifecycle.

Responsibilities:
  * launch llama-server for a chosen model preset
  * poll its /health endpoint (via Qt's async HTTP, so the GUI never blocks)
    and announce when it's actually ready to serve requests
  * swap models cleanly: stop the current server, then start the new one
  * shut down reliably so we never leave an orphaned server holding VRAM

Readiness = a 200 from /health. Failure = the process exiting before it became
ready (bad -hf tag, out-of-memory, binary missing, ...). This is deliberately
version-agnostic: we don't parse llama-server's log output, which changes.
(Its download progress bar is also TTY-only, so it never reaches us through a
pipe.) Download progress instead comes from watching the llama.cpp cache dir:
weights are downloaded to `<file>.downloadInProgress` and renamed when done, so
the temp file's size *is* the byte count. Totals come from the same Hugging
Face manifest endpoint llama.cpp itself resolves repo:tag through.
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QTimer, QUrl, pyqtSignal
from PyQt6.QtNetwork import (
    QNetworkAccessManager, QNetworkReply, QNetworkRequest,
)

from models import ModelPreset, build_command

_DL_SUFFIX = ".downloadInProgress"

# Written after every launch so a crashed session's llama-server (which would
# otherwise keep holding the port and VRAM) can be reclaimed on the next run.
PID_FILE = Path.home() / ".cache" / "backseat" / "llama-server.pid"


def llama_cache_dir() -> Path:
    """Replicate llama.cpp's fs_get_cache_dir() resolution order."""
    env = os.environ.get("LLAMA_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "llama.cpp"


class LlamaServerManager(QObject):
    status = pyqtSignal(str)     # human-readable state ("loading model…", "ready")
    ready = pyqtSignal(str)      # server is serving; payload = running model key
    stopped = pyqtSignal()       # process has exited
    failed = pyqtSignal(str)     # unexpected exit / launch failure, with detail
    # (filename, downloaded_bytes, total_bytes) — total is 0 when unknown
    download_progress = pyqtSignal(object)
    loading = pyqtSignal()       # weights are being loaded into memory (no
                                 # download active); re-emitted after a download

    def __init__(self, binary: str, port: int, parent: QObject | None = None):
        super().__init__(parent)
        self.binary = binary
        self.port = port
        self.proc: QProcess | None = None
        self.current_key: str | None = None
        self._pending: ModelPreset | None = None
        self._intentional_stop = False
        self._is_ready = False

        self._nam = QNetworkAccessManager(self)
        self._health = QTimer(self)
        self._health.setInterval(1500)
        self._health.timeout.connect(self._poll_health)

        # Download detection: poll the llama.cpp cache for *.downloadInProgress
        # while the server is starting up.
        self._dl_timer = QTimer(self)
        self._dl_timer.setInterval(500)
        self._dl_timer.timeout.connect(self._scan_downloads)
        self._file_sizes: dict[str, int] = {}  # manifest rfilename -> bytes
        self._downloading = False

    # -- state queries -------------------------------------------------------
    def is_ready(self) -> bool:
        return self._is_ready

    def is_busy(self) -> bool:
        """True while starting/loading/stopping (i.e. a transition in progress)."""
        return self.proc is not None and not self._is_ready

    # -- public control ------------------------------------------------------
    def start(self, preset: ModelPreset) -> None:
        """Start `preset`. If a server is already running, stop it first."""
        if self.proc is not None:
            self._pending = preset
            self.stop()  # async; _on_finished will launch the pending preset
            return
        self._launch(preset)

    def change_model(self, preset: ModelPreset) -> None:
        if preset.key == self.current_key and self._is_ready:
            return
        self.start(preset)

    def stop(self, wait: bool = False) -> None:
        """Terminate the running server. Pass wait=True on app shutdown."""
        self._health.stop()
        if self.proc is None:
            self.stopped.emit()
            return
        self._intentional_stop = True
        target = self.proc
        self.status.emit("stopping…")
        target.terminate()
        if wait:
            if not target.waitForFinished(8000):
                target.kill()
                target.waitForFinished(3000)
        else:
            # If it ignores SIGTERM, force-kill *that specific* process later.
            QTimer.singleShot(8000, lambda: self._force_kill(target))

    # -- internals -----------------------------------------------------------
    def _kill_stale_server(self) -> None:
        """Reclaim a llama-server left behind by a crashed session.

        Only acts on a pid we recorded ourselves, and only if that pid is
        still actually a llama-server (pids get recycled).
        """
        try:
            pid = int(PID_FILE.read_text())
        except (OSError, ValueError):
            return
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            cmdline = b""  # process already gone
        if b"llama-server" in cmdline:
            self.status.emit("stopping stale llama-server from a previous run…")
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(30):  # up to ~3s to free the port and VRAM
                    if not Path(f"/proc/{pid}").exists():
                        break
                    time.sleep(0.1)
                else:
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            PID_FILE.unlink()
        except OSError:
            pass

    def _launch(self, preset: ModelPreset) -> None:
        self._kill_stale_server()
        self._is_ready = False
        self._intentional_stop = False
        self.current_key = preset.key

        cmd = build_command(self.binary, self.port, preset)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )
        self.proc.finished.connect(self._on_finished)

        self.status.emit(f"starting {preset.label}…")
        self.proc.start(cmd[0], cmd[1:])
        if not self.proc.waitForStarted(5000):
            self.current_key = None
            self._cleanup()
            self.failed.emit(
                f"could not launch '{self.binary}'. Is llama.cpp installed and "
                f"on your PATH?"
            )
            return
        try:
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(str(self.proc.processId()))
        except OSError:
            pass  # crash recovery is best-effort; never block a launch on it
        self.status.emit("loading model (first run downloads weights)…")
        self._file_sizes = {}
        self._downloading = False
        self._fetch_manifest(preset)
        self._dl_timer.start()
        self.loading.emit()
        self._health.start()

    def _fetch_manifest(self, preset: ModelPreset) -> None:
        """Ask the HF manifest endpoint (the same one llama.cpp resolves
        repo:tag through) for file sizes, so download progress has a total."""
        repo, _, tag = preset.hf_repo.partition(":")
        url = f"https://huggingface.co/v2/{repo}/manifests/{tag or 'latest'}"
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"User-Agent", b"llama-cpp")
        token = os.environ.get("HF_TOKEN")
        if token:
            req.setRawHeader(b"Authorization", f"Bearer {token}".encode())
        reply = self._nam.get(req)
        reply.finished.connect(lambda r=reply: self._on_manifest(r))

    def _on_manifest(self, reply: QNetworkReply) -> None:
        data = bytes(reply.readAll())
        reply.deleteLater()
        try:
            manifest = json.loads(data)
            for entry_key in ("ggufFile", "mmprojFile"):
                entry = manifest.get(entry_key)
                if entry and entry.get("rfilename") and entry.get("size"):
                    name = entry["rfilename"].rsplit("/", 1)[-1]
                    self._file_sizes[name] = int(entry["size"])
        except (ValueError, TypeError, KeyError):
            pass  # no totals — the GUI falls back to an indeterminate bar

    def _scan_downloads(self) -> None:
        if self.proc is None or self._is_ready:
            self._dl_timer.stop()
            return
        try:
            cache = llama_cache_dir()
            temps = list(cache.rglob(f"*{_DL_SUFFIX}")) if cache.is_dir() else []
            if temps:
                # Downloads are sequential; the newest temp file is the active one.
                tmp = max(temps, key=lambda p: p.stat().st_mtime)
                done = tmp.stat().st_size
        except OSError:
            return  # scan raced the rename at download completion; next tick
        if temps:
            name = tmp.name[: -len(_DL_SUFFIX)]
            # Cache names embed the HF rfilename; match to get the total.
            total = next(
                (s for n, s in self._file_sizes.items() if n in name), 0
            )
            if not self._downloading:
                self._downloading = True
                self.status.emit("downloading weights…")
            self.download_progress.emit((name, done, total))
        elif self._downloading:
            self._downloading = False
            self.status.emit("loading model…")
            self.loading.emit()

    def _poll_health(self) -> None:
        req = QNetworkRequest(QUrl(f"http://localhost:{self.port}/health"))
        reply = self._nam.get(req)
        reply.finished.connect(lambda r=reply: self._on_health(r))

    def _on_health(self, reply: QNetworkReply) -> None:
        code = reply.attribute(
            QNetworkRequest.Attribute.HttpStatusCodeAttribute
        )
        err = reply.error()
        reply.deleteLater()
        if self._is_ready or self.proc is None:
            return
        if err == QNetworkReply.NetworkError.NoError and code == 200:
            self._is_ready = True
            self._health.stop()
            self._dl_timer.stop()
            self.status.emit("ready")
            self.ready.emit(self.current_key or "")

    def _on_finished(self, code: int, _status) -> None:
        was_intentional = self._intentional_stop
        tail = ""
        if self.proc is not None:
            out = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
            tail = "\n".join(out.strip().splitlines()[-4:])
        self._cleanup()
        self.stopped.emit()

        if was_intentional:
            if self._pending is not None:
                nxt, self._pending = self._pending, None
                self._launch(nxt)
        else:
            self.current_key = None
            msg = f"llama-server exited unexpectedly (code {code})."
            if tail:
                msg += f"\n…{tail}"
            self.failed.emit(msg)

    def _force_kill(self, target: QProcess) -> None:
        try:
            if target.state() != QProcess.ProcessState.NotRunning:
                target.kill()
        except RuntimeError:
            pass  # the QProcess was already deleted; nothing to do

    def _cleanup(self) -> None:
        if self.proc is not None:
            self.proc.deleteLater()
        self.proc = None
        self._is_ready = False
        self._dl_timer.stop()
        self._downloading = False
        try:
            PID_FILE.unlink()  # the process is gone; nothing stale to reclaim
        except OSError:
            pass
