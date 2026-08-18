"""PyQt6 GUI: a single window with everything in it.

Left column:
  * Local model server — pick a VLM, and Change model to swap it (the button is
    only enabled when a *different* model than the running one is selected). The
    default model launches automatically on startup.
  * Watch — pick what to observe and start/stop the co-work loop.
  * Last screenshot sent to the model.
Right column:
  * Suggestions — the model's comments accumulate here, timestamped.
With --debug, a third column logs every send and the raw reply (PASS included).

Keep this window on a monitor you're NOT capturing so the model doesn't end up
reacting to its own output.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from html import escape
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QSettings, Qt, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QProgressBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
    QWidget,
)

from backends import make_backend
from capture import list_monitors, list_windows
from config import Config
from models import DEFAULT_KEY, PRESETS, preset_by_key
from server import LlamaServerManager
from worker import WatchWorker

# Measured weight-load durations per model key, so the load bar can show a
# meaningful "elapsed / total" the second time around.
LOAD_TIMES_PATH = Path.home() / ".cache" / "backseat" / "load_times.json"


def _fmt_secs(s: float) -> str:
    s = max(0, int(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    return f"{n // (1 << 20)} MB"


class ControlPanel(QWidget):
    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        self.thread: QThread | None = None
        self.worker: WatchWorker | None = None
        # Threads whose backend call was still in flight when the user hit
        # Stop. Destroying a running QThread aborts the whole process, so we
        # keep (thread, worker) alive here until the thread actually finishes.
        self._zombies: list[tuple[QThread, WatchWorker]] = []
        self._sources: list = []

        # Static config supplies the server binary/port and defaults.
        self.defaults = Config()
        self.server = LlamaServerManager(
            self.defaults.llama_server_bin, self.defaults.server_port, parent=self
        )
        self._wire_server()

        self.setWindowTitle("Backseat")
        self.resize(420 * (3 if debug else 2), 720)
        self._build()
        self.refresh_sources()
        self._populate_models()

        if self.defaults.autostart_server:
            self.server.start(preset_by_key(DEFAULT_KEY))

    # -- layout --------------------------------------------------------------
    def _build(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(self._server_group())
        left.addWidget(self._watch_group())
        left.addWidget(self._preview_group(), 1)
        root.addLayout(left, 1)
        root.addWidget(self._suggestions_group(), 1)
        if self.debug:
            root.addWidget(self._debug_group(), 1)

    @staticmethod
    def _make_log() -> QTextEdit:
        log = QTextEdit(readOnly=True)
        log.setMinimumHeight(140)
        log.setStyleSheet(
            "QTextEdit { background:#1c1c20; color:#e8e8ea;"
            " border:none; padding:10px; font-size:13px; }"
        )
        return log

    @staticmethod
    def _append(log: QTextEdit, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        log.append(
            f'<div style="margin:6px 0;">'
            f'<span style="color:#8a8a92;font-size:11px;">{ts}</span><br>'
            f'{text}</div>'
        )
        log.verticalScrollBar().setValue(log.verticalScrollBar().maximum())

    def _suggestions_group(self) -> QGroupBox:
        box = QGroupBox("Suggestions")
        self.log = self._make_log()

        # Text size control; the choice is remembered across runs.
        self._settings = QSettings("backseat", "backseat")
        try:
            size = int(self._settings.value("suggestions_font_px", 13))
        except (TypeError, ValueError):
            size = 13
        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 32)
        self.font_spin.setSuffix(" px")
        self.font_spin.setValue(size)
        self.font_spin.valueChanged.connect(self._set_log_font)
        self._set_log_font(size)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Text size:"))
        size_row.addWidget(self.font_spin)
        size_row.addStretch(1)

        layout = QVBoxLayout(box)
        layout.addLayout(size_row)
        layout.addWidget(self.log)
        return box

    def _set_log_font(self, px: int) -> None:
        self.log.setStyleSheet(
            "QTextEdit { background:#1c1c20; color:#e8e8ea;"
            f" border:none; padding:10px; font-size:{px}px; }}"
        )
        self._settings.setValue("suggestions_font_px", px)

    def _debug_group(self) -> QGroupBox:
        box = QGroupBox("Debug")
        self.debug_log = self._make_log()
        layout = QVBoxLayout(box)
        layout.addWidget(self.debug_log)
        return box

    def add_comment(self, text: str) -> None:
        self._append(self.log, text)

    def add_debug(self, text: str) -> None:
        self._append(self.debug_log, text)

    def _server_group(self) -> QGroupBox:
        box = QGroupBox("Local model server")
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._update_change_btn)
        self.change_btn = QPushButton("Change model")
        self.change_btn.setEnabled(False)
        self.change_btn.clicked.connect(self._change_model)

        row = QHBoxLayout()
        row.addWidget(self.model_combo, 1)
        row.addWidget(self.change_btn)

        self.server_status = QLabel("server: idle")
        self.server_status.setStyleSheet("color:#888;")
        self.server_status.setWordWrap(True)

        # Progress rows, hidden except during their phase of a model start.
        self.dl_label = QLabel()
        self.dl_label.setStyleSheet("color:#888;")
        self.dl_bar = QProgressBar()
        self.load_label = QLabel()
        self.load_label.setStyleSheet("color:#888;")
        self.load_bar = QProgressBar()
        for w in (self.dl_label, self.dl_bar, self.load_label, self.load_bar):
            w.setVisible(False)

        self._dl_t0: float | None = None     # when this download session began
        self._dl_frac0 = 0.0                 # fraction already present (resume)
        self._dl_state: tuple[str, int, int] = ("", 0, 0)
        self._load_t0: float | None = None   # when weight-loading began
        self._load_expected: float | None = None
        self._load_times = self._read_load_times()
        self._bar_timer = QTimer(self)
        self._bar_timer.setInterval(500)
        self._bar_timer.timeout.connect(self._tick_bars)

        layout = QVBoxLayout(box)
        layout.addLayout(row)
        layout.addWidget(self.server_status)
        layout.addWidget(self.dl_label)
        layout.addWidget(self.dl_bar)
        layout.addWidget(self.load_label)
        layout.addWidget(self.load_bar)
        return box

    def _watch_group(self) -> QGroupBox:
        box = QGroupBox("Watch")

        self.source_box = QComboBox()
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self.refresh_sources)
        src_row = QHBoxLayout()
        src_row.addWidget(self.source_box, 1)
        src_row.addWidget(refresh_btn)

        self.backend_box = QComboBox()
        self.backend_box.addItems(["llamacpp", "claude"])
        self.backend_box.currentTextChanged.connect(self._on_backend_change)

        self.model_edit = QLineEdit(self.defaults.claude_model)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.5, 30.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(self.defaults.poll_interval)

        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.0, 120.0)
        self.cooldown_spin.setValue(self.defaults.min_seconds_between_calls)

        form = QFormLayout()
        form.addRow("Watch:", src_row)
        form.addRow("Backend:", self.backend_box)
        form.addRow("Claude model:", self.model_edit)
        form.addRow("Poll interval (s):", self.interval_spin)
        form.addRow("Min gap between calls (s):", self.cooldown_spin)

        self.start_btn = QPushButton("Start watching")
        self.start_btn.clicked.connect(self.toggle)
        self.nudge_btn = QPushButton("Ask now")
        self.nudge_btn.clicked.connect(self.nudge)
        self.nudge_btn.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn, 1)
        btn_row.addWidget(self.nudge_btn)

        self.status = QLabel("idle")
        self.status.setStyleSheet("color:#888;")

        layout = QVBoxLayout(box)
        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.status)

        self._on_backend_change(self.backend_box.currentText())
        return box

    def _preview_group(self) -> QGroupBox:
        box = QGroupBox("Last screenshot sent to the model")
        self._preview_pixmap: QPixmap | None = None
        self.preview = QLabel("(nothing sent yet)")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(160)
        self.preview.setStyleSheet("color:#888; background:#1c1c20;")
        layout = QVBoxLayout(box)
        layout.addWidget(self.preview)
        return box

    def _show_frame(self, img: Image.Image) -> None:
        """Render the PIL image the worker just sent to the backend."""
        qimg = QImage(
            img.tobytes("raw", "RGB"), img.width, img.height,
            img.width * 3, QImage.Format.Format_RGB888,
        ).copy()  # copy() detaches from the temporary bytes buffer
        self._preview_pixmap = QPixmap.fromImage(qimg)
        self._rescale_preview()

    def _rescale_preview(self) -> None:
        if self._preview_pixmap is None:
            return
        self.preview.setPixmap(
            self._preview_pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._rescale_preview()
        super().resizeEvent(event)

    # -- model server --------------------------------------------------------
    def _wire_server(self) -> None:
        self.server.status.connect(
            lambda s: self.server_status.setText(f"server: {s}")
        )
        self.server.ready.connect(self._on_server_ready)
        self.server.failed.connect(self._on_server_failed)
        self.server.stopped.connect(self._update_change_btn)
        self.server.stopped.connect(self._hide_progress)
        self.server.download_progress.connect(self._on_download_progress)
        self.server.loading.connect(self._on_loading)

    def _populate_models(self) -> None:
        for p in PRESETS:
            self.model_combo.addItem(p.label, p.key)
        idx = self.model_combo.findData(DEFAULT_KEY)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self._update_change_btn()

    def _update_change_btn(self) -> None:
        selected = self.model_combo.currentData()
        busy = self.server.is_busy()
        self.change_btn.setEnabled(
            selected is not None
            and selected != self.server.current_key
            and not busy
        )
        self.model_combo.setEnabled(not busy)

    def _change_model(self) -> None:
        key = self.model_combo.currentData()
        if key is None:
            return
        self.change_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.server.change_model(preset_by_key(key))

    def _on_server_ready(self, key: str) -> None:
        if self._load_t0 is not None and key:
            # Remember how long the load took for next time's total estimate.
            self._load_times[key] = time.monotonic() - self._load_t0
            self._write_load_times()
        self._hide_progress()
        idx = self.model_combo.findData(key)
        if idx >= 0:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(idx)
            self.model_combo.blockSignals(False)
        self._update_change_btn()

    def _on_server_failed(self, message: str) -> None:
        self._hide_progress()
        self.server_status.setText(f"server: error — {message}")
        self._update_change_btn()

    # -- start-up progress bars ------------------------------------------------
    def _on_download_progress(self, info: tuple[str, int, int]) -> None:
        name, done, total = info
        if self._dl_t0 is None or name != self._dl_state[0]:
            self._dl_t0 = time.monotonic()
            self._dl_frac0 = done / total if total else 0.0
        self._dl_state = (name, done, total)
        # Loading hasn't started while bytes are still arriving.
        self._load_t0 = None
        self.load_label.setVisible(False)
        self.load_bar.setVisible(False)
        if total:
            self.dl_bar.setRange(0, 100)
            self.dl_bar.setValue(int(100 * done / total))
        else:
            self.dl_bar.setRange(0, 0)  # size unknown: busy indicator
        self.dl_label.setVisible(True)
        self.dl_bar.setVisible(True)
        self._bar_timer.start()
        self._tick_bars()

    def _on_loading(self) -> None:
        self._dl_t0 = None
        self._dl_state = ("", 0, 0)
        self.dl_label.setVisible(False)
        self.dl_bar.setVisible(False)
        self._load_t0 = time.monotonic()
        self._load_expected = self._load_times.get(self.server.current_key or "")
        if self._load_expected:
            self.load_bar.setRange(0, 100)
            self.load_bar.setValue(0)
        else:
            self.load_bar.setRange(0, 0)  # first load of this model: unknown
        self.load_label.setVisible(True)
        self.load_bar.setVisible(True)
        self._bar_timer.start()
        self._tick_bars()

    def _tick_bars(self) -> None:
        now = time.monotonic()
        if self._dl_t0 is not None:
            name, done, total = self._dl_state
            elapsed = now - self._dl_t0
            frac = done / total if total else 0.0
            if total and frac > self._dl_frac0 and elapsed > 1.0:
                # Estimated total time for this session from the current rate.
                rate = (frac - self._dl_frac0) / elapsed
                total_time = (1.0 - self._dl_frac0) / rate
                timing = f"{_fmt_secs(elapsed)} / ~{_fmt_secs(max(total_time, elapsed))}"
            else:
                timing = f"{_fmt_secs(elapsed)} / --:--"
            size = f"{_fmt_bytes(done)} / {_fmt_bytes(total)}" if total else _fmt_bytes(done)
            self.dl_label.setText(f"Downloading {name} — {size} — {timing}")
        if self._load_t0 is not None:
            elapsed = now - self._load_t0
            if self._load_expected:
                self.load_bar.setValue(
                    min(99, int(100 * elapsed / self._load_expected))
                )
                expected = max(self._load_expected, elapsed)
                timing = f"{_fmt_secs(elapsed)} / ~{_fmt_secs(expected)}"
            else:
                timing = f"{_fmt_secs(elapsed)} / --:--"
            self.load_label.setText(f"Loading weights — {timing}")

    def _hide_progress(self) -> None:
        self._bar_timer.stop()
        self._dl_t0 = None
        self._load_t0 = None
        for w in (self.dl_label, self.dl_bar, self.load_label, self.load_bar):
            w.setVisible(False)

    def _read_load_times(self) -> dict[str, float]:
        try:
            data = json.loads(LOAD_TIMES_PATH.read_text())
            return {k: float(v) for k, v in data.items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def _write_load_times(self) -> None:
        try:
            LOAD_TIMES_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOAD_TIMES_PATH.write_text(json.dumps(self._load_times))
        except OSError:
            pass  # history is a nicety; never let it break the app

    # -- watch sources -------------------------------------------------------
    def refresh_sources(self) -> None:
        self._sources = list_monitors() + list_windows()
        self.source_box.clear()
        for s in self._sources:
            self.source_box.addItem(s.label)
        if not self._sources:
            self.source_box.addItem("(no capture sources found)")

    def _on_backend_change(self, name: str) -> None:
        is_claude = name == "claude"
        self.model_edit.setEnabled(is_claude)
        self.model_edit.setText(
            self.defaults.claude_model if is_claude else "(managed above)"
        )

    def _config(self) -> Config:
        cfg = Config()
        cfg.backend = self.backend_box.currentText()
        if cfg.backend == "claude":
            cfg.claude_model = self.model_edit.text().strip()
        else:
            cfg.llamacpp_model = self.server.current_key or "local-vlm"
        cfg.poll_interval = self.interval_spin.value()
        cfg.min_seconds_between_calls = self.cooldown_spin.value()
        return cfg

    # -- watch control -------------------------------------------------------
    def toggle(self) -> None:
        if self.worker is None:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if not self._sources:
            self.status.setText("no capture source available")
            return
        cfg = self._config()
        if cfg.backend == "llamacpp" and not self.server.is_ready():
            self.status.setText("waiting for the local model server to be ready…")
            return
        region = self._sources[self.source_box.currentIndex()].region
        try:
            backend = make_backend(cfg)
        except Exception as e:  # noqa: BLE001
            self.status.setText(str(e))
            return

        self.thread = QThread()
        self.worker = WatchWorker(cfg, region, backend)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.comment.connect(self.add_comment)
        self.worker.frame.connect(self._show_frame)
        if self.debug:
            self.worker.sent.connect(lambda: self.add_debug("message sent"))
            self.worker.replied.connect(
                lambda t: self.add_debug(f"reply: {escape(t)}")
            )
        self.worker.status.connect(self.status.setText)
        self.worker.error.connect(self.status.setText)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

        self.start_btn.setText("Stop")
        self.nudge_btn.setEnabled(True)

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
        if self.thread and self.worker:
            thread, worker = self.thread, self.worker
            thread.quit()
            if not thread.wait(200):
                # A backend call is still in flight (they can take minutes).
                # Detach the worker from the UI and park the pair; _reap drops
                # the references once the thread really finishes.
                for sig in (worker.comment, worker.frame, worker.sent,
                            worker.replied, worker.status, worker.error):
                    try:
                        sig.disconnect()
                    except TypeError:
                        pass  # already had no connections
                self._zombies.append((thread, worker))
                thread.finished.connect(
                    lambda t=thread, w=worker: self._reap(t, w)
                )
        self.worker = None
        self.thread = None
        self.start_btn.setText("Start watching")
        self.nudge_btn.setEnabled(False)
        self.status.setText("idle")

    def _reap(self, thread: QThread, worker: WatchWorker) -> None:
        try:
            self._zombies.remove((thread, worker))
        except ValueError:
            pass
        thread.deleteLater()

    def nudge(self) -> None:
        if self.worker:
            self.worker.nudge()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.stop()
        self.server.stop(wait=True)  # free VRAM before we exit
        # Give parked threads a moment; past that, force them down so Qt
        # doesn't abort on a QThread that's destroyed while running.
        for thread, _worker in list(self._zombies):
            if not thread.wait(2000):
                thread.terminate()  # we're exiting; the HTTP reply is moot
                thread.wait(1000)
        self._zombies.clear()
        super().closeEvent(event)
