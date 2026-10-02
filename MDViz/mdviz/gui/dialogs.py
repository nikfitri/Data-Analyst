"""Dialogs: open files, export image, export movie, interaction occupancy."""
from __future__ import annotations

import glob
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QProgressDialog, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)
from PySide6.QtGui import QColor

from ..chem import INTERACTION_COLORS
from ..system import STRUCTURE_FORMATS, TRAJECTORY_FORMATS

STRUCT_FILTER = "Structure / topology (" + " ".join(f"*.{e}" for e in STRUCTURE_FORMATS) + ");;All files (*)"
TRAJ_FILTER = "Trajectory (" + " ".join(f"*.{e}" for e in TRAJECTORY_FORMATS) + ");;All files (*)"


def _progress(parent, title, text):
    dlg = QProgressDialog(text, "Cancel", 0, 100, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(True)
    dlg.setValue(0)

    def cb(i, n):
        dlg.setMaximum(n)
        dlg.setValue(i)
        dlg.setLabelText(f"{text}  {i}/{n}")
        QApplication.processEvents()
        return not dlg.wasCanceled()
    return dlg, cb


class OpenDialog(QDialog):
    """Choose a structure/topology plus any number of trajectory files (like VMD's New Molecule)."""

    def __init__(self, parent=None, start_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Open structure and trajectory")
        self.resize(640, 360)
        self.start_dir = start_dir
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.top = QLineEdit()
        self.top.setPlaceholderText("Structure / topology: .tpr .gro .pdb .psf .prmtop .mol2 .cif .sdf …")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_top)
        row.addWidget(self.top)
        row.addWidget(b)
        lay.addWidget(QLabel("<b>1. Structure / topology</b> (.tpr or .psf/.prmtop give real bonds and masses)"))
        lay.addLayout(row)
        lay.addWidget(QLabel("<b>2. Trajectories</b> (optional; concatenated in order)"))
        self.trajs = QListWidget()
        self.trajs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.trajs)
        r2 = QHBoxLayout()
        for text, fn in (("Add…", self._add_traj), ("Remove", self._remove), ("Suggest from folder", self._suggest)):
            bb = QPushButton(text)
            bb.clicked.connect(fn)
            r2.addWidget(bb)
        r2.addStretch()
        lay.addLayout(r2)
        self.pbc = QCheckBox("Fix periodic boundaries on load (for raw, un-centred GROMACS trajectories)")
        self.align = QCheckBox("Fit every frame to frame 0 (protein CA)")
        lay.addWidget(self.pbc)
        lay.addWidget(self.align)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse_top(self):
        p, _ = QFileDialog.getOpenFileName(self, "Structure / topology", self.start_dir, STRUCT_FILTER)
        if p:
            self.top.setText(p)
            self.start_dir = os.path.dirname(p)
            if self.trajs.count() == 0:
                self._suggest()

    def _add_traj(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Trajectories", self.start_dir, TRAJ_FILTER)
        for p in ps:
            self.trajs.addItem(p)

    def _remove(self):
        for it in self.trajs.selectedItems():
            self.trajs.takeItem(self.trajs.row(it))

    def _suggest(self):
        top = self.top.text().strip()
        if not top:
            return
        folder = os.path.dirname(top)
        stem = os.path.splitext(os.path.basename(top))[0]
        cands = [p for ext in ("xtc", "trr", "dcd", "nc") for p in glob.glob(os.path.join(folder, f"*.{ext}"))]
        cands = [p for p in cands if os.path.basename(p) != ".xtc"]
        same = [p for p in cands if os.path.splitext(os.path.basename(p))[0] == stem]
        pick = same or sorted(cands, key=os.path.getsize, reverse=True)[:1]
        existing = {self.trajs.item(i).text() for i in range(self.trajs.count())}
        for p in pick:
            if p not in existing:
                self.trajs.addItem(p)

    def _ok(self):
        if not os.path.isfile(self.top.text().strip()):
            QMessageBox.warning(self, "Open", "Please choose an existing structure/topology file.")
            return
        self.accept()

    def result_files(self):
        return self.top.text().strip(), [self.trajs.item(i).text() for i in range(self.trajs.count())]


SIZE_PRESETS = [
    ("Current window aspect (3000 px wide)", None),
    ("Full HD 1920 × 1080", (1920, 1080)),
    ("4K UHD 3840 × 2160", (3840, 2160)),
    ("Square 3000 × 3000", (3000, 3000)),
    ("Single column 3.5 in @ 600 dpi (2100 × 1575)", (2100, 1575)),
    ("Double column 7 in @ 300 dpi (2100 × 1400)", (2100, 1400)),
    ("Poster 6000 × 4000", (6000, 4000)),
    ("Custom", "custom"),
]


class ExportImageDialog(QDialog):
    def __init__(self, parent, win_aspect: float, frame: int, n_frames: int, start_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Export image (PNG / JPEG / TIFF)")
        self.win_aspect = win_aspect
        f = QFormLayout(self)
        row = QHBoxLayout()
        self.path = QLineEdit(os.path.join(start_dir, "figure.png"))
        b = QPushButton("…")
        b.clicked.connect(self._browse)
        row.addWidget(self.path)
        row.addWidget(b)
        f.addRow("File", row)
        self.preset = QComboBox()
        for name, _ in SIZE_PRESETS:
            self.preset.addItem(name)
        self.preset.currentIndexChanged.connect(self._preset)
        f.addRow("Size preset", self.preset)
        self.w = QSpinBox()
        self.h = QSpinBox()
        for s in (self.w, self.h):
            s.setRange(64, 8192)
            s.setSuffix(" px")
        wh = QHBoxLayout()
        wh.addWidget(self.w)
        wh.addWidget(QLabel("×"))
        wh.addWidget(self.h)
        f.addRow("Width × height", wh)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        f.addRow("DPI (metadata)", self.dpi)
        self.ss = QComboBox()
        self.ss.addItems(["1× (fast)", "2× (recommended)", "3× (smoothest)"])
        self.ss.setCurrentIndex(1)
        f.addRow("Supersampling", self.ss)
        self.transparent = QCheckBox("Transparent background (PNG/TIFF)")
        f.addRow(self.transparent)
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(95)
        f.addRow("JPEG quality", self.quality)
        self.frame = QSpinBox()
        self.frame.setRange(0, max(0, n_frames - 1))
        self.frame.setValue(frame)
        f.addRow("Frame", self.frame)
        note = QLabel("Text and label sizes scale with image height, so the export matches the screen layout.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        f.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._preset(0)

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save image", self.path.text(),
                                           "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)")
        if p:
            self.path.setText(p)

    def _preset(self, i):
        val = SIZE_PRESETS[i][1]
        custom = val == "custom"
        self.w.setEnabled(custom)
        self.h.setEnabled(custom)
        if val is None:
            self.w.setValue(3000)
            self.h.setValue(int(round(3000 / max(self.win_aspect, 0.1))))
        elif not custom:
            self.w.setValue(val[0])
            self.h.setValue(val[1])

    def values(self):
        return dict(path=self.path.text().strip(), width=self.w.value(), height=self.h.value(), dpi=self.dpi.value(),
                    supersample=self.ss.currentIndex() + 1, transparent=self.transparent.isChecked(),
                    quality=self.quality.value(), frame=self.frame.value())


class ExportMovieDialog(QDialog):
    def __init__(self, parent, n_frames: int, dt_ps: float, saved_views: list, start_dir="", win_aspect=16 / 9):
        super().__init__(parent)
        self.setWindowTitle("Export movie (MP4, H.264)")
        self.n_frames = n_frames
        self.dt = dt_ps
        self.win_aspect = win_aspect
        f = QFormLayout(self)
        row = QHBoxLayout()
        self.path = QLineEdit(os.path.join(start_dir, "movie.mp4"))
        b = QPushButton("…")
        b.clicked.connect(self._browse)
        row.addWidget(self.path)
        row.addWidget(b)
        f.addRow("File", row)
        self.mode = QComboBox()
        self.mode.addItems(["Trajectory (play the simulation)", "Turntable (rotate the current frame 360°)"])
        f.addRow("Movie type", self.mode)
        self.res = QComboBox()
        self.res_values = [(1920, 1080), (1280, 720), (3840, 2160), (1080, 1080), (2160, 2160), None]
        self.res.addItems(["1080p  1920 × 1080", "720p  1280 × 720", "4K  3840 × 2160", "Square 1080 × 1080",
                           "Square 2160 × 2160", "Current window aspect (1080 px high)"])
        f.addRow("Resolution", self.res)
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(30)
        f.addRow("Frames per second", self.fps)
        fr = QHBoxLayout()
        self.start = QSpinBox()
        self.stop = QSpinBox()
        self.stride = QSpinBox()
        for s in (self.start, self.stop):
            s.setRange(0, max(0, n_frames - 1))
        self.stop.setValue(max(0, n_frames - 1))
        self.stride.setRange(1, max(1, n_frames))
        fr.addWidget(QLabel("from"))
        fr.addWidget(self.start)
        fr.addWidget(QLabel("to"))
        fr.addWidget(self.stop)
        fr.addWidget(QLabel("every"))
        fr.addWidget(self.stride)
        f.addRow("Trajectory frames", fr)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-1080, 1080)
        self.spin.setSuffix("°")
        self.spin.setToolTip("Rotate the camera about the vertical axis over the course of the movie.\n"
                             "Turntable mode defaults to 360°.")
        f.addRow("Camera spin", self.spin)
        self.turn_secs = QDoubleSpinBox()
        self.turn_secs.setRange(1, 120)
        self.turn_secs.setValue(12)
        self.turn_secs.setSuffix(" s")
        f.addRow("Turntable duration", self.turn_secs)
        vv = QHBoxLayout()
        self.view_from = QComboBox()
        self.view_to = QComboBox()
        for c in (self.view_from, self.view_to):
            c.addItem("(current view)")
            c.addItems(saved_views)
        vv.addWidget(self.view_from)
        vv.addWidget(QLabel("→"))
        vv.addWidget(self.view_to)
        f.addRow("Camera fly-through", vv)
        self.quality = QComboBox()
        self.quality_values = [16, 12, 20, 0]
        self.quality.addItems(["High (CRF 16)", "Very high (CRF 12)", "Standard (CRF 20, smaller file)",
                               "Lossless (CRF 0, huge)"])
        f.addRow("Quality", self.quality)
        self.ss = QComboBox()
        self.ss.addItems(["1× (fast)", "2× (smoother edges, slower)"])
        f.addRow("Supersampling", self.ss)
        self.hold = QDoubleSpinBox()
        self.hold.setRange(0, 10)
        self.hold.setSuffix(" s")
        f.addRow("Hold last frame", self.hold)
        self.info = QLabel()
        self.info.setStyleSheet("color: #555;")
        f.addRow(self.info)
        for w in (self.start, self.stop, self.stride, self.fps):
            w.valueChanged.connect(self._update_info)
        self.mode.currentIndexChanged.connect(self._update_info)
        self.turn_secs.valueChanged.connect(self._update_info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        if n_frames <= 1:
            self.mode.setCurrentIndex(1)
        self._update_info()

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save movie", self.path.text(), "MP4 video (*.mp4)")
        if p:
            self.path.setText(p if p.lower().endswith(".mp4") else p + ".mp4")

    def _update_info(self):
        if self.mode.currentIndex() == 1:
            n = int(self.turn_secs.value() * self.fps.value())
            self.info.setText(f"{n} rendered frames · {n / self.fps.value():.1f} s of video")
        else:
            n = len(range(self.start.value(), self.stop.value() + 1, self.stride.value()))
            span = (self.stop.value() - self.start.value()) * self.dt / 1000
            self.info.setText(f"{n} rendered frames · {n / self.fps.value():.1f} s of video · covers {span:.2f} ns")

    def options(self):
        from ..render import MovieOptions
        wh = self.res_values[self.res.currentIndex()]
        if wh is None:
            wh = (int(round(1080 * self.win_aspect / 2)) * 2, 1080)
        return MovieOptions(width=wh[0], height=wh[1], fps=self.fps.value(), start=self.start.value(),
                            stop=self.stop.value(), stride=self.stride.value(),
                            mode="turntable" if self.mode.currentIndex() == 1 else "trajectory",
                            spin_degrees=self.spin.value(), turntable_seconds=self.turn_secs.value(),
                            view_from=self.view_from.currentText() if self.view_from.currentIndex() > 0 else "",
                            view_to=self.view_to.currentText() if self.view_to.currentIndex() > 0 else "",
                            crf=self.quality_values[self.quality.currentIndex()],
                            supersample=self.ss.currentIndex() + 1, hold_last=self.hold.value())


class OccupancyDialog(QDialog):
    """Run interaction detection across the trajectory; show, export CSV and publication plots."""

    def __init__(self, parent, system, detector, start_dir="", title=""):
        super().__init__(parent)
        self.setWindowTitle("Interaction occupancy over the trajectory")
        self.resize(760, 620)
        self.sys = system
        self.det = detector
        self.start_dir = start_dir
        self.title = title
        self.result = None
        self.worker = None
        self._closing = False
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        n = system.n_frames
        self.start = QSpinBox()
        self.stop = QSpinBox()
        self.stride = QSpinBox()
        for s in (self.start, self.stop):
            s.setRange(0, max(0, n - 1))
        self.stop.setValue(max(0, n - 1))
        self.stride.setRange(1, max(1, n))
        self.stride.setValue(max(1, n // 500))
        for lbl, w in (("Frames from", self.start), ("to", self.stop), ("every", self.stride)):
            row.addWidget(QLabel(lbl))
            row.addWidget(w)
        run = QPushButton("Run analysis")
        self.run_button = run
        run.clicked.connect(self.run)
        row.addWidget(run)
        row.addStretch()
        lay.addLayout(row)
        self.summary = QLabel("Analysis uses unsmoothed frames. Set every = 1 to include all frames.")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Type", "Residue", "Occupancy (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        lay.addWidget(self.table)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Plot residues with occupancy ≥"))
        self.min_occ = QDoubleSpinBox()
        self.min_occ.setRange(0, 100)
        self.min_occ.setValue(10)
        self.min_occ.setSuffix(" %")
        opts.addWidget(self.min_occ)
        opts.addStretch()
        lay.addLayout(opts)
        btns = QHBoxLayout()
        for text, fn in (("Save table (CSV)…", self.save_csv), ("Save per-frame timeline (CSV)…", self.save_timeline_csv),
                         ("Save occupancy plot…", self.save_bar), ("Save timeline plot…", self.save_timeline)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)
        self.timeline_button = QPushButton("Show clickable timeline in viewer")
        self.timeline_button.setEnabled(False)
        self.timeline_button.clicked.connect(self.accept)
        lay.addWidget(self.timeline_button)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        lay.addWidget(close)

    def run(self):
        from .analysis_worker import OccupancyWorker
        frames = list(range(self.start.value(), self.stop.value() + 1, self.stride.value()))
        if not frames:
            QMessageBox.warning(self, "Occupancy", "Start frame must not exceed end frame.")
            return
        if self.worker is not None:
            return
        self.run_button.setEnabled(False)
        self.timeline_button.setEnabled(False)
        self.result = None
        self.table.setRowCount(0)
        self.job_progress = QProgressDialog("Loading independent trajectory reader…", "Cancel", 0, len(frames), self)
        self.job_progress.setWindowModality(Qt.WindowModal)
        self.job_progress.setAutoClose(False)
        self.job_progress.setMinimumDuration(0)
        self.worker = OccupancyWorker(self.sys, self.det, frames, self)
        self.job_progress.canceled.connect(self.worker.requestInterruption)
        self.worker.progress.connect(self._progress_job)
        self.worker.resultReady.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.worker.start()
        self.job_progress.show()

    def _progress_job(self, i, n):
        if not self.job_progress.wasCanceled():
            self.job_progress.setValue(i)
            self.job_progress.setLabelText(f"Analysing unsmoothed frames… {i}/{n}")

    def _failed(self, message):
        self.summary.setText("Analysis failed: " + message)

    def _finished(self):
        self.job_progress.close()
        self.worker.deleteLater()
        self.worker = None
        self.run_button.setEnabled(True)
        self.timeline_button.setEnabled(self.result is not None)
        if self.result is None and not self.summary.text().startswith("Analysis failed"):
            self.summary.setText("Analysis cancelled before any frames were completed.")
        if self._closing:
            super().reject()

    def reject(self):
        if self.worker is not None:
            self._closing = True
            self.worker.requestInterruption()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

    def _completed(self, result):
        self.result = result
        rows = self.result.residue_occupancy()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, ((kind, res), occ) in enumerate(rows):
            it = QTableWidgetItem(kind)
            it.setForeground(QColor(INTERACTION_COLORS[kind]))
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(res))
            oc = QTableWidgetItem()
            oc.setData(Qt.DisplayRole, round(float(occ), 1))
            self.table.setItem(r, 2, oc)
        self.table.setSortingEnabled(True)
        t0, t1 = result.times_ps[0] / 1000, result.times_ps[-1] / 1000
        prefix = "Partial result (cancelled) · " if result.metadata.get("partial") else ""
        self.summary.setText(prefix + f"{len(result.frames)} frames analysed ({t0:.2f}–{t1:.2f} ns) · "
                             f"{len(rows)} residue-level interactions · unsmoothed coordinates")

    def _need(self):
        if self.result is None:
            QMessageBox.information(self, "Occupancy", "Run the analysis first.")
            return False
        return True

    def _save(self, caption, default, flt):
        p, _ = QFileDialog.getSaveFileName(self, caption, os.path.join(self.start_dir, default), flt)
        return p

    def save_csv(self):
        if self._need() and (p := self._save("Save table", "interaction_occupancy.csv", "CSV (*.csv)")):
            self.result.to_csv(p)

    def save_timeline_csv(self):
        if self._need() and (p := self._save("Save timeline", "interaction_timeline.csv", "CSV (*.csv)")):
            self.result.timeline_csv(p)

    def save_bar(self):
        from ..plots import plot_occupancy
        if self._need() and (p := self._save("Save plot", "interaction_occupancy.png",
                                             "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tif)")):
            plot_occupancy(self.result, p, self.min_occ.value(), title=self.title)

    def save_timeline(self):
        from ..plots import plot_timeline
        if self._need() and (p := self._save("Save plot", "interaction_timeline.png",
                                             "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tif)")):
            plot_timeline(self.result, p, self.min_occ.value(), title=self.title)
