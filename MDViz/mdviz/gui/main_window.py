"""MDViz main window."""
from __future__ import annotations

import os
import tempfile
import time
import traceback

import numpy as np

import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PySide6"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera  # noqa: E402
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget  # noqa: E402
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor  # noqa: E402
from vtkmodules.vtkRenderingCore import vtkRenderer  # noqa: E402

from PySide6.QtCore import QEvent, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QAction, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,  # noqa: E402
                               QInputDialog, QLabel, QMainWindow, QMessageBox, QScrollArea, QTabWidget, QToolBar,
                               QVBoxLayout, QWidget)

from .. import __version__  # noqa: E402
from ..render import render_image, render_movie  # noqa: E402
from ..scene import Scene, _apply_material  # noqa: E402
from ..state import Session, ViewState, default_reps  # noqa: E402
from ..system import MolSystem, STRUCTURE_FORMATS, TRAJECTORY_FORMATS  # noqa: E402
from .pes_panel import PESPanel  # noqa: E402
from .pes_view import PESPlotView  # noqa: E402
from .contact_timeline import ContactTimeline  # noqa: E402
from .dialogs import ExportImageDialog, ExportMovieDialog, OccupancyDialog, OpenDialog, _progress  # noqa: E402
from .widgets import DisplayPanel, InteractionPanel, RepPanel, SystemPanel, TrajectoryBar  # noqa: E402

HELP_TEXT = """<h3>Mouse</h3>
<b>Left-drag</b> rotate · <b>Right-drag / wheel</b> zoom · <b>Middle-drag</b> or <b>Shift+left-drag</b> pan ·
<b>Ctrl+left-drag</b> roll about the screen axis<br>
<b>Ctrl+click</b> an atom: action depends on the <i>Pick</i> mode in the toolbar (info, label, distance, centre).
<h3>Keyboard</h3>
<b>Space</b> play/pause · <b>← / →</b> previous/next frame · <b>Home/End</b> first/last frame<br>
<b>+ / −</b> zoom · <b>R</b> reset view · <b>F</b> focus ligand · <b>Esc</b> clear labels and measurements
<h3>Selections</h3>
VMD/MDAnalysis syntax, e.g. <code>protein and resid 100 to 120</code>, <code>resname UNL</code>,
<code>protein and within 5 of ligand</code>, <code>byres (around 4 ligand)</code>, <code>name CA</code>.<br>
Macros: <code>ligand</code>, <code>water</code>, <code>ions</code>, <code>solute</code>, <code>heavy</code>,
<code>hydrogen</code>, <code>polarH</code>, <code>nonpolarH</code>.
<h3>Publication tips</h3>
Use <b>Display ▸ Clipping slab</b> to look into buried pockets; enable <b>Remove rotation/translation</b>
and a small <b>Smoothing window</b> (3–5) for calm movies; export stills at 2× supersampling.
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"MDViz {__version__} — molecular dynamics visualisation")
        self.resize(1500, 950)
        self.system: MolSystem | None = None
        self.session = Session()
        self.scene: Scene | None = None
        self.session_path = ""
        self.pick_buffer = []
        self._drag_label = None
        self.last_dir = os.getcwd()

        # ---- central 3D view + trajectory bar
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.vtk = QVTKRenderWindowInteractor(central)
        self.ren = vtkRenderer()
        self.renwin = self.vtk.GetRenderWindow()
        self.renwin.AddRenderer(self.ren)
        self.renwin.SetMultiSamples(0)
        self.renwin.SetAlphaBitPlanes(1)
        self.ren.SetBackground(1, 1, 1)
        self.style = vtkInteractorStyleTrackballCamera()
        self.vtk.SetInteractorStyle(self.style)
        self.style.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.style.AddObserver("CharEvent", lambda o, e: None)  # disable VTK single-key shortcuts
        self.style.AddObserver("KeyPressEvent", lambda o, e: None)
        self.vtk.installEventFilter(self)
        cl.addWidget(self.vtk, 1)
        self.traj = TrajectoryBar()
        self.traj.frameChanged.connect(self.set_frame)
        self.traj.playbackChanged.connect(self._playback_quality)
        self.traj.setEnabled(False)
        cl.addWidget(self.traj)
        self.setCentralWidget(central)
        self.axes_widget = None

        # ---- docks
        self.rep_panel = RepPanel()
        self.rep_panel.repChanged.connect(self._rep_changed)
        self.rep_panel.repAdded.connect(self._rep_added)
        self.rep_panel.repRemoved.connect(self._rep_removed)
        self.rep_panel.repsReordered.connect(self._rebuild_scene)
        self.display_panel = DisplayPanel()
        self.display_panel.changed.connect(self._display_changed)
        self.system_panel = SystemPanel()
        self.system_panel.changed.connect(self._processing_changed)
        left_tabs = QTabWidget()
        for w, name in ((self.rep_panel, "Representations"), (self.display_panel, "Display"),
                        (self.system_panel, "System")):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(w)
            left_tabs.addTab(sa, name)
        dock = QDockWidget("Scene", self)
        dock.setWidget(left_tabs)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.inter_panel = InteractionPanel()
        self.inter_panel.changed.connect(self._interactions_changed)
        self.inter_panel.occupancyRequested.connect(self.occupancy)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(self.inter_panel)
        dock2 = QDockWidget("Interactions", self)
        dock2.setWidget(sa)
        dock2.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        dock2.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock2)
        self.inter_dock = dock2

        # ---- potential energy surface: controls (tab beside Interactions) + interactive plots (bottom)
        self.pes_view = PESPlotView()
        self.pes_panel = PESPanel(self.pes_view)
        self.pes_panel.resultChanged.connect(self._pes_result_changed)
        self.pes_panel.pointRequested.connect(self._pes_point)
        self.pes_panel.showMDRequested.connect(self._restore_md)
        # the MD system for ligand torsion scans: the open trajectory (or the one parked behind the PES view)
        self.pes_panel.md_system_fn = lambda: ((self._md_stash[0] if self._md_stash else None)
                                               if self._pes_mode else self.system)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(self.pes_panel)
        self.pes_dock = QDockWidget("PES (φ, ψ)", self)
        self.pes_dock.setWidget(sa)
        self.pes_dock.setFeatures(dock2.features())
        self.addDockWidget(Qt.RightDockWidgetArea, self.pes_dock)
        self.tabifyDockWidget(dock2, self.pes_dock)
        dock2.raise_()
        self.pes_plot_dock = QDockWidget("Potential energy surface — hover or click to show the conformation", self)
        self.pes_plot_dock.setWidget(self.pes_view)
        self.pes_plot_dock.setFeatures(dock2.features())
        self.pes_plot_dock.setMinimumHeight(300)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.pes_plot_dock)
        self.pes_plot_dock.hide()
        self.contact_timeline = ContactTimeline()
        self.contact_timeline.frameRequested.connect(self._timeline_frame)
        self.contact_dock = QDockWidget("Contact timeline", self)
        self.contact_dock.setWidget(self.contact_timeline)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.contact_dock)
        self.contact_dock.hide()
        self._pes_mode = False
        self._pes_res = None
        self._md_stash = None

        self._build_menus()
        self._build_toolbar()
        self._shortcuts()
        self.status = QLabel("Open a structure/trajectory to begin (File ▸ Open, or drag files onto the window).")
        self.statusBar().addWidget(self.status, 1)
        self.setAcceptDrops(True)
        self._set_enabled(False)

    # ================================================================== UI construction
    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self._act(m, "Open structure / trajectory…", self.open_dialog, QKeySequence.Open)
        self.act_addtraj = self._act(m, "Add trajectory to current system…", self.add_trajectory)
        m.addSeparator()
        self._act(m, "Open session…", self.open_session)
        self._act(m, "Open potential energy surface (.npz / .csv)…", self.open_pes)
        self.act_save = self._act(m, "Save session", self.save_session, QKeySequence.Save)
        self.act_saveas = self._act(m, "Save session as…", lambda: self.save_session(True))
        m.addSeparator()
        self.act_img = self._act(m, "Export image (PNG/JPEG/TIFF)…", self.export_image, "Ctrl+E")
        self.act_mov = self._act(m, "Export movie (MP4)…", self.export_movie, "Ctrl+M")
        m.addSeparator()
        self._act(m, "Quit", self.close, QKeySequence.Quit)
        v = mb.addMenu("&View")
        self._act(v, "Reset view", self.reset_view)
        self._act(v, "Focus ligand", self.focus_ligand)
        self._act(v, "Focus on selection…", self.focus_selection)
        self.act_ortho = self._act(v, "Orthographic projection", self.toggle_ortho)
        self.act_ortho.setCheckable(True)
        self.act_ortho.setChecked(True)
        self.act_axes = self._act(v, "Show orientation axes", self.toggle_axes)
        self.act_axes.setCheckable(True)
        v.addSeparator()
        self._act(v, "Save current view as…", self.save_view)
        self._act(v, "Delete saved view…", self.delete_view)
        v.addSeparator()
        sm = v.addMenu("Style preset")
        from ..state import STYLE_PRESETS
        for name in STYLE_PRESETS:
            self._act(sm, name, lambda n=name: self.apply_style(n))
        a = mb.addMenu("&Analysis")
        self._act(a, "Interaction occupancy over trajectory…", self.occupancy)
        self._act(a, "Show interactions panel", lambda: self.inter_dock.show())
        self._act(a, "Show contact timeline", lambda: self.contact_dock.show())
        a.addSeparator()
        self._act(a, "Potential energy surface (φ/ψ scan)…", self.show_pes)
        h = mb.addMenu("&Help")
        self._act(h, "Controls and tips", lambda: QMessageBox.information(self, "MDViz controls", HELP_TEXT))
        self._act(h, "About", lambda: QMessageBox.about(
            self, "About MDViz", f"MDViz {__version__}<br>Reads VMD-compatible formats via MDAnalysis; renders "
                                 "with VTK; movies encoded with ffmpeg (H.264)."))

    def _act(self, menu, text, fn, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(lambda checked=False: fn())
        if shortcut:
            a.setShortcut(shortcut)
        menu.addAction(a)
        return a

    def _build_toolbar(self):
        tb = QToolBar("View")
        tb.setMovable(False)
        self.addToolBar(tb)
        self._tb_btn(tb, "Open", self.open_dialog, "Open structure / trajectory")
        tb.addSeparator()
        self._tb_btn(tb, "Zoom +", lambda: self.zoom(1.25), "Zoom in (+)")
        self._tb_btn(tb, "Zoom −", lambda: self.zoom(0.8), "Zoom out (−)")
        tb.addSeparator()
        tb.addWidget(QLabel(" Rotate "))
        self.rot_step = QDoubleSpinBox()
        self.rot_step.setRange(1, 180)
        self.rot_step.setValue(15)
        self.rot_step.setSuffix("°")
        self.rot_step.setToolTip("Rotation step for the buttons")
        tb.addWidget(self.rot_step)
        for axis in ("X", "Y", "Z"):
            self._tb_btn(tb, f"{axis}−", lambda a=axis: self.rotate(a, -1), f"Rotate about screen {axis} axis")
            self._tb_btn(tb, f"{axis}+", lambda a=axis: self.rotate(a, 1), f"Rotate about screen {axis} axis")
        tb.addSeparator()
        self._tb_btn(tb, "Reset", self.reset_view, "Reset view (R)")
        self._tb_btn(tb, "Focus ligand", self.focus_ligand, "Centre and zoom on the ligand (F)")
        tb.addSeparator()
        tb.addWidget(QLabel(" View "))
        self.views_combo = QComboBox()
        self.views_combo.setMinimumWidth(120)
        self.views_combo.setToolTip("Saved camera views (also usable as movie fly-through key frames)")
        self.views_combo.activated.connect(self._restore_view)
        tb.addWidget(self.views_combo)
        self._tb_btn(tb, "Save view", self.save_view, "Save current camera as a named view")
        tb.addSeparator()
        tb.addWidget(QLabel(" Ctrl+click: "))
        self.pick_mode = QComboBox()
        self.pick_mode.addItems(["Info", "Label atom", "Measure distance", "Centre on atom"])
        tb.addWidget(self.pick_mode)
        self._tb_btn(tb, "Clear", self.clear_annotations, "Clear atom labels and measurements (Esc)")
        tb.addSeparator()
        self._tb_btn(tb, "📷 Image", self.export_image, "Export publication image (Ctrl+E)")
        self._tb_btn(tb, "🎬 Movie", self.export_movie, "Export MP4 movie (Ctrl+M)")

    def _tb_btn(self, tb, text, fn, tip=""):
        a = QAction(text, self)
        a.setToolTip(tip or text)
        a.triggered.connect(lambda checked=False: fn())
        tb.addAction(a)
        return a

    def _shortcuts(self):
        for key, fn in ((Qt.Key_Space, lambda: self.traj.toggle_play()), (Qt.Key_Left, lambda: self.traj.step(-1)),
                        (Qt.Key_Right, lambda: self.traj.step(1)), (Qt.Key_Home, lambda: self.traj.set_frame(0)),
                        (Qt.Key_End, lambda: self.traj.set_frame(self.traj.n - 1)),
                        (Qt.Key_Plus, lambda: self.zoom(1.25)), (Qt.Key_Equal, lambda: self.zoom(1.25)),
                        (Qt.Key_Minus, lambda: self.zoom(0.8)), (Qt.Key_R, self.reset_view),
                        (Qt.Key_F, self.focus_ligand), (Qt.Key_Escape, self.clear_annotations)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)

    def _set_enabled(self, on):
        for a in (self.act_addtraj, self.act_save, self.act_saveas, self.act_img, self.act_mov):
            a.setEnabled(on)

    # ================================================================== helpers
    def render(self):
        if self.scene is not None:
            self.inter_panel.show_interactions(self.scene.current_interactions)
        self.renwin.Render()

    def _guard(self, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "MDViz", f"{type(exc).__name__}: {exc}")

    def eventFilter(self, obj, ev):
        if obj is self.vtk and self.scene is not None:
            if ev.type() in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
                w, h = self.renwin.GetSize()
                x = ev.position().x() * w / max(1, self.vtk.width())
                y = h - ev.position().y() * h / max(1, self.vtk.height())
                if (ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton
                        and ev.modifiers() & Qt.AltModifier):
                    self._drag_label = self.scene.label_at(x, y)
                    if self._drag_label is not None:
                        self.traj.stop()
                        return True
                if self._drag_label is not None:
                    self.scene.move_label(self._drag_label, x, y)
                    self.render()
                    if ev.type() == QEvent.MouseButtonRelease:
                        self._drag_label = None
                    return True
        if obj is self.vtk and ev.type() == QEvent.Resize and self.scene is not None:
            QTimer.singleShot(0, self._update_text_scale)
        return super().eventFilter(obj, ev)

    def _update_text_scale(self):
        if self.scene is not None:
            h = self.renwin.GetSize()[1] or 1080
            self.scene.set_text_scale(h / 1080.0)
            self.renwin.Render()

    # ================================================================== loading
    def open_dialog(self):
        dlg = OpenDialog(self, self.last_dir)
        if dlg.exec():
            top, trajs = dlg.result_files()
            self.last_dir = os.path.dirname(top)
            sess = Session(pbc_fix=dlg.pbc.isChecked(), align=dlg.align.isChecked())
            self._guard(self.load_system, top, trajs, sess, True)

    def load_system(self, top, trajs, session: Session, fresh=True):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status.setText(f"Loading {os.path.basename(top)} …")
        QApplication.processEvents()
        try:
            t0 = time.time()
            system = MolSystem(top, trajs)
            note = ""
            if system.n_frames > 1 and not session.pbc_fix:
                chk = system.detect_pbc_problems()
                if chk["broken"]:
                    session.pbc_fix = True
                    if system.is_protein.any():
                        session.align = True
                    note = (f"  ·  Molecules were split across the periodic box (bonds up to "
                            f"{chk['longest_bond']:.0f} Å): 'Fix periodic boundaries' and 'fit to frame 0' "
                            f"were switched on automatically.")
            system.set_processing(session.pbc_fix, session.align, session.align_selection, session.smoothing)
            self.system = system
            self._pes_mode = False
            self._md_stash = None
            session.topology, session.trajectories = system.topology, system.trajectories
            if fresh or not session.reps:
                session.reps = default_reps(system)
                if not system.ligand_resnames:
                    session.interactions.show = False
            self.session = session
            self._make_scene()
            self.status.setText(f"{system.summary()}  (loaded in {time.time() - t0:.1f} s){note}")
        finally:
            QApplication.restoreOverrideCursor()

    def _make_scene(self):
        self._drag_label = None
        self.contact_timeline.clear()
        self.contact_dock.hide()
        s = self.session
        if self.scene is not None:
            self.scene.dispose()
        self.scene = Scene(self.ren, self.system, s, quality="screen")
        self.scene.frame = int(np.clip(s.frame, 0, self.system.n_frames - 1))
        self.scene.text_scale = (self.renwin.GetSize()[1] or 1080) / 1080.0
        self.scene.rebuild()
        self._report_rep_errors()
        if s.view.auto:
            self.scene.auto_view()
        else:
            self.scene.apply_view(s.view)
        self.act_ortho.setChecked(self.ren.GetActiveCamera().GetParallelProjection())
        self.rep_panel.set_reps(s.reps)
        self.display_panel.load(s.render)
        self.system_panel.load(s, self.system.summary().replace(" | ", "\n"))
        self.inter_panel.load(s.interactions)
        labels = self.system.frame_labels
        self.traj.setup(self.system.n_frames, self.system.frame_time, self.scene.frame,
                        label_fn=(lambda f: labels[f].split("  ·  ")[0]) if labels else None)
        self._refresh_views_combo()
        self._set_enabled(True)
        title = os.path.basename(os.path.dirname(self.system.topology))
        self.setWindowTitle(f"MDViz — {os.path.basename(self.system.topology)} ({title})")
        self._update_text_scale()
        self.render()

    def add_trajectory(self):
        if self.system is None:
            return
        ps, _ = QFileDialog.getOpenFileNames(self, "Add trajectory", self.last_dir,
                                             "Trajectory (" + " ".join(f"*.{e}" for e in TRAJECTORY_FORMATS) + ")")
        if ps:
            self.session.view = self.scene.capture_view()
            self._guard(self.load_system, self.system.topology, self.system.trajectories + ps, self.session, False)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls()]
        ext = lambda p: os.path.splitext(p)[1].lower().lstrip(".")
        pes = [p for p in paths if p.lower().endswith(".npz")]
        if pes:
            self.show_pes()
            return self.pes_panel.load_file(pes[0])
        sessions = [p for p in paths if p.endswith(".json")]
        if sessions:
            return self._guard(self._load_session_file, sessions[0])
        tops = [p for p in paths if ext(p) in STRUCTURE_FORMATS and ext(p) not in ("xtc", "trr", "dcd")]
        trajs = [p for p in paths if ext(p) in ("xtc", "trr", "dcd", "nc", "ncdf", "mdcrd", "lammpstrj", "trj")]
        if tops:
            self.last_dir = os.path.dirname(tops[0])
            self._guard(self.load_system, tops[0], trajs, Session(), True)
        elif trajs and self.system is not None:
            self.session.view = self.scene.capture_view()
            self._guard(self.load_system, self.system.topology, self.system.trajectories + trajs, self.session, False)

    # ================================================================== sessions
    def _sync_session(self):
        if self.scene is not None:
            self.session.view = self.scene.capture_view()
            self.session.frame = self.scene.frame

    def save_session(self, ask=False):
        if self.system is None:
            return
        if self._pes_mode:
            QMessageBox.information(self, "Save session", "The viewer is showing PES conformations. "
                                    "Save the surface from the PES panel (Save PES…) instead.")
            return
        if ask or not self.session_path:
            p, _ = QFileDialog.getSaveFileName(self, "Save session", os.path.join(self.last_dir, "session.mdviz.json"),
                                               "MDViz session (*.json)")
            if not p:
                return
            self.session_path = p
        self._sync_session()
        self.session.save(self.session_path)
        self.status.setText(f"Session saved: {self.session_path}")

    def open_session(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open session", self.last_dir, "MDViz session (*.json)")
        if p:
            self._guard(self._load_session_file, p)

    def _load_session_file(self, p):
        sess = Session.load(p)
        self.session_path = p
        self.last_dir = os.path.dirname(p)
        self.load_system(sess.topology, sess.trajectories, sess, fresh=False)

    # ================================================================== frame / scene updates
    def set_frame(self, f):
        if self.scene is None:
            return
        if self._pes_mode and self._pes_res is not None:
            i, j = divmod(int(f), len(self._pes_res.psi))
            if self.pes_panel.selected != (i, j) and self.pes_panel.result is self._pes_res:
                self.pes_panel.select(i, j, from_viewer=True)
        try:
            self.scene.set_frame(f)
            self.contact_timeline.set_frame(f)
        except Exception as exc:
            self.traj.stop()
            traceback.print_exc()
            self.status.setText(f"Error at frame {f}: {exc}")
        self.render()

    def _playback_quality(self, playing):
        if self.scene is not None:
            self.scene.set_playback_quality(playing)
            if not playing:
                self.render()

    def _report_rep_errors(self):
        for i, v in enumerate(self.scene.visuals):
            self.rep_panel.show_error(i, v.error)

    def _rep_changed(self, i, rebuild):
        if self.scene is None:
            return
        if rebuild:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.scene.rebuild_rep(i)
            finally:
                QApplication.restoreOverrideCursor()
            self.rep_panel.show_error(i, self.scene.visuals[i].error)
            if not self.scene.visuals[i].error:
                n = len(self.scene.visuals[i].idx)
                self.rep_panel.show_error(i, "" if n else "Selection matched no atoms.")
        else:
            v = self.scene.visuals[i]
            for a in v.actors:
                _apply_material(a, v.rep.material, v.rep.opacity)
                a.SetVisibility(v.rep.visible)
            if v.rep.visible:
                v.update(self.scene.frame)
        self.render()

    def _rep_added(self, i):
        if self.scene is not None:
            self._guard(self.scene.insert_rep, i)
            self.rep_panel.show_error(i, self.scene.visuals[i].error)
            self.render()

    def _rep_removed(self, i):
        if self.scene is not None:
            self.scene.remove_rep(i)
            self.render()

    def _rebuild_scene(self):
        if self.scene is not None:
            self.scene.rebuild()
            self.render()

    def _display_changed(self):
        if self.scene is not None:
            self.scene.apply_render_settings()
            self.scene.refresh_annotations()
            self.render()

    def _processing_changed(self):
        if self.scene is None:
            return
        s = self.session
        self.contact_timeline.clear()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.system.set_processing(s.pbc_fix, s.align, s.align_selection, s.smoothing)
            self.system._ss_cache.clear()
            self.scene.rebuild()
            self._report_rep_errors()
        finally:
            QApplication.restoreOverrideCursor()
        self.render()

    def _interactions_changed(self, rebuild):
        if self.scene is None:
            return
        if rebuild:
            self.contact_timeline.clear()
            uses_set = any("interacting" in r.selection for r in self.session.reps)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                if uses_set:  # the residue set shown depends on the interaction settings
                    self.scene.rebuild()
                else:
                    self.scene.rebuild_interactions()
            finally:
                QApplication.restoreOverrideCursor()
        self.scene.refresh_annotations()
        self.render()

    def apply_style(self, name):
        """Apply a named look (representations, interaction display, lighting, camera)."""
        if self.scene is None:
            return
        self.contact_timeline.clear()
        from ..state import apply_style
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            apply_style(self.session, self.system, name)
            s = self.session
            self.system.set_processing(s.pbc_fix, s.align, s.align_selection, s.smoothing)
            self.scene.apply_render_settings()
            self.scene.rebuild()
            self.scene.auto_view()
            self.act_ortho.setChecked(self.ren.GetActiveCamera().GetParallelProjection())
            self.rep_panel.set_reps(s.reps)
            self.display_panel.load(s.render)
            self.system_panel.load(s, self.system.summary().replace(" | ", "\n"))
            self.inter_panel.load(s.interactions)
            self._report_rep_errors()
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Style", f"{type(exc).__name__}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self.status.setText(f"Style: {name}")
        self.render()

    # ================================================================== camera
    def zoom(self, factor):
        cam = self.ren.GetActiveCamera()
        cam.Zoom(factor)
        self.ren.ResetCameraClippingRange()
        self.renwin.Render()

    def rotate(self, axis, sign):
        cam = self.ren.GetActiveCamera()
        d = sign * self.rot_step.value()
        if axis == "X":
            cam.Elevation(d)
            cam.OrthogonalizeViewUp()
        elif axis == "Y":
            cam.Azimuth(d)
        else:
            cam.Roll(d)
        self.ren.ResetCameraClippingRange()
        self.renwin.Render()

    def reset_view(self):
        if self.scene is not None:
            self.scene.auto_view()
            self.renwin.Render()

    def focus_ligand(self):
        if self.scene is None:
            return
        idx = self.system.select(self.session.interactions.ligand) if self.system.ligand_resnames else []
        if len(idx):
            self.scene.focus(idx, margin=2.2)
            self.renwin.Render()
        else:
            self.status.setText("No ligand found (non-protein, non-solvent residue).")

    def focus_selection(self):
        if self.scene is None:
            return
        text, ok = QInputDialog.getText(self, "Focus", "Selection to centre on:", text="resid 100 to 110")
        if ok and text:
            try:
                idx = self.system.select(text, self.scene.frame)
            except Exception as exc:
                return QMessageBox.warning(self, "Focus", str(exc))
            if len(idx):
                self.scene.focus(idx, margin=1.4)
                self.renwin.Render()

    def toggle_ortho(self):
        cam = self.ren.GetActiveCamera()
        cam.SetParallelProjection(self.act_ortho.isChecked())
        self.session.view.parallel = self.act_ortho.isChecked()
        self.renwin.Render()

    def toggle_axes(self):
        if self.axes_widget is None:
            self.axes_widget = vtkOrientationMarkerWidget()
            self.axes_widget.SetOrientationMarker(vtkAxesActor())
            self.axes_widget.SetInteractor(self.vtk)
            self.axes_widget.SetViewport(0.0, 0.0, 0.14, 0.2)
        self.axes_widget.SetEnabled(1 if self.act_axes.isChecked() else 0)
        self.axes_widget.InteractiveOff()
        self.renwin.Render()

    def save_view(self):
        if self.scene is None:
            return
        name, ok = QInputDialog.getText(self, "Save view", "Name for this view:",
                                        text=f"view {len(self.session.saved_views) + 1}")
        if ok and name:
            v = self.scene.capture_view()
            self.session.saved_views[name] = v.__dict__.copy()
            self._refresh_views_combo(name)

    def delete_view(self):
        names = list(self.session.saved_views)
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Delete view", "View:", names, 0, False)
        if ok:
            self.session.saved_views.pop(name, None)
            self._refresh_views_combo()

    def _refresh_views_combo(self, select=None):
        self.views_combo.blockSignals(True)
        self.views_combo.clear()
        self.views_combo.addItem("(saved views)")
        self.views_combo.addItems(list(self.session.saved_views))
        if select:
            self.views_combo.setCurrentText(select)
        self.views_combo.blockSignals(False)

    def _restore_view(self, i):
        name = self.views_combo.itemText(i)
        v = self.session.saved_views.get(name)
        if v and self.scene is not None:
            self.scene.apply_view(ViewState(**v))
            self.act_ortho.setChecked(bool(v.get("parallel", True)))
            self.renwin.Render()

    # ================================================================== picking
    def _on_left_press(self, obj, ev):
        iren = self.vtk
        if iren.GetControlKey() and not iren.GetShiftKey() and self.scene is not None:
            x, y = iren.GetEventPosition()
            self._guard(self._pick, x, y)
            return
        obj.OnLeftButtonDown()

    def _pick(self, x, y):
        idx = self.scene.displayed_atoms()
        if not len(idx):
            return
        ren = self.ren

        def world(z):
            ren.SetDisplayPoint(x, y, z)
            ren.DisplayToWorld()
            p = np.array(ren.GetWorldPoint())
            return p[:3] / (p[3] if p[3] else 1.0)
        p0, p1 = world(0.0), world(1.0)
        d = p1 - p0
        pos = self.system.positions(self.scene.frame)[idx]
        t = (pos - p0) @ d / max(d @ d, 1e-12)
        dist = np.linalg.norm(pos - (p0 + np.outer(t, d)), axis=1)
        close = dist < 1.0
        if close.any():
            k = np.where(close)[0][np.argmin(t[close])]
        else:
            k = int(np.argmin(dist))
            if dist[k] > 2.5:
                self.status.setText("No atom under the cursor.")
                return
        atom = int(idx[k])
        mode = self.pick_mode.currentIndex()
        info = self.scene.atom_info(atom)
        if mode == 0:
            self.status.setText(info)
        elif mode == 1:
            labels = self.session.atom_labels
            labels.remove(atom) if atom in labels else labels.append(atom)
            self.status.setText(info)
        elif mode == 2:
            self.pick_buffer.append(atom)
            if len(self.pick_buffer) == 2:
                a, b = self.pick_buffer
                self.pick_buffer = []
                if a != b:
                    self.session.measurements.append([a, b])
                    dd = np.linalg.norm(self.system.positions(self.scene.frame)[a] - self.system.positions(self.scene.frame)[b])
                    self.status.setText(f"Distance {self.scene.atom_label(a)} – {self.scene.atom_label(b)} = {dd:.2f} Å")
            else:
                self.status.setText(f"First atom: {info}  — Ctrl+click a second atom.")
        else:
            cam = self.ren.GetActiveCamera()
            shift = self.system.positions(self.scene.frame)[atom] - np.array(cam.GetFocalPoint())
            cam.SetFocalPoint(*(np.array(cam.GetFocalPoint()) + shift))
            cam.SetPosition(*(np.array(cam.GetPosition()) + shift))
            self.status.setText(f"Centred on {info}")
        self.scene.refresh_annotations()
        self.render()

    def clear_annotations(self):
        if self.scene is None:
            return
        self.session.atom_labels.clear()
        self.session.measurements.clear()
        self.pick_buffer = []
        self.scene.refresh_annotations()
        self.render()

    # ================================================================== export
    def _window_aspect(self):
        w, h = self.renwin.GetSize()
        return (w / h) if h else 16 / 9

    def export_image(self):
        if self.system is None:
            return
        self.traj.stop()
        dlg = ExportImageDialog(self, self._window_aspect(), self.scene.frame, self.system.n_frames, self.last_dir)
        if not dlg.exec():
            return
        v = dlg.values()
        if not v["path"]:
            return
        self.last_dir = os.path.dirname(v["path"]) or self.last_dir
        self._sync_session()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            t0 = time.time()
            render_image(self.system, self.session, v["path"], v["width"], v["height"], v["dpi"], v["frame"],
                         v["supersample"], v["transparent"], v["quality"])
            self.status.setText(f"Saved {v['path']}  ({v['width']}×{v['height']}, {v['dpi']} dpi, "
                                f"{time.time() - t0:.1f} s)")
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Export image", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.renwin.Render()

    def export_movie(self):
        if self.system is None:
            return
        self.traj.stop()
        dlg = ExportMovieDialog(self, self.system.n_frames, self.system.dt, list(self.session.saved_views),
                                self.last_dir, self._window_aspect())
        if not dlg.exec():
            return
        opt = dlg.options()
        path = dlg.path.text().strip()
        if not path:
            return
        self.last_dir = os.path.dirname(path) or self.last_dir
        self._sync_session()
        prog, cb = _progress(self, "Export movie", "Rendering frames…")
        t0 = time.time()
        try:
            render_movie(self.system, self.session, path, opt, cb)
            self.status.setText(f"Saved {path}  ({opt.width}×{opt.height} @ {opt.fps} fps, {time.time() - t0:.0f} s)")
            QMessageBox.information(self, "Export movie", f"Movie written:\n{path}")
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.warning(self, "Export movie", str(exc))
        finally:
            prog.close()
            self.renwin.Render()

    def occupancy(self):
        if self.scene is None:
            return
        det = self.scene.detector
        if det is None:
            from ..interactions import InteractionDetector, InteractionParams
            st = self.session.interactions
            try:
                det = InteractionDetector(self.system, self.system.select(st.ligand), self.system.select(st.receptor),
                                          InteractionParams.from_dict(st.params), self.scene.frame)
            except Exception as exc:
                return QMessageBox.warning(self, "Occupancy", f"Cannot set up detection: {exc}")
        if not len(det.lig):
            return QMessageBox.warning(self, "Occupancy", "The ligand selection matched no atoms.")
        self.traj.stop()
        name = os.path.basename(os.path.dirname(self.system.topology))
        dlg = OccupancyDialog(self, self.system, det, self.last_dir, title=name)
        accepted = dlg.exec()
        if accepted and dlg.result is not None:
            self.contact_timeline.set_result(dlg.result)
            self.contact_dock.show()
        self.set_frame(self.scene.frame)

    def _timeline_frame(self, frame):
        self.traj.stop()
        self.traj.set_frame(frame, True)

    # ================================================================== potential energy surface
    def show_pes(self):
        self.pes_dock.show()
        self.pes_dock.raise_()
        if self.pes_panel.result is not None:
            self.pes_plot_dock.show()

    def open_pes(self):
        self.show_pes()
        self.pes_panel.load_dialog()

    def _pes_result_changed(self, res, points):
        self.pes_plot_dock.show()
        self.show_pes()
        self._guard(self._load_pes_viewer, res)

    def _load_pes_viewer(self, res):
        """Show the PES conformations (one frame per grid point) in the 3D viewer."""
        from ..pes import moving_side, viewer_frames, write_topology_pdb
        from ..state import Representation
        X, labels, ref = viewer_frames(res)
        top = os.path.join(tempfile.gettempdir(), f"mdviz_pes_{os.getpid()}.pdb")
        write_topology_pdb(top, res.atoms, ref, res.bonds)
        if not self._pes_mode and self.system is not None:
            self._sync_session()
            self._md_stash = (self.system, self.session, self.session_path)
        system = MolSystem(top, coordinates=X, frame_labels=labels)
        sess = Session()
        dih = sorted(set(res.phi_atoms) | set(res.psi_atoms))
        sess.reps = [Representation("Molecule", "all", "ball+stick", "element"),
                     Representation("φ/ψ backbone atoms", "index " + " ".join(map(str, dih)), "licorice", "uniform",
                                    uniform_color="#eb6834", opacity=0.4, size=1.8, material="matte")]
        sess.interactions.show = False
        sess.render.show_legend = False
        sess.render.title_size = 24
        if res.bonds:
            try:  # superimpose every conformation on the part that does not move with phi/psi
                fixed = np.where(~moving_side(len(res.atoms), res.bonds, res.phi_atoms[1], res.phi_atoms[2]))[0]
                if len(fixed) >= 3:
                    sess.align, sess.align_selection = True, "index " + " ".join(map(str, fixed))
            except ValueError:
                pass
        i, j = self.pes_panel.selected or np.unravel_index(np.nanargmin(res.energy), res.energy.shape)
        sess.frame = int(i) * len(res.psi) + int(j)
        system.set_processing(False, sess.align, sess.align_selection, 1)
        self.system, self.session, self.session_path = system, sess, ""
        self._pes_mode, self._pes_res = True, res
        self._make_scene()
        self.scene.principal_view()
        self.setWindowTitle("MDViz — potential energy surface conformations")
        self.status.setText(f"PES: {res.energy.size} conformations loaded. Hover or click the surface; "
                            f"the trajectory slider also steps through the grid.")
        self.renwin.Render()

    def _pes_point(self, i, j):
        res = self.pes_panel.result
        if res is None:
            return
        if not self._pes_mode or self._pes_res is not res:
            self._guard(self._load_pes_viewer, res)
        f = i * len(res.psi) + j
        if self.scene is not None and self.scene.frame != f:
            self.traj.set_frame(f, True)
        E = res.energy[i, j]
        if np.isfinite(E):
            xl, yl = res.labels
            self.status.setText(f"{xl} = {res.phi[i]:.0f}°, {yl} = {res.psi[j]:.0f}°  ·  E = {E:.2f} kJ/mol  ·  "
                                f"ΔE = {res.rel[i, j]:.2f} kJ/mol")

    def _restore_md(self):
        if not self._pes_mode:
            return
        if self._md_stash is None:
            self.status.setText("No MD system was open before the PES view.")
            return
        self.system, self.session, self.session_path = self._md_stash
        self._md_stash = None
        self._pes_mode = False
        self._make_scene()

    def closeEvent(self, ev):
        self.pes_panel.closing()
        self.traj.stop()
        try:
            if self.scene is not None:
                self.scene.dispose()
            self.vtk.Finalize()
        except Exception:
            pass
        super().closeEvent(ev)
