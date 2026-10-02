"""PES dock panel: configure and run (phi, psi) scans in the background, load/save surfaces, pick points."""
from __future__ import annotations

import os
import subprocess
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..pes import FORCE_FIELDS, PESConfig, PESResult, PESScanner, find_stationary_points, md_strain_summary

STEPS = [5.0, 10.0, 15.0, 20.0, 30.0]


def wsl_distros() -> list[str]:
    try:
        out = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        text = out.decode("utf-16-le", errors="ignore") if b"\x00" in out else out.decode(errors="ignore")
        return [d.strip() for d in text.splitlines() if d.strip() and "docker" not in d.lower()]
    except Exception:
        return []


class _ScanSignals(QObject):
    setupDone = Signal(object)
    log = Signal(str)
    progress = Signal(int, int, int, int, float)
    finished = Signal(object)
    failed = Signal(str)


class PESPanel(QWidget):
    resultChanged = Signal(object, object)   # PESResult, stationary points
    pointRequested = Signal(int, int)        # grid indices (i: phi, j: psi)
    showMDRequested = Signal()

    def __init__(self, plot_view, parent=None):
        super().__init__(parent)
        self.plot = plot_view
        self.result: PESResult | None = None
        self.points = []
        self.scanner = None
        self.thread = None
        self.selected = None
        self.md_system_fn = lambda: None   # set by the main window: the MD system currently open
        self.md_setup = None               # LigandScanSetup from the MD run
        self._md_for_scan = None
        self._partial = None
        self._dirty = False
        self._t0 = 0.0
        self.sig = _ScanSignals()
        self.sig.log.connect(self._log)
        self.sig.progress.connect(self._on_progress)
        self.sig.finished.connect(self._on_finished)
        self.sig.failed.connect(self._on_failed)
        self.sig.setupDone.connect(self._on_md_setup)
        self.plot.hovered.connect(lambda x, y: self._from_plot(x, y, hover=True))
        self.plot.clicked.connect(lambda x, y: self._from_plot(x, y, hover=False))
        self._refresh = QTimer(self)
        self._refresh.setInterval(1000)
        self._refresh.timeout.connect(self._push_partial)

        lay = QVBoxLayout(self)
        intro = QLabel("Scan the potential energy surface E(φ, ψ): each grid point is minimised in GROMACS "
                       "with φ and ψ restrained and all other degrees of freedom relaxed.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555;")
        lay.addWidget(intro)

        g = QGroupBox("Molecule and force field")
        f = QFormLayout(g)
        self.molecule = QComboBox()
        self.molecule.addItems(["Alanine dipeptide (Ace-Ala-Nme), built automatically",
                                "Custom molecule (.gro + .top)",
                                "Ligand from the loaded MD run (torsion scan)"])
        self.molecule.currentIndexChanged.connect(self._mol_changed)
        f.addRow(self.molecule)
        self.ff = QComboBox()
        self.ff.addItems(FORCE_FIELDS)
        f.addRow("Force field", self.ff)
        self.custom_gro = QLineEdit()
        self.custom_top = QLineEdit()
        for w, flt, label in ((self.custom_gro, "Coordinates (*.gro)", "Structure .gro"),
                              (self.custom_top, "Topology (*.top)", "Topology .top")):
            row = QHBoxLayout()
            row.addWidget(w)
            b = QPushButton("…")
            b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, w=w, flt=flt: self._browse(w, flt))
            row.addWidget(b)
            f.addRow(label, row)
        self.phi_atoms = QLineEdit()
        self.psi_atoms = QLineEdit()
        for w in (self.phi_atoms, self.psi_atoms):
            w.setPlaceholderText("auto (C-N-CA-C / N-CA-C-N)")
            w.setToolTip("Four 1-based atom numbers from the .gro, e.g. 5 7 9 15. Leave empty to auto-detect.")
        f.addRow("φ atoms", self.phi_atoms)
        f.addRow("ψ atoms", self.psi_atoms)
        self._custom_rows = [self.custom_gro, self.custom_top, self.phi_atoms, self.psi_atoms]
        # ligand-from-MD mode
        self.md_btn = QPushButton("Find ligand torsions in the MD run")
        self.md_btn.setToolTip("Finds the ligand topology (e.g. acpype GAFF files) in the run folder, lists its "
                               "rotatable bonds and ranks them by how much they move during the MD run.")
        self.md_btn.clicked.connect(self.analyse_md)
        f.addRow(self.md_btn)
        self.tor1 = QComboBox()
        self.tor2 = QComboBox()
        for c in (self.tor1, self.tor2):
            c.currentIndexChanged.connect(self._torsion_chosen)
        f.addRow("Torsion 1 (x)", self.tor1)
        f.addRow("Torsion 2 (y)", self.tor2)
        self.md_info = QLabel("")
        self.md_info.setWordWrap(True)
        self.md_info.setStyleSheet("color: #555;")
        f.addRow(self.md_info)
        self._md_rows = [self.md_btn, self.tor1, self.tor2, self.md_info]
        lay.addWidget(g)

        g2 = QGroupBox("Scan settings")
        f2 = QFormLayout(g2)
        self.step = QComboBox()
        for s in STEPS:
            n = int(360 / s)
            self.step.addItem(f"{s:g}°  ({n} × {n} = {n * n} minimisations)")
        self.step.setCurrentIndex(1)
        f2.addRow("Grid spacing", self.step)
        self.k = QDoubleSpinBox()
        self.k.setRange(100, 100000)
        self.k.setValue(10000)
        self.k.setSingleStep(1000)
        self.k.setSuffix(" kJ/mol/rad²")
        self.k.setToolTip("Force constant of the φ/ψ dihedral restraints. The restraint energy is subtracted from E.")
        f2.addRow("Restraint k", self.k)
        self.integrator = QComboBox()
        self.integrator.addItems(["l-bfgs", "steep"])
        f2.addRow("Minimiser", self.integrator)
        self.emtol = QDoubleSpinBox()
        self.emtol.setRange(0.01, 100)
        self.emtol.setValue(1.0)
        self.emtol.setSuffix(" kJ/mol/nm")
        f2.addRow("Force tolerance", self.emtol)
        self.nsteps = QSpinBox()
        self.nsteps.setRange(100, 100000)
        self.nsteps.setValue(5000)
        f2.addRow("Max steps", self.nsteps)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText("auto")
        f2.addRow("Parallel workers", self.workers)
        self.eps = QDoubleSpinBox()
        self.eps.setRange(1, 100)
        self.eps.setValue(1)
        self.eps.setToolTip("Dielectric constant for intramolecular electrostatics.\n"
                            "1 = vacuum (standard for alanine dipeptide); 4 ≈ protein interior; 80 ≈ water screening.\n"
                            "Vacuum over-stabilises folded, internally H-bonded ligand conformations.")
        f2.addRow("Dielectric εr", self.eps)
        lay.addWidget(g2)

        g3 = QGroupBox("GROMACS engine")
        f3 = QFormLayout(g3)
        self.gmx = QLineEdit("gmx")
        f3.addRow("gmx command", self.gmx)
        row = QHBoxLayout()
        self.use_wsl = QCheckBox("run inside WSL")
        self.use_wsl.setChecked(os.name == "nt")
        self.distro = QComboBox()
        self.distro.addItem("(default)")
        if os.name == "nt":
            self.distro.addItems(wsl_distros())
        row.addWidget(self.use_wsl)
        row.addWidget(self.distro, 1)
        f3.addRow(row)
        row2 = QHBoxLayout()
        self.workdir = QLineEdit()
        self.workdir.setPlaceholderText("default: MDViz/pes_runs/<date-time>")
        self.workdir.setToolTip("Scan folder. Re-using a folder resumes an interrupted scan with the same settings.")
        b = QPushButton("…")
        b.setFixedWidth(28)
        b.clicked.connect(self._browse_workdir)
        row2.addWidget(self.workdir)
        row2.addWidget(b)
        f3.addRow("Work folder", row2)
        test = QPushButton("Test GROMACS")
        test.clicked.connect(self._test_engine)
        f3.addRow(test)
        lay.addWidget(g3)

        row = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run PES scan")
        self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(self.start_scan)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_scan)
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        lay.addWidget(self.progress)
        self.eta = QLabel("")
        self.eta.setStyleSheet("color: #555;")
        lay.addWidget(self.eta)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumHeight(90)
        lay.addWidget(self.logbox)

        g4 = QGroupBox("Surface")
        v4 = QVBoxLayout(g4)
        row = QHBoxLayout()
        for text, fn in (("Load PES…", self.load_dialog), ("Save PES…", self.save_dialog),
                         ("Publication figure…", self.figure_dialog)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v4.addLayout(row)
        f4 = QFormLayout()
        self.cap = QDoubleSpinBox()
        self.cap.setRange(5, 500)
        self.cap.setValue(60)
        self.cap.setSuffix(" kJ/mol")
        self.cap.setToolTip("Upper limit of the colour scale and surface height (higher energies are clipped)")
        self.cap.editingFinished.connect(self._redraw)
        f4.addRow("Show ΔE up to", self.cap)
        self.persist = QDoubleSpinBox()
        self.persist.setRange(0.1, 50)
        self.persist.setValue(2.0)
        self.persist.setSuffix(" kJ/mol")
        self.persist.setToolTip("A basin counts as a minimum only if it must climb at least this much\n"
                                "to reach a lower basin (filters the bumps of a relaxed scan).")
        self.persist.editingFinished.connect(self._recompute_points)
        f4.addRow("Minimum basin depth", self.persist)
        self.follow = QCheckBox("Hovering the plot updates the 3D molecule")
        self.follow.setChecked(True)
        f4.addRow(self.follow)
        v4.addLayout(f4)
        self.readout = QLabel("No surface loaded.")
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.readout.setStyleSheet("background: #f4f4f2; padding: 6px; border-radius: 4px;")
        v4.addWidget(self.readout)
        v4.addWidget(QLabel("<b>Stationary points</b> (click to show)"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "φ (°)", "ψ (°)", "ΔE (kJ/mol)", "Region"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(170)
        self.table.cellClicked.connect(self._table_clicked)
        v4.addWidget(self.table)
        back = QPushButton("Return the 3D viewer to the MD system")
        back.clicked.connect(self.showMDRequested.emit)
        v4.addWidget(back)
        lay.addWidget(g4)
        lay.addStretch()
        self._mol_changed(0)

    # ------------------------------------------------------------------ helpers
    def _log(self, text):
        self.logbox.appendPlainText(text)

    def _browse(self, w, flt):
        p, _ = QFileDialog.getOpenFileName(self, "Choose file", os.path.dirname(w.text()), flt)
        if p:
            w.setText(p)

    def _browse_workdir(self):
        p = QFileDialog.getExistingDirectory(self, "Scan work folder", self.workdir.text())
        if p:
            self.workdir.setText(p)

    def _mol_changed(self, i):
        custom = i == 1
        for w in self._custom_rows:
            w.setEnabled(custom)
            w.setVisible(i != 2)
        for w in self._md_rows:
            w.setVisible(i == 2)
        self.ff.setEnabled(i == 0)
        self.eps.setValue(4.0 if i == 2 else 1.0)

    # ------------------------------------------------------------------ ligand torsions from an MD run
    def analyse_md(self):
        system = self.md_system_fn()
        if system is None:
            QMessageBox.information(self, "Ligand from MD", "Open the MD run first (File ▸ Open structure / "
                                    "trajectory, e.g. md_10ns_52E5.tpr + md_10ns_52E5.xtc).")
            return
        if not system.ligand_resnames:
            QMessageBox.warning(self, "Ligand from MD", "No ligand residue found in the open system.")
            return
        self.md_btn.setEnabled(False)
        self.md_info.setText("Reading the trajectory and measuring every ligand torsion …")
        sig = self.sig

        def job():
            try:
                from ..ligand_pes import setup_from_md
                sig.setupDone.emit((setup_from_md(system), system))
            except Exception as exc:
                sig.setupDone.emit(exc)
        threading.Thread(target=job, daemon=True).start()

    def _on_md_setup(self, payload):
        self.md_btn.setEnabled(True)
        if isinstance(payload, Exception):
            self.md_info.setText("")
            QMessageBox.warning(self, "Ligand from MD", f"{type(payload).__name__}: {payload}")
            return
        setup, system = payload
        self.md_setup, self._md_for_scan = setup, system
        for c in (self.tor1, self.tor2):
            c.blockSignals(True)
            c.clear()
            for t in setup.torsions:
                c.addItem(f"{t.label}  (atoms {' '.join(str(i + 1) for i in t.quad)}) · MD spread {t.spread:.0f}°")
            c.blockSignals(False)
        self.tor1.setCurrentIndex(0)
        self.tor2.setCurrentIndex(1)
        self.custom_gro.setText(setup.gro)
        self.custom_top.setText(setup.top)
        self._torsion_chosen()
        self.md_info.setText(
            f"{setup.resname}: {len(setup.torsions)} rotatable torsions, ranked by how much they move over "
            f"{len(setup.frames)} MD frames ({setup.times_ps[-1] / 1000:.1f} ns). The two most mobile are selected.\n"
            f"Topology: {os.path.relpath(setup.top, os.path.dirname(system.topology))}")
        run = os.path.dirname(system.topology)
        if not self.workdir.text().strip():
            self.workdir.setText(os.path.join(run, f"pes_ligand_{os.path.basename(run)}"))

    def _torsion_chosen(self, *_):
        if not self.md_setup or self.tor1.currentIndex() < 0 or self.tor2.currentIndex() < 0:
            return
        t1 = self.md_setup.torsions[self.tor1.currentIndex()]
        t2 = self.md_setup.torsions[self.tor2.currentIndex()]
        self.phi_atoms.setText(" ".join(str(i + 1) for i in t1.quad))
        self.psi_atoms.setText(" ".join(str(i + 1) for i in t2.quad))

    @staticmethod
    def _quad(text):
        parts = [int(x) - 1 for x in text.replace(",", " ").split()]
        if parts and len(parts) != 4:
            raise ValueError("Dihedral atoms need exactly four atom numbers.")
        return parts

    def config(self) -> PESConfig:
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                   "pes_runs", time.strftime("%Y%m%d-%H%M%S"))
        return PESConfig(
            molecule="custom" if self.molecule.currentIndex() in (1, 2) else "ala2",
            forcefield=self.ff.currentText(), custom_gro=self.custom_gro.text().strip(),
            custom_top=self.custom_top.text().strip(), phi_atoms=self._quad(self.phi_atoms.text()),
            psi_atoms=self._quad(self.psi_atoms.text()), step=STEPS[self.step.currentIndex()],
            restraint_k=self.k.value(), integrator=self.integrator.currentText(), emtol=self.emtol.value(),
            nsteps=self.nsteps.value(), workers=self.workers.value(), gmx=self.gmx.text().strip() or "gmx",
            epsilon_r=self.eps.value(),
            use_wsl=self.use_wsl.isChecked(),
            wsl_distro="" if self.distro.currentIndex() == 0 else self.distro.currentText(),
            workdir=self.workdir.text().strip() or default_dir)

    def _test_engine(self):
        from ..pes import GromacsEngine
        cfg = self.config()
        try:
            v = GromacsEngine(cfg.gmx, cfg.use_wsl, cfg.wsl_distro).check()
            QMessageBox.information(self, "GROMACS", f"Found GROMACS {v}" + (" in WSL." if cfg.use_wsl else "."))
        except Exception as exc:
            QMessageBox.warning(self, "GROMACS", f"Could not run gmx:\n{exc}")

    # ------------------------------------------------------------------ scanning (background thread)
    def start_scan(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            cfg = self.config()
            if self.molecule.currentIndex() == 2 and not self.md_setup:
                raise ValueError("Click 'Find ligand torsions in the MD run' first.")
            if self.molecule.currentIndex() == 2 and self.tor1.currentIndex() == self.tor2.currentIndex():
                raise ValueError("Choose two different torsions.")
            if cfg.molecule == "custom" and not (os.path.isfile(cfg.custom_gro) and os.path.isfile(cfg.custom_top)):
                raise ValueError("Choose the custom .gro and .top files.")
            if cfg.molecule == "custom" and not (cfg.phi_atoms and cfg.psi_atoms) and self.molecule.currentIndex() == 2:
                raise ValueError("Choose the two torsions to scan.")
        except Exception as exc:
            QMessageBox.warning(self, "PES scan", str(exc))
            return
        self.workdir.setText(cfg.workdir)
        n = int(round(360 / cfg.step))
        grid = cfg.grid()
        self._partial = PESResult(grid, grid.copy(), np.full((n, n), np.nan), np.full((n, n), np.nan),
                                  np.full((n, n), np.nan), None, [], (), (), {"partial": True})
        self.progress.setRange(0, n * n)
        self.progress.setValue(0)
        self.logbox.clear()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._t0 = time.time()
        self.scanner = PESScanner(cfg)
        sig = self.sig

        def job():
            try:
                self.scanner.prepare(log=sig.log.emit)
                sig.log.emit(f"Scanning {n * n} grid points …")
                res = self.scanner.run(progress=lambda d, t, i, j, e, x: sig.progress.emit(d, t, i, j, e))
                sig.finished.emit(res)
            except Exception as exc:
                sig.failed.emit(f"{type(exc).__name__}: {exc}")
        self.thread = threading.Thread(target=job, daemon=True)
        self.thread.start()
        self._refresh.start()

    def cancel_scan(self):
        if self.scanner:
            self.scanner.cancel()
            self._log("Cancelling … (finished points are kept; run again with the same work folder to resume)")

    def _on_progress(self, done, total, i, j, e):
        self.progress.setValue(done)
        self._partial.energy[i, j] = e
        self._dirty = True
        el = time.time() - self._t0
        if done > 5:
            rate = done / el
            self.eta.setText(f"{rate:.1f} points/s · about {max(0, (total - done) / rate):.0f} s remaining")

    def _push_partial(self):
        if self._dirty and self._partial is not None and np.isfinite(self._partial.energy).any():
            self._dirty = False
            self.plot.set_result(self._partial, [], self.cap.value(), title="scanning …")

    def _on_finished(self, res):
        self._refresh.stop()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        m = res.meta
        state = "cancelled" if m.get("cancelled") else "finished"
        self.eta.setText(f"Scan {state} in {m.get('seconds', 0):.0f} s · {int(np.isfinite(res.energy).sum())} points"
                         + (f" · {m['failed']} failed (see {m.get('first_failure_log', '')})" if m.get("failed") else ""))
        self._log(f"Scan {state}. Results in {m.get('workdir')}")
        if self.molecule.currentIndex() == 2 and self.md_setup and self._md_for_scan is not None:
            from ..ligand_pes import attach_md_samples
            try:
                attach_md_samples(res, self.md_setup, self._md_for_scan)
            except Exception as exc:
                self._log(f"Could not attach MD torsions: {exc}")
        try:
            res.save(os.path.join(m["workdir"], "pes.npz"))
            self._log(f"Saved {os.path.join(m['workdir'], 'pes.npz')}")
        except Exception as exc:
            self._log(f"Could not save pes.npz: {exc}")
        if np.isfinite(res.energy).any():
            self.set_result(res)

    def _on_failed(self, msg):
        self._refresh.stop()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._log("ERROR: " + msg)
        QMessageBox.critical(self, "PES scan failed", msg)

    # ------------------------------------------------------------------ results
    def set_result(self, res: PESResult):
        self.result = res
        self.selected = None  # grid indices of a previous surface do not apply to this one
        self._compute_points()
        strain = md_strain_summary(res) if np.isfinite(res.energy).all() else None
        if strain:
            self._log(f"MD frames on this surface: mean ΔE {strain['mean']:.1f} kJ/mol, median {strain['median']:.1f}; "
                      f"{strain['within_5']:.0f}% within 5 kJ/mol of the minimum (εr = {res.meta.get('epsilon_r', 1)})")
        self._fill_table()
        self._redraw()
        self.resultChanged.emit(res, self.points)
        mins = [p for p in self.points if p["type"] == "minimum"]
        if mins:
            self.select(*res.index(mins[0]["phi"], mins[0]["psi"]))
        else:
            i, j = np.unravel_index(np.nanargmin(res.energy), res.energy.shape)
            self.select(int(i), int(j))

    def _compute_points(self):
        try:
            self.points = find_stationary_points(self.result, min_persistence=self.persist.value())
        except Exception as exc:
            self.points = []
            self._log(f"Stationary-point search failed: {exc}")

    def _recompute_points(self):
        if self.result is not None:
            self._compute_points()
            self._fill_table()
            self._redraw()

    def _title(self):
        m = self.result.meta if self.result else {}
        if m.get("idealised_geometry"):
            return os.path.basename(m.get("source", ""))
        eps = m.get("epsilon_r", 1)
        lig = m.get("ligand")
        return (f"{lig + ' · ' if lig else ''}{m.get('forcefield', '')}, {m.get('step', '')}° grid"
                + (f", εr = {eps:g}" if eps and eps != 1 else "")).strip(", ")

    def _redraw(self):
        if self.result is not None:
            self.plot.set_result(self.result, self.points, self.cap.value(), title=self._title())
            if self.selected:
                self._mark(*self.selected)

    def _fill_table(self):
        self.table.setRowCount(len(self.points))
        for r, p in enumerate(self.points):
            vals = [p["label"], f"{p['phi']:.1f}", f"{p['psi']:.1f}", f"{p['rel']:.2f}", p["region"]]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setToolTip("local minimum" if p["type"] == "minimum" else "first-order saddle point")
                self.table.setItem(r, c, it)

    def _table_clicked(self, row, col):
        if self.result is not None and row < len(self.points):
            p = self.points[row]
            self.select(*self.result.index(p["phi"], p["psi"]), exact=p)

    def _from_plot(self, x, y, hover):
        if self.result is None:
            return
        if hover and not self.follow.isChecked():
            return
        i, j = self.result.index(x, y)
        if hover and self.selected == (i, j):
            return
        self.select(i, j)

    def select(self, i, j, exact=None, from_viewer=False):
        """Select grid point (i, j): update readout + plot marker, and ask the viewer to show it."""
        res = self.result
        if res is None:
            return
        self.selected = (i, j)
        self._mark(i, j)
        E = res.energy[i, j]
        rel = res.rel[i, j]
        pa, qa = res.phi_actual[i, j], res.psi_actual[i, j]
        xl, yl = res.labels
        txt = (f"<b>{xl} = {res.phi[i]:.0f}°, {yl} = {res.psi[j]:.0f}°</b>"
               + (f"  (minimised: {pa:.1f}°, {qa:.1f}°)" if np.isfinite(pa) and not res.meta.get("idealised_geometry") else "")
               + (f"<br><b>E = {E:.2f} kJ/mol</b> · ΔE = {rel:.2f} kJ/mol above the global minimum"
                  if np.isfinite(E) else "<br>Not computed")
               + (f"<br>{exact['label']}: {exact['type']} at ({exact['phi']:.1f}°, {exact['psi']:.1f}°), "
                  f"ΔE = {exact['rel']:.2f} kJ/mol (spline); showing the nearest grid conformation" if exact else ""))
        if res.md_samples is not None and len(res.md_samples):
            from ..pes import wrap180
            d = np.hypot(wrap180(res.md_samples[:, 0] - res.phi[i]), wrap180(res.md_samples[:, 1] - res.psi[j]))
            near = int((d <= 15).sum())
            st = md_strain_summary(res)
            txt += (f"<br>MD frames within 15° of this point: {near} of {len(d)}"
                    + (f"<br>MD average: ΔE {st['mean']:.1f} kJ/mol above the surface minimum "
                       f"({st['within_5']:.0f}% of frames within 5 kJ/mol)" if st else ""))
        if res.meta.get("idealised_geometry"):
            txt += "<br><i>Energies from file; molecule shown with idealised geometry.</i>"
        self.readout.setText(txt)
        if not from_viewer:
            self.pointRequested.emit(i, j)

    def _mark(self, i, j):
        res = self.result
        self.plot.set_selected(res.phi[i], res.psi[j], res.rel[i, j])

    # ------------------------------------------------------------------ files
    def load_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load PES", self.workdir.text() or "",
                                           "PES (*.npz *.csv *.tsv *.dat *.txt *.xvg);;All files (*)")
        if p:
            self.load_file(p)

    def load_file(self, p):
        try:
            res = PESResult.load(p)
        except Exception as exc:
            QMessageBox.warning(self, "Load PES", f"{type(exc).__name__}: {exc}")
            return
        self._log(f"Loaded {p}")
        self.set_result(res)

    def save_dialog(self):
        if self.result is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save PES", "pes.npz",
                                           "MDViz PES with conformations (*.npz);;Table (*.csv)")
        if not p:
            return
        if p.lower().endswith(".csv"):
            self.result.to_csv(p)
        else:
            self.result.save(p)
        self._log(f"Saved {p}")

    def figure_dialog(self):
        if self.result is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save publication figure", "pes_map.png",
                                           "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tif)")
        if not p:
            return
        from ..plots import plot_pes_map, plot_pes_surface
        base, ext = os.path.splitext(p)
        plot_pes_map(self.result, self.points, p, cap=self.cap.value(), title=self._title())
        plot_pes_surface(self.result, self.points, base + "_3d" + ext, cap=self.cap.value(), title=self._title())
        self._log(f"Saved {p} and {base + '_3d' + ext}")
        QMessageBox.information(self, "Publication figure", f"Saved:\n{p}\n{base}_3d{ext}")

    def closing(self):
        if self.scanner:
            self.scanner.cancel()
