"""Dock panels: representations, display settings, system processing, interactions, trajectory bar."""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QPushButton, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from ..chem import INTERACTION_COLORS
from ..interactions import KINDS, InteractionParams
from ..state import COLOR_SCHEMES, MATERIALS, STYLES, Representation

REP_PRESETS = {
    "Protein cartoon (secondary structure)": Representation("Protein", "protein", "cartoon", "secondary"),
    "Protein cartoon (by chain)": Representation("Protein chains", "protein", "cartoon", "chain"),
    "Protein cartoon (rainbow N→C)": Representation("Protein rainbow", "protein", "cartoon", "residue index"),
    "Backbone tube (RMSF / putty)": Representation("RMSF tube", "protein", "tube", "rmsf", size=1.3),
    "Binding-pocket side chains (5 Å)": Representation(
        "Binding pocket", "protein and not name N C O H HN and not nonpolarH and byres (around 5 ligand)",
        "licorice", "element", carbon_color="#b8b8b8", size=0.8),
    "Ligand sticks": Representation("Ligand", "ligand and not nonpolarH", "licorice", "element",
                                    carbon_color="#35b779", size=1.2),
    "Ligand ball & stick": Representation("Ligand (ball+stick)", "ligand", "ball+stick", "element",
                                          carbon_color="#35b779"),
    "Ligand space-filling": Representation("Ligand (VDW)", "ligand", "vdw", "element", carbon_color="#35b779"),
    "Pocket surface (hydrophobicity, transparent)": Representation(
        "Pocket surface", "protein and byres (around 8 ligand)", "surface", "hydrophobicity", opacity=0.45),
    "Protein surface (by chain)": Representation("Protein surface", "protein", "surface", "chain", opacity=0.6),
    "Waters within 3.5 Å of ligand (dynamic)": Representation(
        "Bridging waters", "water and byres (around 3.5 ligand)", "licorice", "element",
        update_every_frame=True, size=0.8),
    "Ions": Representation("Ions", "ions", "vdw", "element", size=0.6),
    "All atoms (lines)": Representation("All (lines)", "solute", "lines", "element"),
}


class ColorButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, color="#ffffff", allow_empty=False, parent=None):
        super().__init__(parent)
        self._color = color
        self.allow_empty = allow_empty
        self.setFixedWidth(64)
        self.clicked.connect(self._choose)
        if allow_empty:
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(lambda _: self.set_color("", True))
        self._refresh()

    def color(self):
        return self._color

    def set_color(self, c, emit=False):
        self._color = c or ""
        self._refresh()
        if emit:
            self.colorChanged.emit(self._color)

    def _refresh(self):
        if self._color:
            self.setStyleSheet(f"background-color: {self._color}; border: 1px solid #888; min-height: 18px;")
            self.setText("")
        else:
            self.setStyleSheet("min-height: 18px;")
            self.setText("auto")
        self.setToolTip("Click to choose" + (" · right-click to reset to automatic" if self.allow_empty else ""))

    def _choose(self):
        c = QColorDialog.getColor(QColor(self._color or "#909090"), self, "Choose colour")
        if c.isValid():
            self.set_color(c.name(), True)


def _dspin(lo, hi, val, step=0.1, dec=2, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


# ---------------------------------------------------------------------- representations
class RepPanel(QWidget):
    repChanged = Signal(int, bool)   # index, needs rebuild
    repAdded = Signal(int)
    repRemoved = Signal(int)
    repsReordered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reps: list[Representation] = []
        self._loading = False
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._load_editor)
        self.list.itemChanged.connect(self._item_checked)
        self.list.setMinimumHeight(120)
        lay.addWidget(QLabel("<b>Representations</b> (tick = visible)"))
        lay.addWidget(self.list)
        btns = QHBoxLayout()
        self.add_btn = QToolButton()
        self.add_btn.setText("Add ▾")
        self.add_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self)
        menu.addAction("Blank representation", lambda: self._add(Representation("New rep", "protein", "licorice", "element")))
        menu.addSeparator()
        for name, rep in REP_PRESETS.items():
            menu.addAction(name, lambda r=rep: self._add(copy.deepcopy(r)))
        self.add_btn.setMenu(menu)
        btns.addWidget(self.add_btn)
        for text, fn, tip in (("Duplicate", self._dup, "Duplicate the selected representation"),
                              ("Delete", self._delete, "Delete the selected representation"),
                              ("▲", lambda: self._move(-1), "Move up"), ("▼", lambda: self._move(1), "Move down")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            if len(text) == 1:
                b.setFixedWidth(28)
            btns.addWidget(b)
        lay.addLayout(btns)

        box = QGroupBox("Selected representation")
        f = QFormLayout(box)
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set("name", self.name.text(), False))
        f.addRow("Name", self.name)
        self.sel = QLineEdit()
        self.sel.setToolTip(
            "VMD/MDAnalysis selection language, e.g.\n"
            "  protein and resid 100 to 120\n  resname UNL\n  protein and within 5 of ligand\n"
            "  byres (around 4 ligand)\n  name CA and chainID A\n"
            "Macros: ligand, water, ions, solute, heavy, hydrogen, polarH, nonpolarH")
        self.sel.returnPressed.connect(self._apply_sel)
        sel_row = QHBoxLayout()
        sel_row.addWidget(self.sel)
        ap = QPushButton("Apply")
        ap.clicked.connect(self._apply_sel)
        sel_row.addWidget(ap)
        f.addRow("Selection", sel_row)
        self.err = QLabel("")
        self.err.setStyleSheet("color: #c0392b;")
        self.err.setWordWrap(True)
        f.addRow("", self.err)
        self.style = QComboBox()
        self.style.addItems(STYLES)
        self.style.currentTextChanged.connect(lambda t: self._set("style", t, True))
        f.addRow("Drawing style", self.style)
        self.color = QComboBox()
        self.color.addItems(COLOR_SCHEMES)
        self.color.currentTextChanged.connect(lambda t: self._set("color", t, True))
        f.addRow("Colouring", self.color)
        crow = QHBoxLayout()
        self.ucolor = ColorButton("#4e79a7")
        self.ucolor.colorChanged.connect(lambda c: self._set("uniform_color", c, True))
        self.ccolor = ColorButton("", allow_empty=True)
        self.ccolor.colorChanged.connect(lambda c: self._set("carbon_color", c, True))
        crow.addWidget(QLabel("uniform"))
        crow.addWidget(self.ucolor)
        crow.addWidget(QLabel("carbon"))
        crow.addWidget(self.ccolor)
        crow.addStretch()
        f.addRow("Colours", crow)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(5, 100)
        self.opacity.valueChanged.connect(lambda v: self._set("opacity", v / 100.0, False))
        f.addRow("Opacity", self.opacity)
        self.size = _dspin(0.1, 5.0, 1.0, 0.1)
        self.size.valueChanged.connect(lambda v: self._set("size", v, True))
        f.addRow("Size / thickness", self.size)
        self.material = QComboBox()
        self.material.addItems(MATERIALS)
        self.material.currentTextChanged.connect(lambda t: self._set("material", t, False))
        f.addRow("Material", self.material)
        self.dyn = QCheckBox("Re-evaluate selection every frame")
        self.dyn.setToolTip("For distance-based selections (e.g. waters near the ligand).\n"
                            "Off = evaluated once at the selection frame (like VMD without 'update').")
        self.dyn.toggled.connect(lambda v: self._set("update_every_frame", v, True))
        f.addRow("", self.dyn)
        lay.addWidget(box)
        lay.addStretch()
        box.setEnabled(False)
        self.box = box

    def set_reps(self, reps):
        self.reps = reps
        self._rebuild_list()

    def _rebuild_list(self, select=0):
        self._loading = True
        self.list.clear()
        for r in self.reps:
            it = QListWidgetItem(self._label(r))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r.visible else Qt.Unchecked)
            self.list.addItem(it)
        self._loading = False
        if self.reps:
            self.list.setCurrentRow(min(max(select, 0), len(self.reps) - 1))
        else:
            self.box.setEnabled(False)

    @staticmethod
    def _label(r):
        return f"{r.name}   [{r.style} · {r.color}]"

    def current(self):
        i = self.list.currentRow()
        return i if 0 <= i < len(self.reps) else -1

    def _load_editor(self, i):
        if not (0 <= i < len(self.reps)):
            self.box.setEnabled(False)
            return
        self._loading = True
        r = self.reps[i]
        self.box.setEnabled(True)
        self.name.setText(r.name)
        self.sel.setText(r.selection)
        self.style.setCurrentText(r.style)
        self.color.setCurrentText(r.color)
        self.ucolor.set_color(r.uniform_color)
        self.ccolor.set_color(r.carbon_color)
        self.opacity.setValue(int(round(r.opacity * 100)))
        self.size.setValue(r.size)
        self.material.setCurrentText(r.material)
        self.dyn.setChecked(r.update_every_frame)
        self._loading = False

    def show_error(self, i, text):
        if i == self.current():
            self.err.setText(text)

    def _set(self, attr, value, rebuild):
        i = self.current()
        if self._loading or i < 0:
            return
        setattr(self.reps[i], attr, value)
        self.list.item(i).setText(self._label(self.reps[i]))
        self.repChanged.emit(i, rebuild)

    def _apply_sel(self):
        self._set("selection", self.sel.text().strip() or "all", True)

    def _item_checked(self, item):
        if self._loading:
            return
        i = self.list.row(item)
        self.reps[i].visible = item.checkState() == Qt.Checked
        self.repChanged.emit(i, False)

    def _add(self, rep):
        i = self.current() + 1 if self.current() >= 0 else len(self.reps)
        self.reps.insert(i, rep)
        self._rebuild_list(i)
        self.repAdded.emit(i)

    def _dup(self):
        i = self.current()
        if i >= 0:
            r = copy.deepcopy(self.reps[i])
            r.name += " (copy)"
            self._add(r)

    def _delete(self):
        i = self.current()
        if i >= 0:
            self.reps.pop(i)
            self._rebuild_list(i - 1)
            self.repRemoved.emit(i)

    def _move(self, d):
        i = self.current()
        j = i + d
        if i < 0 or not (0 <= j < len(self.reps)):
            return
        self.reps[i], self.reps[j] = self.reps[j], self.reps[i]
        self._rebuild_list(j)
        self.repsReordered.emit()


# ---------------------------------------------------------------------- display settings
class DisplayPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rs = None
        self._loading = False
        lay = QVBoxLayout(self)
        g = QGroupBox("Scene")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.bg = ColorButton("#ffffff")
        self.bg2 = ColorButton("", allow_empty=True)
        self.bg2.setToolTip("Optional gradient colour (bottom). Right-click for none.")
        row.addWidget(self.bg)
        row.addWidget(QLabel("gradient"))
        row.addWidget(self.bg2)
        row.addStretch()
        f.addRow("Background", row)
        self.ssao = QCheckBox("Ambient occlusion (depth shading)")
        self.fxaa = QCheckBox("Anti-aliasing (FXAA)")
        f.addRow(self.ssao)
        f.addRow(self.fxaa)
        self.light = QSlider(Qt.Horizontal)
        self.light.setRange(20, 200)
        f.addRow("Light intensity", self.light)
        self.slab = _dspin(0, 200, 0, 1, 0, " Å")
        self.slab.setSpecialValueText("off")
        self.slab.setToolTip("Clipping slab: show only a slice this thick around the view centre\n"
                             "(great for looking into buried pockets). 0 = off.")
        f.addRow("Clipping slab", self.slab)
        lay.addWidget(g)
        g2 = QGroupBox("Text overlays")
        f2 = QFormLayout(g2)
        self.title = QLineEdit()
        self.title.setPlaceholderText("e.g. 52E5 bound to GluA2 LBD")
        f2.addRow("Title", self.title)
        self.title_size = QSpinBox()
        self.title_size.setRange(8, 120)
        f2.addRow("Title size (px @1080p)", self.title_size)
        self.show_time = QCheckBox("Show simulation time")
        f2.addRow(self.show_time)
        self.time_fmt = QLineEdit()
        self.time_fmt.setToolTip("Python format string; fields: {ns}, {ps}, {frame}\nExample: t = {ns:.1f} ns")
        f2.addRow("Time format", self.time_fmt)
        self.legend = QCheckBox("Show interaction legend")
        f2.addRow(self.legend)
        self.label_box = QCheckBox("Background box behind 3D labels")
        f2.addRow(self.label_box)
        self.text_color = ColorButton("#202020")
        f2.addRow("Text colour", self.text_color)
        lay.addWidget(g2)
        lay.addStretch()
        for w in (self.bg, self.bg2, self.text_color):
            w.colorChanged.connect(self._emit)
        for w in (self.ssao, self.fxaa, self.show_time, self.legend, self.label_box):
            w.toggled.connect(self._emit)
        self.light.valueChanged.connect(self._emit)
        self.slab.valueChanged.connect(self._emit)
        self.title_size.valueChanged.connect(self._emit)
        self.title.editingFinished.connect(self._emit)
        self.time_fmt.editingFinished.connect(self._emit)

    def load(self, rs):
        self.rs = rs
        self._loading = True
        self.bg.set_color(rs.background)
        self.bg2.set_color(rs.background2)
        self.ssao.setChecked(rs.ssao)
        self.fxaa.setChecked(rs.fxaa)
        self.light.setValue(int(rs.light_intensity * 100))
        self.slab.setValue(rs.slab)
        self.title.setText(rs.title)
        self.title_size.setValue(rs.title_size)
        self.show_time.setChecked(rs.show_time)
        self.time_fmt.setText(rs.time_format)
        self.legend.setChecked(rs.show_legend)
        self.label_box.setChecked(rs.label_box)
        self.text_color.set_color(rs.text_color)
        self._loading = False

    def _emit(self, *a):
        if self._loading or self.rs is None:
            return
        rs = self.rs
        rs.background = self.bg.color() or "#ffffff"
        rs.background2 = self.bg2.color()
        rs.ssao = self.ssao.isChecked()
        rs.fxaa = self.fxaa.isChecked()
        rs.light_intensity = self.light.value() / 100.0
        rs.slab = self.slab.value()
        rs.title = self.title.text()
        rs.title_size = self.title_size.value()
        rs.show_time = self.show_time.isChecked()
        rs.time_format = self.time_fmt.text() or "t = {ns:.2f} ns"
        rs.show_legend = self.legend.isChecked()
        rs.label_box = self.label_box.isChecked()
        rs.text_color = self.text_color.color() or "#202020"
        self.changed.emit()


# ---------------------------------------------------------------------- system / trajectory processing
class SystemPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sess = None
        self._loading = False
        lay = QVBoxLayout(self)
        self.info = QLabel("No system loaded.\nFile ▸ Open structure / trajectory…")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        g = QGroupBox("Trajectory processing")
        f = QFormLayout(g)
        self.pbc = QCheckBox("Fix periodic boundaries (make molecules whole, keep ligand with protein)")
        self.pbc.setToolTip("Use for raw GROMACS output that was not processed with trjconv -pbc.")
        f.addRow(self.pbc)
        self.align = QCheckBox("Remove rotation/translation (fit to frame 0)")
        f.addRow(self.align)
        self.align_sel = QLineEdit()
        f.addRow("Fit selection", self.align_sel)
        self.smooth = QSpinBox()
        self.smooth.setRange(1, 51)
        self.smooth.setSingleStep(2)
        self.smooth.setToolTip("Display only (1 = off). Distances, contacts, RMSF and secondary structure use unsmoothed frames.")
        f.addRow("Smoothing window", self.smooth)
        self.ss_every = QCheckBox("Recompute secondary structure every frame (slower)")
        f.addRow(self.ss_every)
        ap = QPushButton("Apply processing")
        ap.clicked.connect(self._emit)
        f.addRow(ap)
        lay.addWidget(g)
        lay.addStretch()

    def load(self, sess, info_text):
        self.sess = sess
        self._loading = True
        self.info.setText(info_text)
        self.pbc.setChecked(sess.pbc_fix)
        self.align.setChecked(sess.align)
        self.align_sel.setText(sess.align_selection)
        self.smooth.setValue(sess.smoothing)
        self.ss_every.setChecked(sess.ss_every_frame)
        self._loading = False

    def _emit(self):
        if self._loading or self.sess is None:
            return
        s = self.sess
        s.pbc_fix = self.pbc.isChecked()
        s.align = self.align.isChecked()
        s.align_selection = self.align_sel.text().strip() or "protein and name CA"
        s.smoothing = self.smooth.value()
        s.ss_every_frame = self.ss_every.isChecked()
        self.changed.emit()


# ---------------------------------------------------------------------- interactions
class InteractionPanel(QWidget):
    changed = Signal(bool)     # rebuild detector?
    occupancyRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.st = None
        self._loading = False
        lay = QVBoxLayout(self)
        g = QGroupBox("Protein–ligand interactions")
        f = QFormLayout(g)
        self.show = QCheckBox("Show interactions in 3D")
        self.labels = QCheckBox("Label interacting residues")
        f.addRow(self.show)
        f.addRow(self.labels)
        self.lig = QLineEdit()
        self.rec = QLineEdit()
        f.addRow("Ligand", self.lig)
        f.addRow("Receptor", self.rec)
        grid = QGridLayout()
        self.kind_boxes = {}
        for n, k in enumerate(KINDS):
            cb = QCheckBox(k)
            cb.setStyleSheet(f"color: {INTERACTION_COLORS[k]}; font-weight: bold;")
            self.kind_boxes[k] = cb
            grid.addWidget(cb, n // 2, n % 2)
        f.addRow(grid)
        self.p_widgets = {
            "hbond_dist": _dspin(2.5, 4.5, 3.5, 0.1, 1, " Å"),
            "hbond_angle": _dspin(90, 180, 120, 5, 0, "°"),
            "hydrophobic_dist": _dspin(3.0, 5.5, 4.0, 0.1, 1, " Å"),
            "salt_dist": _dspin(3.0, 6.0, 4.0, 0.1, 1, " Å"),
            "pi_dist": _dspin(4.0, 7.0, 5.5, 0.1, 1, " Å"),
            "cation_pi_dist": _dspin(4.0, 7.5, 6.0, 0.1, 1, " Å"),
            "halogen_dist": _dspin(3.0, 4.5, 3.5, 0.1, 1, " Å"),
        }
        names = {"hbond_dist": "H-bond D···A", "hbond_angle": "H-bond D–H···A ≥", "hydrophobic_dist": "Hydrophobic ≤",
                 "salt_dist": "Salt bridge ≤", "pi_dist": "π-stacking centroid ≤", "cation_pi_dist": "Cation–π ≤",
                 "halogen_dist": "Halogen X···A ≤"}
        for k, w in self.p_widgets.items():
            f.addRow(names[k], w)
        self.label_size = QSpinBox()
        self.label_size.setRange(6, 72)
        f.addRow("Label size (px @1080p)", self.label_size)
        self.callouts = QCheckBox("Place residue labels beside pocket with leader lines")
        self.callouts.setToolTip("Alt+drag a label in the viewer to pin its position; saved with the session.")
        f.addRow(self.callouts)
        reset_labels = QPushButton("Reset label positions")
        reset_labels.clicked.connect(self._reset_labels)
        f.addRow(reset_labels)
        self.pocket_samples = QComboBox()
        self.pocket_samples.addItem("Sampled preview (40 frames)", 40)
        self.pocket_samples.addItem("All frames (may take longer)", 0)
        self.pocket_samples.setToolTip("Determines the stable pocket residue set. Preview can miss brief contacts; occupancy has its own frame range.")
        f.addRow("Stable pocket residues", self.pocket_samples)
        self.label_color = ColorButton("#202020")
        f.addRow("Label colour", self.label_color)
        self.dash = _dspin(0.02, 0.3, 0.08, 0.01, 2, " Å")
        f.addRow("Dash radius", self.dash)
        ap = QPushButton("Apply")
        ap.clicked.connect(lambda: self._emit(True))
        f.addRow(ap)
        lay.addWidget(g)
        lay.addWidget(QLabel("<b>Current frame</b>"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Type", "Residue", "Ligand / receptor atoms", "Dist (Å)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(160)
        lay.addWidget(self.table)
        occ = QPushButton("Trajectory occupancy analysis…")
        occ.clicked.connect(self.occupancyRequested.emit)
        lay.addWidget(occ)
        self.show.toggled.connect(lambda: self._emit(True))
        self.labels.toggled.connect(lambda: self._emit(False))
        self.label_size.valueChanged.connect(lambda: self._emit(False))
        self.callouts.toggled.connect(lambda: self._emit(False))
        self.pocket_samples.currentIndexChanged.connect(lambda: self._emit(True))
        self.label_color.colorChanged.connect(lambda: self._emit(False))
        self.dash.valueChanged.connect(lambda: self._emit(False))
        self.lig.returnPressed.connect(lambda: self._emit(True))
        self.rec.returnPressed.connect(lambda: self._emit(True))
        for cb in self.kind_boxes.values():
            cb.toggled.connect(lambda: self._emit(True))

    def load(self, st):
        self.st = st
        self._loading = True
        p = InteractionParams.from_dict(st.params)
        self.show.setChecked(st.show)
        self.labels.setChecked(st.labels)
        self.lig.setText(st.ligand)
        self.rec.setText(st.receptor)
        for k, cb in self.kind_boxes.items():
            cb.setChecked(k in p.enabled)
        for k, w in self.p_widgets.items():
            w.setValue(getattr(p, k))
        self.label_size.setValue(st.label_size)
        self.callouts.setChecked(st.callout_labels)
        self.pocket_samples.setCurrentIndex(1 if st.pocket_samples == 0 else 0)
        self.label_color.set_color(st.label_color)
        self.dash.setValue(st.dash_radius)
        self._loading = False

    def _emit(self, rebuild):
        if self._loading or self.st is None:
            return
        st = self.st
        st.show = self.show.isChecked()
        st.labels = self.labels.isChecked()
        st.ligand = self.lig.text().strip() or "ligand"
        st.receptor = self.rec.text().strip() or "protein"
        p = InteractionParams.from_dict(st.params)
        p.enabled = [k for k, cb in self.kind_boxes.items() if cb.isChecked()]
        for k, w in self.p_widgets.items():
            setattr(p, k, w.value())
        st.params = p.to_dict()
        st.label_size = self.label_size.value()
        st.callout_labels = self.callouts.isChecked()
        st.pocket_samples = self.pocket_samples.currentData()
        st.label_color = self.label_color.color() or "#202020"
        st.dash_radius = self.dash.value()
        self.changed.emit(rebuild)

    def show_interactions(self, inter):
        self.table.setRowCount(len(inter))
        for r, i in enumerate(inter):
            vals = [i.kind, i.residue, f"{i.lig_label} ↔ {i.rec_label}", f"{i.distance:.2f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setForeground(QColor(INTERACTION_COLORS[i.kind]))
                self.table.setItem(r, c, it)

    def _reset_labels(self):
        if self.st is not None:
            self.st.label_positions.clear()
            self.changed.emit(False)


# ---------------------------------------------------------------------- trajectory bar
class TrajectoryBar(QWidget):
    frameChanged = Signal(int)
    playbackChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n = 1
        self.time_fn = lambda f: 0.0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self.buttons = {}
        for key, text, tip in (("first", "⏮", "First frame"), ("prev", "◀", "Previous frame (←)"),
                               ("play", "▶", "Play / pause (Space)"), ("next", "▶|", "Next frame (→)"),
                               ("last", "⏭", "Last frame")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedWidth(36)
            lay.addWidget(b)
            self.buttons[key] = b
        self.slider = QSlider(Qt.Horizontal)
        lay.addWidget(self.slider, 1)
        self.spin = QSpinBox()
        self.spin.setToolTip("Frame")
        lay.addWidget(self.spin)
        self.time = QLabel("0.00 ns")
        self.time.setMinimumWidth(80)
        lay.addWidget(self.time)
        lay.addWidget(QLabel("step"))
        self.stride = QSpinBox()
        self.stride.setRange(1, 1000)
        lay.addWidget(self.stride)
        lay.addWidget(QLabel("fps"))
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(15)
        lay.addWidget(self.fps)
        self.loop = QCheckBox("loop")
        self.loop.setChecked(True)
        lay.addWidget(self.loop)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.slider.valueChanged.connect(self._slider)
        self.spin.valueChanged.connect(self._spin)
        self.buttons["first"].clicked.connect(lambda: self.set_frame(0, True))
        self.buttons["last"].clicked.connect(lambda: self.set_frame(self.n - 1, True))
        self.buttons["prev"].clicked.connect(lambda: self.step(-1))
        self.buttons["next"].clicked.connect(lambda: self.step(1))
        self.buttons["play"].clicked.connect(self.toggle_play)
        self.fps.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
        self._busy = False

    def setup(self, n_frames, time_fn, frame=0, label_fn=None):
        self.stop()
        self.n = max(1, n_frames)
        self.time_fn = time_fn
        self.label_fn = label_fn
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setRange(0, self.n - 1)
            w.setValue(frame)
            w.blockSignals(False)
        self._update_time(frame)
        self.setEnabled(self.n > 1)

    def frame(self):
        return self.slider.value()

    def set_frame(self, f, emit=True):
        f = max(0, min(self.n - 1, int(f)))
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setValue(f)
            w.blockSignals(False)
        self._update_time(f)
        if emit:
            self.frameChanged.emit(f)

    def _update_time(self, f):
        try:
            if getattr(self, "label_fn", None):
                self.time.setText(self.label_fn(f))
                return
            self.time.setText(f"{self.time_fn(f) / 1000:.2f} ns")
        except Exception:
            self.time.setText("")

    def _slider(self, v):
        self.set_frame(v, True)

    def _spin(self, v):
        self.set_frame(v, True)

    def step(self, d):
        self.set_frame(self.frame() + d * self.stride.value(), True)

    def toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.timer.start(int(1000 / self.fps.value()))
            self.buttons["play"].setText("⏸")
            self.playbackChanged.emit(True)

    def stop(self):
        was_playing = self.timer.isActive()
        self.timer.stop()
        self.buttons["play"].setText("▶")
        if was_playing:
            self.playbackChanged.emit(False)

    def is_playing(self):
        return self.timer.isActive()

    def _tick(self):
        if self._busy:
            return
        self._busy = True
        try:
            nxt = self.frame() + self.stride.value()
            if nxt >= self.n:
                if self.loop.isChecked():
                    nxt = 0
                else:
                    self.stop()
                    return
            self.set_frame(nxt, True)
        finally:
            self._busy = False
