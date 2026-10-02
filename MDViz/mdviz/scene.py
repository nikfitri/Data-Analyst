"""VTK scene built from a Session: representations, interactions, labels and overlays.

The same Scene class drives the interactive viewer and the off-screen exporter, so exported
images and movies match what is on screen.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict

import numpy as np

import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkRenderingCore import (vtkActor, vtkBillboardTextActor3D, vtkGlyph3DMapper, vtkLightKit,
                                         vtkPolyDataMapper, vtkRenderer, vtkTextActor)

from . import chem
from .geometry import cartoon_mesh, gaussian_surface_grid
from .interactions import InteractionDetector, InteractionParams
from .state import Representation, Session, ViewState

MATERIAL_PRESETS = {
    "default": (0.15, 0.80, 0.25, 25.0),
    "matte": (0.25, 0.85, 0.0, 1.0),
    "glossy": (0.10, 0.75, 0.55, 60.0),
    "shiny": (0.08, 0.70, 0.90, 110.0),
    "flat": (0.55, 0.45, 0.0, 1.0),
    "soft": (0.30, 0.72, 0.35, 40.0),   # bright, satin look (publication close-ups)
}
# Bonds longer than this are never drawn: they only occur when a molecule is split across the periodic box,
# and would otherwise appear as long streaks across the image.
MAX_DRAWN_BOND = 3.0
MAX_DRAWN_INTERACTION = 12.0
RES_TYPE_COLORS = {"hydrophobic": "#d9c3a0", "polar": "#6abf69", "acidic": "#e05252", "basic": "#4a7bd0", "other": "#bdbdbd"}


_FONT_FILES = {
    False: [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"],
    True: [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
}


def _set_font(tp):
    """Use a system TrueType font: VTK's built-in fonts lack Greek letters (φ, ψ, Δ) and Å."""
    for f in _FONT_FILES[bool(tp.GetBold())]:
        if os.path.exists(f):
            tp.SetFontFamily(4)  # VTK_FONT_FILE
            tp.SetFontFile(f)
            return
    tp.SetFontFamilyToArial()


# ---------------------------------------------------------------------- vtk helpers
def _vtk_points(arr: np.ndarray) -> vtkPoints:
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float32), deep=True))
    return pts


def _colors(arr: np.ndarray, name="colors"):
    c = numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.uint8), deep=True)
    c.SetName(name)
    return c


def _float_array(arr: np.ndarray, name: str):
    a = numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float32), deep=True)
    a.SetName(name)
    return a


def _cells(conn: np.ndarray, per: int) -> vtkCellArray:
    conn = np.ascontiguousarray(conn, dtype=np.int64).ravel()
    offsets = np.arange(0, len(conn) + 1, per, dtype=np.int64)
    ca = vtkCellArray()
    ca.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(conn, deep=True))
    return ca


def _apply_material(actor, material: str, opacity: float):
    amb, dif, spec, power = MATERIAL_PRESETS.get(material, MATERIAL_PRESETS["default"])
    p = actor.GetProperty()
    p.SetInterpolationToPhong()
    p.SetAmbient(amb)
    p.SetDiffuse(dif)
    p.SetSpecular(spec)
    p.SetSpecularPower(power)
    p.SetSpecularColor(1, 1, 1)
    p.SetOpacity(opacity)


def _sphere_source(res):
    s = vtkSphereSource()
    s.SetRadius(1.0)
    s.SetThetaResolution(res)
    s.SetPhiResolution(max(8, res * 2 // 3))
    return s


def _cylinder_source(res):
    """Unit cylinder along +X from x=0 to x=1, radius 1 (scaled per glyph)."""
    c = vtkCylinderSource()
    c.SetResolution(res)
    c.SetHeight(1.0)
    c.SetRadius(1.0)
    c.CappingOff()
    t = vtkTransform()
    t.Translate(0.5, 0, 0)
    t.RotateZ(-90)
    f = vtkTransformPolyDataFilter()
    f.SetTransform(t)
    f.SetInputConnection(c.GetOutputPort())
    return f


def _glyph_actor(source, orient=False):
    m = vtkGlyph3DMapper()
    m.SetSourceConnection(source.GetOutputPort())
    m.SetScalarModeToUsePointFieldData()
    m.SelectColorArray("colors")
    m.SetColorModeToDirectScalars()
    m.ScalarVisibilityOn()
    m.ScalingOn()
    m.SetScaleArray("scale")
    if orient:
        m.SetScaleModeToScaleByVectorComponents()
        m.SetOrientationArray("orient")
        m.SetOrientationModeToDirection()
        m.OrientOn()
    else:
        m.SetScaleModeToScaleByMagnitude()
        m.OrientOff()
    a = vtkActor()
    a.SetMapper(m)
    return a, m


def _set_glyph_data(mapper, centers, scales, colors, orient=None):
    centers = np.asarray(centers if len(centers) else np.zeros((0, 3)), dtype=np.float32)
    pd = mapper.GetInput()
    arrays = {"scale": np.asarray(scales, dtype=np.float32),
              "colors": np.asarray(colors if len(colors) else np.zeros((0, 3)), dtype=np.uint8)}
    if orient is not None:
        arrays["orient"] = np.asarray(orient, dtype=np.float32)
    if (pd is not None and pd.GetNumberOfPoints() == len(centers)
            and all(pd.GetPointData().GetArray(k) is not None
                    and vtk_to_numpy(pd.GetPointData().GetArray(k)).shape == v.shape
                    for k, v in arrays.items())):
        vtk_to_numpy(pd.GetPoints().GetData())[:] = centers
        pd.GetPoints().GetData().Modified()
        pd.GetPoints().Modified()
        for k, v in arrays.items():
            arr = pd.GetPointData().GetArray(k)
            vtk_to_numpy(arr)[:] = v
            arr.Modified()
        pd.Modified()
        return
    pd = vtkPolyData()
    pd.SetPoints(_vtk_points(centers))
    pdat = pd.GetPointData()
    pdat.AddArray(_float_array(scales, "scale"))
    pdat.AddArray(_colors(colors if len(colors) else np.zeros((0, 3), np.uint8)))
    if orient is not None:
        pdat.AddArray(_float_array(orient, "orient"))
    mapper.SetInputData(pd)


def _mesh_polydata(points, normals, tris, colors):
    pd = vtkPolyData()
    pd.SetPoints(_vtk_points(points))
    pd.SetPolys(_cells(tris, 3))
    n = _float_array(normals, "Normals")
    pd.GetPointData().SetNormals(n)
    pd.GetPointData().SetScalars(_colors(colors))
    return pd


def _dashes(p1, p2, dash=0.28, gap=0.22):
    """Split segments into dash (start, vector) pairs."""
    starts, vecs = [], []
    for a, b in zip(p1, p2):
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-3:
            continue
        u = v / L
        n = max(1, int((L + gap) // (dash + gap)))
        pad = (L - (n * dash + (n - 1) * gap)) / 2
        for k in range(n):
            s = a + u * (pad + k * (dash + gap))
            starts.append(s)
            vecs.append(u * dash)
    return np.array(starts).reshape(-1, 3), np.array(vecs).reshape(-1, 3)


def _cmap(values: np.ndarray, name: str, vmin=None, vmax=None) -> np.ndarray:
    import matplotlib
    cm = matplotlib.colormaps[name]
    v = np.asarray(values, dtype=float)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    return (np.asarray(cm(t))[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------- representation visual
class RepVisual:
    def __init__(self, scene: "Scene", rep: Representation):
        self.scene = scene
        self.rep = rep
        self.actors: list = []
        self.idx = np.zeros(0, np.int64)
        self.error = ""
        self._sel_frame = None

    @property
    def sys(self):
        return self.scene.sys

    def remove(self):
        for a in self.actors:
            self.scene.ren.RemoveActor(a)
        self.actors = []

    def _add(self, actor):
        _apply_material(actor, self.rep.material, self.rep.opacity)
        actor.SetVisibility(self.rep.visible)
        self.scene.ren.AddActor(actor)
        self.actors.append(actor)
        return actor

    # -------------------------------------------------------------- setup
    def build(self, frame: int):
        self.remove()
        self.error = ""
        rep = self.rep
        sel_frame = frame if rep.update_every_frame else rep.selection_frame
        try:
            self.idx = self.sys.select(rep.selection, sel_frame)
        except Exception as exc:  # invalid selection text
            self.idx = np.zeros(0, np.int64)
            self.error = str(exc).splitlines()[0][:200]
            return
        self._sel_frame = sel_frame
        self._prepare()
        q = self.scene.quality
        style = rep.style
        if style in ("cartoon", "tube", "surface"):
            self.mesh_actor = self._add(vtkActor())
            self.mesh_mapper = vtkPolyDataMapper()
            self.mesh_mapper.SetScalarModeToUsePointData()
            self.mesh_mapper.SetColorModeToDirectScalars()
            self.mesh_actor.SetMapper(self.mesh_mapper)
        elif style == "lines":
            self.line_actor = self._add(vtkActor())
            self.line_mapper = vtkPolyDataMapper()
            self.line_mapper.SetColorModeToDirectScalars()
            self.line_actor.SetMapper(self.line_mapper)
            self.line_actor.GetProperty().SetLighting(False)
            self.iso_actor, self.iso_mapper = _glyph_actor(_sphere_source(q["sphere"]))
            self._add(self.iso_actor)
        else:
            self.sph_actor, self.sph_mapper = _glyph_actor(_sphere_source(q["sphere"]))
            self._add(self.sph_actor)
            if style != "vdw":
                self.cyl_actor, self.cyl_mapper = _glyph_actor(_cylinder_source(q["cylinder"]), orient=True)
                self._add(self.cyl_actor)
        self.update(frame)

    def _prepare(self):
        s = self.sys
        idx = self.idx
        rep = self.rep
        mask = np.zeros(s.n_atoms, bool)
        mask[idx] = True
        b = s.bonds
        self.bonds = b[mask[b[:, 0]] & mask[b[:, 1]]] if len(b) else np.zeros((0, 2), np.int64)
        self.atom_colors = self.scene.atom_colors(rep, idx)
        el = s.elements[idx]
        r = rep.size
        if rep.style == "licorice":
            self.radii = np.full(len(idx), 0.22 * r, np.float32)
            self.bond_r = 0.22 * r
        elif rep.style == "ball+stick":
            self.radii = (0.28 * s.vdw[idx] * r).astype(np.float32)
            self.bond_r = 0.12 * r
        elif rep.style == "vdw":
            self.radii = (s.vdw[idx] * r).astype(np.float32)
            self.bond_r = 0.0
        else:
            self.radii = np.full(len(idx), 0.2 * r, np.float32)
            self.bond_r = 0.1 * r
        bonded = np.zeros(s.n_atoms, bool)
        if len(self.bonds):
            bonded[self.bonds.ravel()] = True
        self.isolated = idx[~bonded[idx]]
        self.local = {int(a): k for k, a in enumerate(idx)}
        if rep.style in ("cartoon", "tube"):
            self._prepare_backbone()

    def _prepare_backbone(self):
        s = self.sys
        atoms = s.u.atoms
        sel = set(self.idx.tolist())
        ca, o, res, chains = [], [], [], []
        names = atoms.names
        for r in s.u.atoms[self.idx].residues:
            if not s.is_protein[r.atoms.indices[0]]:
                continue
            ai = r.atoms.indices
            nm = names[ai]
            ca_i = ai[nm == "CA"]
            if not len(ca_i) or int(ca_i[0]) not in sel:
                continue
            o_i = ai[np.isin(nm, ["O", "OC1", "OT1", "O1"])]
            ca.append(int(ca_i[0]))
            o.append(int(o_i[0]) if len(o_i) else -1)
            res.append(r.resindex)
            chains.append(s.chains[ca_i[0]])
        self.bb_ca = np.array(ca, np.int64)
        self.bb_o = np.array(o, np.int64)
        self.bb_res = np.array(res, np.int64)
        self.bb_chain = np.array(chains)
        # colour per residue = colour of its CA in the chosen scheme
        self.bb_colors = self.scene.atom_colors(self.rep, self.bb_ca) if len(ca) else np.zeros((0, 3), np.uint8)

    # -------------------------------------------------------------- per frame
    def update(self, frame: int):
        if self.error or not self.rep.visible:
            return
        rep = self.rep
        if rep.update_every_frame and self._sel_frame != frame:
            new = self.sys.select(rep.selection, frame)
            if not np.array_equal(new, self.idx):
                self.idx = new
                self._prepare()
            self._sel_frame = frame
        if self.scene.ss_every_frame and rep.color == "secondary":
            self.atom_colors = self.scene.atom_colors(rep, self.idx, frame)
            if rep.style in ("cartoon", "tube") and len(self.bb_ca):
                self.bb_colors = self.scene.atom_colors(rep, self.bb_ca, frame)
        pos = self.sys.positions(frame)
        style = rep.style
        if style in ("cartoon", "tube"):
            self._update_cartoon(pos, frame)
        elif style == "surface":
            self._update_surface(pos)
        elif style == "lines":
            self._update_lines(pos)
        else:
            self._update_spheres_cylinders(pos)

    def _drawable_bonds(self, pos):
        b = self.bonds
        if len(b):
            b = b[np.linalg.norm(pos[b[:, 0]] - pos[b[:, 1]], axis=1) <= MAX_DRAWN_BOND]
        return b

    def _update_spheres_cylinders(self, pos):
        idx = self.idx
        _set_glyph_data(self.sph_mapper, pos[idx], self.radii, self.atom_colors)
        if self.rep.style != "vdw":
            b = self._drawable_bonds(pos)
            if len(b):
                a, c = b[:, 0], b[:, 1]
                pa, pc = pos[a], pos[c]
                half = (pc - pa) / 2
                starts = np.vstack([pa, pc])
                vec = np.vstack([half, -half])
                length = np.linalg.norm(vec, axis=1)
                scales = np.stack([length, np.full_like(length, self.bond_r), np.full_like(length, self.bond_r)], axis=1)
                la = np.array([self.local[int(x)] for x in a])
                lc = np.array([self.local[int(x)] for x in c])
                cols = np.vstack([self.atom_colors[la], self.atom_colors[lc]])
                _set_glyph_data(self.cyl_mapper, starts, scales, cols, orient=vec)
            else:
                _set_glyph_data(self.cyl_mapper, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.uint8),
                                orient=np.zeros((0, 3)))

    def _update_lines(self, pos):
        b = self._drawable_bonds(pos)
        if len(b):
            a, c = b[:, 0], b[:, 1]
            mid = (pos[a] + pos[c]) / 2
            la = np.array([self.local[int(x)] for x in a])
            lc = np.array([self.local[int(x)] for x in c])
            pts = np.vstack([pos[a], mid, pos[c], mid])
            n = len(b)
            conn = np.vstack([np.stack([np.arange(n), np.arange(n) + n], 1),
                              np.stack([np.arange(n) + 2 * n, np.arange(n) + 3 * n], 1)])
            cols = np.vstack([self.atom_colors[la], self.atom_colors[la], self.atom_colors[lc], self.atom_colors[lc]])
            pd = vtkPolyData()
            pd.SetPoints(_vtk_points(pts))
            pd.SetLines(_cells(conn, 2))
            pd.GetPointData().SetScalars(_colors(cols))
        else:
            pd = vtkPolyData()
        self.line_mapper.SetInputData(pd)
        self.line_actor.GetProperty().SetLineWidth(max(1.0, 2.0 * self.rep.size * self.scene.text_scale))
        iso = self.isolated
        li = np.array([self.local[int(x)] for x in iso], dtype=np.int64)
        _set_glyph_data(self.iso_mapper, pos[iso], np.full(len(iso), 0.25 * self.rep.size, np.float32),
                        self.atom_colors[li] if len(li) else np.zeros((0, 3), np.uint8))

    def _update_cartoon(self, pos, frame):
        if not len(self.bb_ca):
            self.mesh_mapper.SetInputData(vtkPolyData())
            return
        res_idx, codes = self.scene.ss_codes(frame if self.scene.ss_every_frame else self.scene.ss_frame)
        # codes are per protein residue; look them up by residue index (ligands/ions may sit in between)
        lut = dict(zip(np.asarray(res_idx).tolist(), np.asarray(codes).tolist()))
        ss = np.array([lut.get(int(r), "-") for r in self.bb_res])
        ca = pos[self.bb_ca]
        o = np.where((self.bb_o >= 0)[:, None], pos[np.maximum(self.bb_o, 0)], np.nan)
        q = self.scene.quality
        P, N, T, C = cartoon_mesh(ca, o, ss, self.bb_colors, self.bb_chain, style=self.rep.style,
                                  scale=self.rep.size, sub=q["cartoon_sub"], sides=q["cartoon_sides"])
        self.mesh_mapper.SetInputData(_mesh_polydata(P, N, T, C) if len(P) else vtkPolyData())

    def _update_surface(self, pos):
        from scipy.spatial import cKDTree
        idx = self.idx
        heavy = idx[self.sys.elements[idx] != "H"] if (self.sys.elements[idx] != "H").any() else idx
        if not len(heavy):
            self.mesh_mapper.SetInputData(vtkPolyData())
            return
        p = pos[heavy]
        key = (p.tobytes(), heavy.tobytes(), self.atom_colors.tobytes(), self.scene.quality["surface_spacing"])
        cache = getattr(self, "_surface_cache", OrderedDict())
        self._surface_cache = cache
        if key in cache:
            cache.move_to_end(key)
            self.mesh_mapper.SetInputData(cache[key])
            return
        grid, lo, spacing, dims = gaussian_surface_grid(p, self.sys.vdw[heavy], spacing=self.scene.quality["surface_spacing"])
        img = vtkImageData()
        img.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
        img.SetOrigin(*lo)
        img.SetSpacing(spacing, spacing, spacing)
        img.GetPointData().SetScalars(numpy_to_vtk(grid.ravel(), deep=True))
        fe = vtkFlyingEdges3D()
        fe.SetInputData(img)
        fe.SetValue(0, 1.0)
        fe.ComputeNormalsOn()
        fe.Update()
        out = fe.GetOutput()
        npts = out.GetNumberOfPoints()
        if npts:
            from vtkmodules.util.numpy_support import vtk_to_numpy
            verts = vtk_to_numpy(out.GetPoints().GetData())
            _, nearest = cKDTree(p).query(verts)
            local = np.array([self.local[int(a)] for a in heavy])
            out.GetPointData().SetScalars(_colors(self.atom_colors[local[nearest]]))
        self.mesh_mapper.SetInputData(out)
        cache[key] = out
        if len(cache) > 2:
            cache.popitem(last=False)

    def set_visible(self, v):
        for a in self.actors:
            a.SetVisibility(v)


# ---------------------------------------------------------------------- scene
class Scene:
    QUALITY = {
        "playback": {"sphere": 14, "cylinder": 10, "cartoon_sub": 3, "cartoon_sides": 6, "surface_spacing": 1.5},
        "screen": {"sphere": 14, "cylinder": 10, "cartoon_sub": 6, "cartoon_sides": 10, "surface_spacing": 0.9},
        "export": {"sphere": 32, "cylinder": 24, "cartoon_sub": 10, "cartoon_sides": 20, "surface_spacing": 0.5},
    }

    def __init__(self, renderer, system, session: Session, quality="screen"):
        self.ren = renderer
        self.sys = system
        self.sess = session
        self.quality = self.QUALITY[quality]
        self.quality_name = quality
        self.text_scale = 1.0
        self.frame = 0
        self.ss_frame = 0
        self.visuals: list[RepVisual] = []
        self.detector = None
        self.current_interactions = []
        self._label_actors = []
        self._legend_actors = []
        self._measure_actors = []
        self._callouts = []
        self._callout_actors = []
        self._callout_keys = []
        self._callout_boxes = []
        self._chain_order = {c: i for i, c in enumerate(dict.fromkeys(system.chains.tolist()))}
        # labels and 2D text live in an overlay layer so they are never hidden behind geometry
        self.overlay = vtkRenderer()
        self.overlay.SetLayer(1)
        self.overlay.InteractiveOff()
        self.overlay.SetActiveCamera(renderer.GetActiveCamera())
        win = renderer.GetRenderWindow()
        if win is not None:
            win.SetNumberOfLayers(max(2, win.GetNumberOfLayers()))
            win.AddRenderer(self.overlay)
        self._obs = renderer.AddObserver("StartEvent", self._apply_slab)
        self._label_source = win if win is not None else renderer
        self._label_obs = self._label_source.AddObserver("StartEvent", self._layout_callouts)
        self.light_kit = vtkLightKit()
        self.light_kit.AddLightsToRenderer(renderer)
        renderer.SetUseDepthPeeling(True)
        renderer.SetMaximumNumberOfPeels(8)
        renderer.SetOcclusionRatio(0.0)
        # interaction dashes
        self.dash_actor, self.dash_mapper = _glyph_actor(_cylinder_source(self.quality["cylinder"]), orient=True)
        _apply_material(self.dash_actor, "matte", 1.0)
        self.dash_actor.GetProperty().SetAmbient(0.5)
        renderer.AddActor(self.dash_actor)
        self.ring_actor, self.ring_mapper = _glyph_actor(_sphere_source(self.quality["sphere"]))
        _apply_material(self.ring_actor, "matte", 1.0)
        renderer.AddActor(self.ring_actor)
        self.meas_actor, self.meas_mapper = _glyph_actor(_cylinder_source(self.quality["cylinder"]), orient=True)
        _apply_material(self.meas_actor, "matte", 1.0)
        renderer.AddActor(self.meas_actor)
        # overlays
        self.title_actor = vtkTextActor()
        self.time_actor = vtkTextActor()
        for a in (self.title_actor, self.time_actor):
            a.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            self.overlay.AddViewProp(a)
        self.apply_render_settings()

    def dispose(self):
        """Detach everything this scene added to the renderer / window."""
        self.ren.RemoveObserver(self._obs)
        self._label_source.RemoveObserver(self._label_obs)
        self.ren.RemoveAllViewProps()
        self.ren.RemoveAllLights()
        self.overlay.RemoveAllViewProps()
        win = self.ren.GetRenderWindow()
        if win is not None:
            win.RemoveRenderer(self.overlay)

    # ------------------------------------------------------------------ colours
    def ss_codes(self, frame):
        return self.sys.secondary_structure(frame)

    @property
    def ss_every_frame(self):
        return self.sess.ss_every_frame

    def atom_colors(self, rep: Representation, idx: np.ndarray, frame=None) -> np.ndarray:
        s = self.sys
        idx = np.asarray(idx, dtype=np.int64)
        el = s.elements[idx]
        base = chem.element_colors(el, rep.carbon_color or None)
        scheme = rep.color
        if scheme == "element":
            return base
        if scheme == "uniform":
            return np.tile(np.array(chem.hex_to_rgb(rep.uniform_color), np.uint8), (len(idx), 1))
        resindex = s.u.atoms.resindices[idx]
        resnames = np.char.upper(s.u.atoms.resnames[idx].astype(str))
        if scheme == "secondary":
            ri, codes = self.ss_codes(self.frame if frame is None else frame) if self.ss_every_frame else self.ss_codes(self.ss_frame)
            lut = dict(zip(ri.tolist(), codes.tolist()))
            out = base.copy()
            prot = s.is_protein[idx]
            for k in np.where(prot)[0]:
                out[k] = chem.hex_to_rgb(chem.SS_COLORS.get(lut.get(int(resindex[k]), "-"), "#d8d8d8"))
            return out
        if scheme == "chain":
            pal = chem.CHAIN_PALETTE
            return np.array([chem.hex_to_rgb(pal[self._chain_order.get(c, 0) % len(pal)]) for c in s.chains[idx]],
                            np.uint8).reshape(-1, 3)
        if scheme == "residue type":
            def cls(rn):
                if rn in chem.ACIDIC_RES:
                    return "acidic"
                if rn in chem.BASIC_RES:
                    return "basic"
                if rn in chem.HYDROPHOBIC_RES:
                    return "hydrophobic"
                if rn in chem.POLAR_RES:
                    return "polar"
                return "other"
            out = np.array([chem.hex_to_rgb(RES_TYPE_COLORS[cls(rn)]) for rn in resnames], np.uint8).reshape(-1, 3)
            nonprot = ~s.is_protein[idx]
            out[nonprot] = base[nonprot]
            return out
        if scheme == "residue index":
            uniq = np.unique(resindex)
            rank = np.searchsorted(uniq, resindex)
            return _cmap(rank, "turbo", 0, max(len(uniq) - 1, 1))
        if scheme == "hydrophobicity":
            vals = np.array([chem.KYTE_DOOLITTLE.get(rn[:3], 0.0) for rn in resnames])
            out = _cmap(vals, "BrBG_r", -4.5, 4.5)
            nonprot = ~s.is_protein[idx]
            out[nonprot] = base[nonprot]
            return out
        if scheme == "rmsf":
            vals = s.rmsf()[idx]
            hi = np.percentile(s.rmsf()[s.is_protein] if s.is_protein.any() else vals, 97)
            return _cmap(vals, "coolwarm", 0, hi)
        if scheme == "bfactor":
            vals = s.bfactors[idx] if s.bfactors is not None else np.zeros(len(idx))
            return _cmap(vals, "coolwarm")
        return base

    # ------------------------------------------------------------------ building
    def rebuild(self):
        for v in self.visuals:
            v.remove()
        self.rebuild_interactions(draw=False)
        self.visuals = [RepVisual(self, r) for r in self.sess.reps]
        for v in self.visuals:
            v.build(self.frame)
        self._update_interactions()
        self.update_overlays()

    def compute_interacting(self, n_samples=40):
        """Residues that interact with the ligand in any of up to `n_samples` frames across the trajectory.

        Used by the `interacting` selection keyword and by pocket labels, so the displayed residue set stays
        constant during a movie instead of flickering frame to frame.
        """
        st = self.sess.interactions
        self.pocket_kinds = set()
        n_samples = st.pocket_samples
        wants = st.label_mode == "pocket" or any(re.search(r"\b(interacting|contact_atoms)\b", r.selection) for r in self.sess.reps)
        if not wants or self.detector is None:
            self.sys.interacting_resindices = set() if wants else None
            self.sys.contact_atom_indices = set()
            return
        key = (st.ligand, st.receptor, repr(sorted(st.params.items())), self.sys.pbc_fix, self.sys.align,
               self.sys.align_selection, n_samples)
        cache = getattr(self.sys, "_interacting_cache", {})
        if key not in cache:
            n = self.sys.n_frames
            frames = range(n) if n_samples == 0 else np.linspace(0, n - 1, min(n, max(1, n_samples)), dtype=int)
            res = set()
            kinds = set()
            atoms = set()
            for f in frames:
                interactions = self.detector.detect(self.sys.analysis_positions(int(f)))
                res |= {i.resindex for i in interactions}
                kinds |= {i.kind for i in interactions}
                atoms |= {a for i in interactions for a in i.rec_atoms}
            # Include heavy neighbours so backbone contact atoms are bonded into the pocket view.
            atoms |= {b for a in tuple(atoms) for b in self.sys.neighbors[a] if self.sys.elements[b] != "H"}
            cache[key] = (res, kinds, atoms)
            self.sys._interacting_cache = cache
        self.sys.interacting_resindices, self.pocket_kinds, self.sys.contact_atom_indices = cache[key]
        self.sys._sel_cache.clear()

    def rebuild_rep(self, i: int):
        if i < len(self.visuals):
            self.visuals[i].remove()
            self.visuals[i] = RepVisual(self, self.sess.reps[i])
            self.visuals[i].build(self.frame)

    def insert_rep(self, i: int):
        v = RepVisual(self, self.sess.reps[i])
        self.visuals.insert(i, v)
        v.build(self.frame)

    def remove_rep(self, i: int):
        v = self.visuals.pop(i)
        v.remove()

    def rebuild_interactions(self, draw=True):
        st = self.sess.interactions
        self.detector = None
        if st.show:
            try:
                lig = self.sys.select(st.ligand)
                rec = self.sys.select(st.receptor)
                if len(lig) and len(rec):
                    self.detector = InteractionDetector(self.sys, lig, rec, InteractionParams.from_dict(st.params),
                                                        ref_frame=self.frame)
            except Exception:
                self.detector = None
        self.compute_interacting()
        if draw:
            self._update_interactions()

    def set_frame(self, frame: int):
        self.frame = int(np.clip(frame, 0, max(self.sys.n_frames - 1, 0)))
        for v in self.visuals:
            v.update(self.frame)
        self._update_interactions()
        self.update_overlays()

    def set_playback_quality(self, playing):
        if self.quality_name == "export":
            return
        self.quality = self.QUALITY["playback" if playing else "screen"]
        if not playing:
            self.set_frame(self.frame)

    def refresh_annotations(self):
        self._update_interactions()
        self.update_overlays()

    # ------------------------------------------------------------------ interactions / labels
    def _clear_labels(self):
        self._label_cursor = 0
        self._callouts = []
        self._callout_keys = []
        self._callout_boxes = []
        for a in self._label_actors + self._callout_actors:
            a.SetVisibility(False)

    def _billboard(self, text, pos, size, color, bold=True):
        cursor = getattr(self, "_label_cursor", 0)
        if cursor == len(self._label_actors):
            a = vtkBillboardTextActor3D()
            self.overlay.AddActor(a)
            self._label_actors.append(a)
        a = self._label_actors[cursor]
        self._label_cursor = cursor + 1
        a.SetVisibility(True)
        a.SetInput(text)
        a.SetPosition(*map(float, pos))
        tp = a.GetTextProperty()
        tp.SetFontSize(max(6, int(round(size * self.text_scale))))
        tp.SetColor(*chem.hex_to_float(color))
        tp.SetBold(bold)
        _set_font(tp)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        tp.SetShadow(False)
        tp.SetBackgroundOpacity(0.0)
        if self.sess.render.label_box:
            tp.SetBackgroundColor(*chem.hex_to_float(self.sess.render.background))
            tp.SetBackgroundOpacity(0.7)
        a.SetForceOpaque(True)
        return a

    def _update_interactions(self):
        st = self.sess.interactions
        self._clear_labels()
        self.current_interactions = []
        pos = self.sys.positions(self.frame)
        if self.detector is not None and st.show:
            self.current_interactions = [i for i in self.detector.detect(self.sys.analysis_positions(self.frame))
                                         if i.distance <= MAX_DRAWN_INTERACTION]
        inter = self.current_interactions
        if inter:
            p1 = np.array([pos[list(i.lig_atoms)].mean(axis=0) for i in inter])
            p2 = np.array([pos[list(i.rec_atoms)].mean(axis=0) for i in inter])
            starts, vecs, cols = [], [], []
            for i, a, b in zip(inter, p1, p2):
                s, v = _dashes([a], [b])
                starts.append(s)
                vecs.append(v)
                cols.append(np.tile(chem.hex_to_rgb(self.icolor(i.kind)), (len(s), 1)))
            starts = np.vstack(starts)
            vecs = np.vstack(vecs)
            cols = np.vstack(cols).astype(np.uint8)
            L = np.linalg.norm(vecs, axis=1)
            r = st.dash_radius
            _set_glyph_data(self.dash_mapper, starts, np.stack([L, np.full_like(L, r), np.full_like(L, r)], 1), cols, vecs)
            # ring centroids as small spheres
            ring_pts, ring_cols = [], []
            for i, a, b in zip(inter, p1, p2):
                if i.kind in ("Pi-stacking", "Cation-pi"):
                    c = chem.hex_to_rgb(self.icolor(i.kind))
                    if i.lig_label.startswith("ring"):
                        ring_pts.append(a)
                        ring_cols.append(c)
                    if i.rec_label in ("ring", "parallel", "T-shaped"):
                        ring_pts.append(b)
                        ring_cols.append(c)
            _set_glyph_data(self.ring_mapper, np.array(ring_pts).reshape(-1, 3), np.full(len(ring_pts), r * 2.5, np.float32),
                            np.array(ring_cols, np.uint8).reshape(-1, 3))
        if st.labels and self.detector is not None and st.show and (inter or st.label_mode == "pocket"):
            self._interaction_labels(pos)
        if not inter:
            _set_glyph_data(self.dash_mapper, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 3)))
            _set_glyph_data(self.ring_mapper, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), np.uint8))
        self.dash_actor.SetVisibility(bool(inter))
        self.ring_actor.SetVisibility(bool(inter))
        self._draw_measurements(pos)

    def _draw_measurements(self, pos):
        physical = self.sys.analysis_positions(self.frame)
        pairs = [m for m in self.sess.measurements if len(m) == 2]
        if pairs:
            a = np.array([pos[i] for i, _ in pairs])
            b = np.array([pos[j] for _, j in pairs])
            s, v = _dashes(a, b, 0.35, 0.25)
            L = np.linalg.norm(v, axis=1)
            _set_glyph_data(self.meas_mapper, s, np.stack([L, np.full_like(L, 0.05), np.full_like(L, 0.05)], 1),
                            np.tile(np.array([230, 150, 0], np.uint8), (len(s), 1)), v)
            for (i, j), pa, pb in zip(pairs, a, b):
                self._billboard(f"{np.linalg.norm(physical[i] - physical[j]):.2f} Å", (pa + pb) / 2, self.sess.interactions.label_size,
                                "#c07800")
        else:
            _set_glyph_data(self.meas_mapper, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 3)))
        self.meas_actor.SetVisibility(bool(pairs))
        for i in self.sess.atom_labels:
            self._billboard(self.atom_label(i), pos[i], self.sess.interactions.label_size * 0.85,
                            self.sess.interactions.label_color, bold=False)

    def icolor(self, kind):
        return self.sess.interactions.colors.get(kind) or chem.INTERACTION_COLORS[kind]

    def residue_label(self, resindex):
        r = self.sys.u.residues[int(resindex)]
        text = f"{r.resname[:1]}{r.resname[1:3].lower()}{r.resid}"
        if self.sess.interactions.label_chain and len(set(self.sys.chains[self.sys.is_protein])) > 1:
            text += f":{self.sys.chains[r.atoms.indices[0]]}"
        return text

    def _interaction_labels(self, pos):
        st = self.sess.interactions
        lig_c = pos[self.detector.lig].mean(axis=0)
        if st.label_mode == "pocket" and self.sys.interacting_resindices:
            resids = sorted(self.sys.interacting_resindices)
        else:
            resids = list(dict.fromkeys(i.resindex for i in self.current_interactions))
        for ri in resids:
            res_atoms = self.sys.u.residues[int(ri)].atoms.indices
            heavy = res_atoms[self.sys.elements[res_atoms] != "H"]
            # anchor on the side-chain tip (the part pointing at the ligand), pushed a little outward
            ca = res_atoms[self.sys.u.atoms.names[res_atoms] == "CA"]
            side = heavy[~np.isin(self.sys.u.atoms.names[heavy], ["N", "C", "O", "CA"])]
            pts = pos[side] if len(side) else pos[heavy]
            anchor = pts.mean(axis=0) if not len(ca) else 0.35 * pos[ca[0]] + 0.65 * pts.mean(axis=0)
            target = anchor.copy()
            d = anchor - lig_c
            anchor = anchor + d / max(np.linalg.norm(d), 1e-6) * 1.8
            if st.callout_labels:
                self._callouts.append((self.residue_label(ri), target))
                self._callout_keys.append(str(ri))
            else:
                self._billboard(self.residue_label(ri), anchor, st.label_size, st.label_color)

    def _layout_callouts(self, *args):
        """Lay labels beside the pocket at render time, including camera rotation/export."""
        from .labels import layout_labels
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D
        if not self._callouts:
            return
        w, h = self.ren.GetSize()
        fs = max(8, int(self.sess.interactions.label_size * self.text_scale))
        anchors = []
        for text, anchor in self._callouts:
            self.ren.SetWorldPoint(*map(float, anchor), 1)
            self.ren.WorldToDisplay()
            anchors.append(self.ren.GetDisplayPoint()[:2])
        positions = layout_labels(anchors, [len(t) * fs * 0.65 for t, _ in self._callouts], w, h, fs)
        self._callout_boxes = []
        while len(self._callout_actors) < len(positions) * 2:
            text = vtkTextActor()
            line = vtkActor2D()
            line.SetMapper(vtkPolyDataMapper2D())
            self.overlay.AddViewProp(line)
            self.overlay.AddViewProp(text)
            self._callout_actors.extend([text, line])
        for k, ((text, _), anchor, xy) in enumerate(zip(self._callouts, anchors, positions)):
            key = self._callout_keys[k]
            pin = self.sess.interactions.label_positions.get(key)
            if pin is not None:
                xy = (max(0, min(pin[0] * w, w - len(text)*fs*0.65)),
                      max(0, min(pin[1] * h, h - fs*1.5)))
            self._callout_boxes.append((key, xy[0], xy[1], len(text)*fs*0.65, fs*1.5))
            actor, line = self._callout_actors[2*k:2*k+2]
            actor.SetInput(text)
            actor.SetPosition(*xy)
            tp = actor.GetTextProperty()
            tp.SetFontSize(fs)
            tp.SetBold(True)
            _set_font(tp)
            tp.SetColor(*chem.hex_to_float(self.sess.interactions.label_color))
            tp.SetBackgroundColor(*chem.hex_to_float(self.sess.render.background))
            tp.SetBackgroundOpacity(0.85 if self.sess.render.label_box else 0)
            actor.SetVisibility(True)
            end_x = xy[0] + (len(text) * fs * 0.65 if xy[0] < anchor[0] else 0)
            pd = vtkPolyData()
            pd.SetPoints(_vtk_points([[*anchor, 0], [end_x, xy[1] + fs * 0.5, 0]]))
            pd.SetLines(_cells(np.array([[0, 1]]), 2))
            line.GetMapper().SetInputData(pd)
            line.GetProperty().SetColor(0.45, 0.48, 0.52)
            line.SetVisibility(True)

    def label_at(self, x, y):
        for key, bx, by, bw, bh in reversed(self._callout_boxes):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return None

    def move_label(self, key, x, y):
        w, h = self.ren.GetSize()
        self.sess.interactions.label_positions[key] = [float(x / max(w, 1)), float(y / max(h, 1))]

    def atom_label(self, i: int) -> str:
        a = self.sys.u.atoms[int(i)]
        rn = a.resname
        return f"{rn[:1]}{rn[1:].lower()}{a.resid}:{a.name}"

    def atom_info(self, i: int) -> str:
        a = self.sys.u.atoms[int(i)]
        p = self.sys.positions(self.frame)[int(i)]
        return (f"atom {a.index}  {a.name} ({self.sys.elements[i]})  {a.resname} {a.resid}  chain {self.sys.chains[i]}  "
                f"xyz = ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    # ------------------------------------------------------------------ overlays / settings
    def update_overlays(self):
        r = self.sess.render
        col = chem.hex_to_float(r.text_color)
        ts = self.text_scale
        # title
        self.title_actor.SetInput(r.title or "")
        tp = self.title_actor.GetTextProperty()
        tp.SetFontSize(max(6, int(r.title_size * ts)))
        tp.SetColor(*col)
        tp.SetBold(True)
        _set_font(tp)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToTop()
        self.title_actor.SetPosition(0.5, 0.97)
        self.title_actor.SetVisibility(bool(r.title))
        # time stamp
        t_ps = self.sys.frame_time(self.frame)
        labels = getattr(self.sys, "frame_labels", None)
        try:
            txt = labels[self.frame] if labels else r.time_format.format(ns=t_ps / 1000.0, ps=t_ps, frame=self.frame)
        except Exception:
            txt = f"t = {t_ps / 1000:.2f} ns"
        self.time_actor.SetInput(txt)
        tp = self.time_actor.GetTextProperty()
        tp.SetFontSize(max(6, int(0.8 * r.title_size * ts)))
        tp.SetColor(*col)
        tp.SetBold(False)
        _set_font(tp)
        tp.SetJustificationToRight()
        tp.SetVerticalJustificationToBottom()
        self.time_actor.SetPosition(0.97, 0.03)
        self.time_actor.SetVisibility(bool(r.show_time) and self.sys.n_frames > 1)
        # legend of interaction types present in this frame
        kinds = [k for k in chem.INTERACTION_COLORS if any(i.kind == k for i in self.current_interactions)]
        if self.sess.interactions.label_mode == "pocket":
            kinds = [k for k in chem.INTERACTION_COLORS if k in getattr(self, "pocket_kinds", set())]
        legend_key = (tuple(kinds), r.show_legend, ts, r.title_size, r.text_color,
                      repr(self.sess.interactions.colors))
        if getattr(self, "_legend_key", None) == legend_key:
            return
        self._legend_key = legend_key
        for a in self._legend_actors:
            self.overlay.RemoveViewProp(a)
        self._legend_actors = []
        if r.show_legend and kinds:
            fs = max(6, int(0.62 * r.title_size * ts))
            for n, k in enumerate(reversed(kinds)):
                y = int(20 * ts + n * fs * 1.45)
                # coloured dash carries identity; the text stays in the text colour
                for text, colour, x in (("——", self.icolor(k), int(20 * ts)),
                                        (k, r.text_color, int(20 * ts + fs * 3.0))):
                    a = vtkTextActor()
                    a.SetInput(text)
                    a.GetPositionCoordinate().SetCoordinateSystemToDisplay()
                    a.SetPosition(x, y)
                    tp = a.GetTextProperty()
                    tp.SetFontSize(fs)
                    tp.SetColor(*chem.hex_to_float(colour))
                    tp.SetBold(text != k)
                    _set_font(tp)
                    tp.SetVerticalJustificationToBottom()
                    self.overlay.AddViewProp(a)
                    self._legend_actors.append(a)

    def apply_render_settings(self):
        r = self.sess.render
        bg = chem.hex_to_float(r.background)
        if r.background2:
            self.ren.GradientBackgroundOn()
            self.ren.SetBackground(*chem.hex_to_float(r.background2))
            self.ren.SetBackground2(*bg)
        else:
            self.ren.GradientBackgroundOff()
            self.ren.SetBackground(*bg)
        self.ren.SetUseSSAO(bool(r.ssao))
        if r.ssao:
            self.ren.SetSSAORadius(4.0)
            self.ren.SetSSAOBias(0.05)
            self.ren.SetSSAOKernelSize(128)
            self.ren.SetSSAOBlur(True)
        self.ren.SetUseFXAA(bool(r.fxaa))
        self.light_kit.SetKeyLightIntensity(0.75 * r.light_intensity)
        self.light_kit.Update()
        self.update_overlays()

    def _apply_slab(self, *args):
        """VMD-like clipping slab: keep only a slice of `slab` Angstrom around the focal point."""
        slab = float(self.sess.render.slab or 0)
        if slab > 0:
            cam = self.ren.GetActiveCamera()
            d = cam.GetDistance()
            cam.SetClippingRange(max(0.05, d - slab / 2), d + slab / 2)

    def set_text_scale(self, s: float):
        if abs(s - self.text_scale) > 1e-3:
            self.text_scale = s
            self._update_interactions()
            self.update_overlays()

    # ------------------------------------------------------------------ camera
    def apply_view(self, v: ViewState):
        cam = self.ren.GetActiveCamera()
        cam.SetPosition(*v.position)
        cam.SetFocalPoint(*v.focal_point)
        cam.SetViewUp(*v.view_up)
        cam.SetViewAngle(v.view_angle)
        cam.SetParallelProjection(bool(v.parallel))
        cam.SetParallelScale(v.parallel_scale)
        self.ren.ResetCameraClippingRange()

    def capture_view(self) -> ViewState:
        cam = self.ren.GetActiveCamera()
        return ViewState(list(cam.GetPosition()), list(cam.GetFocalPoint()), list(cam.GetViewUp()),
                         cam.GetViewAngle(), bool(cam.GetParallelProjection()), cam.GetParallelScale(), auto=False)

    def pocket_view(self, margin=1.55):
        """Look into the binding site from outside the protein (protein centre -> ligand direction)."""
        lig = self.detector.lig if self.detector is not None else self.sys.select("ligand")
        if not len(lig):
            return self.auto_view(force_overview=True)
        pos = self.sys.positions(self.frame)
        lc = pos[lig].mean(axis=0)
        prot = np.where(self.sys.is_protein)[0]
        d = lc - (pos[prot].mean(axis=0) if len(prot) else lc - np.array([0, 0, 1.0]))
        if np.linalg.norm(d) < 1.0:
            d = np.array([0, 0, 1.0])
        d /= np.linalg.norm(d)
        up = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(up) < 0.1:
            up = np.cross(d, [0, 1.0, 0])
        up = np.cross(up, d)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(*lc)
        cam.SetPosition(*(lc + d * 100.0))
        cam.SetViewUp(*(up / np.linalg.norm(up)))
        cam.SetParallelProjection(bool(self.sess.view.parallel))
        show = lig
        if self.sys.interacting_resindices:
            show = np.concatenate([lig, np.where(np.isin(self.sys.u.atoms.resindices,
                                                          list(self.sys.interacting_resindices)))[0]])
        self.focus(show, margin=margin * 0.62)

    def auto_view(self, force_overview=False):
        if not force_overview and getattr(self.sess.view, "auto_mode", "overview") == "pocket":
            return self.pocket_view()
        """Default framing: whole displayed system, slightly rotated (like the VMD script), orthographic."""
        cam = self.ren.GetActiveCamera()
        cam.SetPosition(0, 0, 1)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 1, 0)
        cam.SetParallelProjection(bool(self.sess.view.parallel))
        cam.Azimuth(-35)
        cam.Elevation(15)
        cam.OrthogonalizeViewUp()
        self.focus(None, margin=1.05)

    def principal_view(self, idx=None, margin=1.4):
        """Look along the molecule's smallest principal axis, longest axis horizontal (good for small molecules)."""
        if idx is None or not len(idx):
            idx = self.displayed_atoms()
        pos = self.sys.positions(self.frame)[idx].astype(float)
        c = pos.mean(axis=0)
        _, _, vt = np.linalg.svd(pos - c)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + vt[2] * 100.0))
        cam.SetViewUp(*vt[1])
        self.focus(idx, margin)

    def displayed_atoms(self) -> np.ndarray:
        idx = [v.idx for v in self.visuals if v.rep.visible and v.rep.style not in ("cartoon", "tube", "surface")]
        idx += [v.bb_ca for v in self.visuals if v.rep.visible and v.rep.style in ("cartoon", "tube") and hasattr(v, "bb_ca")]
        return np.unique(np.concatenate(idx)) if idx else np.zeros(0, np.int64)

    def focus(self, idx=None, margin=1.15):
        """Centre the camera on atoms `idx` (default: everything displayed), keeping the view direction."""
        if idx is None or not len(idx):
            idx = self.displayed_atoms()
        if not len(idx):
            self.ren.ResetCamera()
            return
        pos = self.sys.positions(self.frame)[idx]
        c = pos.mean(axis=0)
        radius = max(np.linalg.norm(pos - c, axis=1).max(), 4.0) * margin
        cam = self.ren.GetActiveCamera()
        d = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        d = d / max(np.linalg.norm(d), 1e-6)
        dist = radius / np.tan(np.radians(cam.GetViewAngle() / 2))
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + d * dist))
        cam.SetParallelScale(radius)
        self.ren.ResetCameraClippingRange()
