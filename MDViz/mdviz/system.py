"""Loading molecular systems (VMD-compatible formats) and serving processed frames."""
from __future__ import annotations

import io
import os
import re
import warnings
from collections import OrderedDict

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors

from . import chem

warnings.filterwarnings("ignore", module="MDAnalysis")

# File-dialog filters: the formats VMD reads that we support.
STRUCTURE_FORMATS = ["gro", "pdb", "ent", "pqr", "pdbqt", "psf", "tpr", "prmtop", "parm7", "top", "mol2",
                     "xyz", "crd", "cif", "mmcif", "sdf", "mol", "dms", "data", "xpdb", "gsd", "mmtf"]
TRAJECTORY_FORMATS = ["xtc", "trr", "dcd", "nc", "ncdf", "netcdf", "mdcrd", "crd", "trj", "inpcrd", "rst7", "restrt",
                      "gro", "pdb", "xyz", "lammpstrj", "dump", "tng", "h5md", "trz", "coor", "namdbin"]
# Topologies without coordinates need a coordinate file to go with them
COORDLESS_TOPOLOGIES = {"psf", "prmtop", "parm7", "top", "tpr_nocoords"}


def _read_sdf(path: str) -> mda.Universe:
    """Minimal V2000 SDF/MOL reader (multi-record files become frames)."""
    with open(path, "r", errors="replace") as fh:
        records = [r for r in fh.read().split("$$$$") if r.strip()]
    frames, names, elements, bonds = [], None, None, None
    for rec in records:
        lines = rec.strip("\n").splitlines()
        if len(lines) < 4:
            continue
        counts = lines[3]
        na, nb = int(counts[0:3]), int(counts[3:6])
        xyz, els = [], []
        for ln in lines[4:4 + na]:
            xyz.append([float(ln[0:10]), float(ln[10:20]), float(ln[20:30])])
            els.append(ln[31:34].strip().upper() or "X")
        if elements is None:
            elements = els
            bonds = [(int(ln[0:3]) - 1, int(ln[3:6]) - 1) for ln in lines[4 + na:4 + na + nb]]
            counters = {}
            names = []
            for e in els:
                counters[e] = counters.get(e, 0) + 1
                names.append(f"{e.capitalize()}{counters[e]}")
        if len(xyz) == len(elements):
            frames.append(xyz)
    if not frames:
        raise ValueError(f"No molecule found in {path}")
    n = len(elements)
    u = mda.Universe.empty(n, n_residues=1, n_segments=1, atom_resindex=np.zeros(n, int),
                           residue_segindex=[0], trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("types", elements)
    u.add_TopologyAttr("elements", [e.capitalize() for e in elements])
    u.add_TopologyAttr("resnames", ["LIG"])
    u.add_TopologyAttr("resids", [1])
    u.add_TopologyAttr("segids", ["LIG"])
    u.add_TopologyAttr("masses", [chem.ELEMENTS.get(e, chem.ELEMENTS["X"])[0] for e in elements])
    u.add_TopologyAttr("bonds", bonds)
    from MDAnalysis.coordinates.memory import MemoryReader
    u.load_new(np.asarray(frames, dtype=np.float32), format=MemoryReader)
    return u


def _read_cif(path: str) -> mda.Universe:
    import gemmi
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_empty_chains()
    return mda.Universe(io.StringIO(st.make_pdb_string()), format="PDB")


def load_universe(topology: str, trajectories: list[str] | None = None) -> mda.Universe:
    trajectories = [t for t in (trajectories or []) if t]
    ext = os.path.splitext(topology)[1].lower().lstrip(".")
    if ext in ("sdf", "mol"):
        return _read_sdf(topology)
    if ext in ("cif", "mmcif"):
        u = _read_cif(topology)
        if trajectories:
            u.load_new(trajectories if len(trajectories) > 1 else trajectories[0])
        return u
    kwargs = {}
    if ext in ("prmtop", "parm7"):
        kwargs["topology_format"] = "PRMTOP"
    if trajectories:
        traj = trajectories if len(trajectories) > 1 else trajectories[0]
        return mda.Universe(topology, traj, **kwargs)
    u = mda.Universe(topology, **kwargs)
    if ext == "tpr":
        # TPR coordinates come back in nm; keep an in-memory copy converted to Angstrom
        u.transfer_to_memory()
        u.trajectory.coordinate_array[:] *= 10.0
    return u


class MolSystem:
    """A loaded structure/trajectory with the per-frame processing pipeline."""

    def __init__(self, topology: str, trajectories: list[str] | None = None, coordinates=None, frame_labels=None):
        self.topology = os.path.abspath(topology)
        self.trajectories = [os.path.abspath(t) for t in (trajectories or [])]
        self.u = load_universe(self.topology, self.trajectories)
        if coordinates is not None:  # in-memory frames (Angstrom), e.g. PES conformations
            from MDAnalysis.coordinates.memory import MemoryReader
            self.u.load_new(np.asarray(coordinates, dtype=np.float32), format=MemoryReader)
        self.frame_labels = frame_labels  # optional per-frame text replacing the time stamp
        self.n_atoms = self.u.atoms.n_atoms
        self._setup_topology()
        # processing options
        self.pbc_fix = False
        self.align = False
        self.align_selection = "protein and name CA"
        self.smoothing = 1
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._raw_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = 96
        self._ref = None
        self._ss_cache: dict[int, np.ndarray] = {}
        self._rmsf = None
        self._sel_cache: dict[tuple, np.ndarray] = {}
        self.interacting_resindices = None   # set by the scene: residues that interact with the ligand
        self.contact_atom_indices = set()

    # ------------------------------------------------------------------ topology
    def _setup_topology(self):
        u = self.u
        atoms = u.atoms
        names = atoms.names
        resnames = atoms.resnames
        masses = atoms.masses if hasattr(atoms, "masses") else None
        existing = atoms.elements if hasattr(atoms, "elements") else None
        res_sizes = np.array([r.atoms.n_atoms for r in u.residues])[atoms.resindices]
        self.elements = chem.guess_elements(names, resnames, masses, existing, res_sizes)
        try:
            u.add_TopologyAttr("elements", [e.capitalize() for e in self.elements])
        except Exception:
            pass
        self.is_water = np.isin(np.char.upper(resnames.astype(str)), list(chem.WATER_RESNAMES))
        self.is_ion = np.isin(np.char.upper(resnames.astype(str)), list(chem.ION_RESNAMES)) & (res_sizes == 1)
        self.is_protein = np.zeros(self.n_atoms, bool)
        try:
            self.is_protein[u.select_atoms("protein").indices] = True
        except Exception:
            pass
        # Bonds: use the topology's, otherwise guess for the solute (skip bulk water for speed)
        has_bonds = hasattr(u, "bonds") and len(u.bonds) > 0
        if has_bonds:
            # Some PDB writers (YASARA, docking tools) give CONECT records for the ligand only; guess the rest
            solute_mask = ~(self.is_water | self.is_ion)
            n_bonds = np.zeros(self.n_atoms, int)
            np.add.at(n_bonds, np.asarray(u.bonds.indices).ravel(), 1)
            unbonded = (n_bonds == 0) & solute_mask & (res_sizes > 1)
            if unbonded.sum() > 0.05 * max(solute_mask.sum(), 1):
                todo = u.atoms[np.isin(atoms.resindices, np.unique(atoms.resindices[unbonded]))]
                try:
                    from MDAnalysis.guesser.default_guesser import DefaultGuesser
                    new = DefaultGuesser(None).guess_bonds(todo, todo.positions, box=u.dimensions)
                    u.add_bonds([(todo[i].index, todo[j].index) for i, j in new])
                except Exception:
                    try:
                        todo.guess_bonds()
                    except Exception:
                        pass
        if not has_bonds:
            solute = u.atoms[~(self.is_water | self.is_ion)]
            try:
                if solute.n_atoms:
                    solute.guess_bonds()
            except Exception:
                try:
                    from MDAnalysis.topology.guessers import guess_bonds
                    b = guess_bonds(solute, solute.positions)
                    u.add_TopologyAttr("bonds", [(solute[i].index, solute[j].index) for i, j in b])
                except Exception:
                    pass
            # water bonds are cheap to infer from residue membership
            water = u.atoms[self.is_water]
            wb = []
            for res in water.residues:
                idx = res.atoms.indices
                ox = [i for i in idx if self.elements[i] == "O"]
                if ox:
                    wb += [(ox[0], i) for i in idx if self.elements[i] == "H"]
            if wb:
                try:
                    u.add_bonds(wb)
                except Exception:
                    pass
        self.bonds = np.asarray(u.bonds.indices, dtype=np.int64).reshape(-1, 2) if hasattr(u, "bonds") else np.zeros((0, 2), np.int64)
        self.neighbors = [[] for _ in range(self.n_atoms)]
        for a, b in self.bonds:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        # chain labels: chainIDs when informative, else parse segids ("..._chain_A" -> A)
        chains = None
        if hasattr(atoms, "chainIDs") and len(set(atoms.chainIDs)) > 1:
            chains = atoms.chainIDs.astype(str)
        elif hasattr(atoms, "segids"):
            segs = atoms.segids.astype(str)
            m = [re.search(r"chain_?([A-Za-z0-9])$", s) for s in segs]
            chains = np.array([mm.group(1) if mm else s for mm, s in zip(m, segs)])
        self.chains = chains if chains is not None else np.array(["A"] * self.n_atoms)
        # ligand guess: non-protein, non-water, non-ion residues
        other = ~(self.is_protein | self.is_water | self.is_ion)
        self.ligand_resnames = sorted(set(resnames[other].astype(str)))
        self.polar_h = np.zeros(self.n_atoms, bool)
        for i in np.where(self.elements == "H")[0]:
            if any(self.elements[j] in ("N", "O", "S") for j in self.neighbors[i]):
                self.polar_h[i] = True
        self.nonpolar_h = (self.elements == "H") & ~self.polar_h
        self.vdw = chem.vdw_radii(self.elements)
        self.bfactors = atoms.tempfactors if hasattr(atoms, "tempfactors") else None

    @property
    def n_frames(self) -> int:
        return len(self.u.trajectory)

    @property
    def dt(self) -> float:
        try:
            return float(self.u.trajectory.dt)
        except Exception:
            return 1.0

    def frame_time(self, frame: int) -> float:
        """Time in ps."""
        try:
            return float(self.u.trajectory[frame].time)
        except Exception:
            return frame * self.dt

    def summary(self) -> str:
        lig = ", ".join(self.ligand_resnames) or "none"
        return (f"{os.path.basename(self.topology)} | {self.n_atoms} atoms | {self.u.residues.n_residues} residues | "
                f"{self.n_frames} frames | ligand residues: {lig}")

    # ------------------------------------------------------------------ selections
    def expand_macros(self, text: str) -> str:
        t = text.strip() or "all"
        lig = " ".join(self.ligand_resnames)
        repl = {
            "ligand": f"(resname {lig})" if lig else "(not all)",
            "water": "(resname " + " ".join(sorted(chem.WATER_RESNAMES)) + ")",
            "ions": "(resname " + " ".join(sorted(chem.ION_RESNAMES)) + ")",
            "solute": "(not (resname " + " ".join(sorted(chem.WATER_RESNAMES | chem.ION_RESNAMES)) + "))",
            "hydrogen": "(element H)",
            "heavy": "(not element H)",
        }
        for key, val in repl.items():
            t = re.sub(rf"(?<![\w.]){key}(?![\w.])", val, t)
        if re.search(r"\binteracting\b", t):
            res = self.interacting_resindices
            idx = np.where(np.isin(self.u.atoms.resindices, list(res)))[0] if res is not None and len(res) else []
            t = re.sub(r"\binteracting\b", "(index " + " ".join(map(str, idx)) + ")" if len(idx) else "(not all)", t)
        if re.search(r"\bcontact_atoms\b", t):
            idx = sorted(self.contact_atom_indices)
            t = re.sub(r"\bcontact_atoms\b", "(index " + " ".join(map(str, idx)) + ")" if idx else "(not all)", t)
        for key, mask in (("nonpolarH", self.nonpolar_h), ("polarH", self.polar_h)):
            if re.search(rf"\b{key}\b", t):
                idx = np.where(mask)[0]
                val = "(index " + " ".join(map(str, idx)) + ")" if len(idx) else "(not all)"
                t = re.sub(rf"\b{key}\b", val, t)
        return t

    def select(self, text: str, frame: int | None = None) -> np.ndarray:
        """Atom indices for a VMD-like selection, evaluated at `frame` (processed coordinates)."""
        expr = self.expand_macros(text)
        needs_coords = bool(re.search(r"\b(around|sphzone|sphlayer|cylayer|cyzone|point|prop|within|cog|com)\b", expr))
        expr = re.sub(r"\bwithin\s+([\d.]+)\s+of\b", r"around \1", expr)  # VMD syntax -> MDAnalysis
        key = (expr, frame if needs_coords else None)
        if key in self._sel_cache:
            return self._sel_cache[key]
        if needs_coords and frame is not None:
            pos = self.positions(frame)
            original = self.u.atoms.positions.copy()
            try:
                self.u.atoms.positions = pos
                ag = self.u.select_atoms(expr, periodic=False)
            finally:
                # MemoryReader positions are writable views of the trajectory.
                self.u.atoms.positions = original
        else:
            ag = self.u.select_atoms(expr, periodic=False) if needs_coords else self.u.select_atoms(expr)
        idx = np.asarray(ag.indices, dtype=np.int64)
        if len(self._sel_cache) > 256:
            self._sel_cache.clear()
        self._sel_cache[key] = idx
        return idx

    def detect_pbc_problems(self, n_samples=5, max_bond=4.0) -> dict:
        """Look for molecules split across the periodic box in the raw trajectory.

        Returns {"broken": bool, "longest_bond": A, "frames": [...]}. Split molecules show up as bonds
        longer than a few Angstrom; drawn as sticks they become long streaks across the picture.
        """
        solute = ~(self.is_water | self.is_ion)
        b = self.bonds[solute[self.bonds[:, 0]]] if len(self.bonds) else self.bonds
        out = {"broken": False, "longest_bond": 0.0, "frames": []}
        if not len(b):
            return out
        n = self.n_frames
        for f in sorted(set(np.linspace(0, n - 1, min(n, n_samples)).astype(int))):
            p = self.u.trajectory[int(f)].positions
            longest = float(np.linalg.norm(p[b[:, 0]] - p[b[:, 1]], axis=1).max())
            out["longest_bond"] = max(out["longest_bond"], longest)
            if longest > max_bond:
                out["broken"] = True
                out["frames"].append(int(f))
        return out

    # ------------------------------------------------------------------ frames
    def set_processing(self, pbc_fix=None, align=None, align_selection=None, smoothing=None):
        if pbc_fix is not None:
            self.pbc_fix = bool(pbc_fix)
        if align is not None:
            self.align = bool(align)
        if align_selection is not None:
            self.align_selection = align_selection
        if smoothing is not None:
            self.smoothing = max(1, int(smoothing))
        self._cache.clear()
        self._raw_cache.clear()
        self._sel_cache.clear()
        self._ref = None
        self._rmsf = None
        self._ss_cache.clear()
        self._interacting_cache = {}

    def _fragments(self):
        if not hasattr(self, "_frag_list"):
            self._frag_list = []
            if len(self.bonds):
                solute = self.u.atoms[~(self.is_water | self.is_ion)]
                try:
                    self._frag_list = list(solute.fragments)
                except Exception:
                    self._frag_list = []
        return self._frag_list

    def _raw(self, frame: int) -> np.ndarray:
        if frame in self._raw_cache:
            self._raw_cache.move_to_end(frame)
            return self._raw_cache[frame]
        ts = self.u.trajectory[frame]
        pos = ts.positions.astype(np.float32).copy()
        if self.pbc_fix and ts.dimensions is not None and np.all(ts.dimensions[:3] > 0):
            pos = self._fix_pbc(pos, ts.dimensions)
        if self.align:
            pos = self._align(pos)
        self._raw_cache[frame] = pos
        if len(self._raw_cache) > self._cache_size:
            self._raw_cache.popitem(last=False)
        return pos

    def _unwrap_levels(self):
        """BFS spanning trees of the solute fragments, grouped by depth, for vectorised unwrapping."""
        if hasattr(self, "_levels"):
            return self._levels
        from collections import deque
        depth_edges: dict[int, list[tuple[int, int]]] = {}
        for frag in self._fragments():
            idx = frag.indices
            if len(idx) < 2:
                continue
            root = idx[0]
            seen = {root}
            q = deque([(root, 0)])
            while q:
                a, d = q.popleft()
                for b in self.neighbors[a]:
                    if b not in seen:
                        seen.add(b)
                        depth_edges.setdefault(d, []).append((a, b))
                        q.append((b, d + 1))
        self._levels = [np.array(depth_edges[d], dtype=np.int64) for d in sorted(depth_edges)]
        return self._levels

    def _fix_pbc(self, pos: np.ndarray, box) -> np.ndarray:
        frags = self._fragments()
        if not frags:
            return pos
        box = np.asarray(box, dtype=np.float32)
        for edges in self._unwrap_levels():
            p, c = edges[:, 0], edges[:, 1]
            d = minimize_vectors((pos[c] - pos[p]).astype(np.float32), box)
            pos[c] = pos[p] + d
        # bring every solute fragment into the periodic image nearest the first (largest) fragment
        anchor = max(frags, key=lambda f: f.n_atoms)
        c0 = pos[anchor.indices].mean(axis=0)
        for f in frags:
            if f is anchor:
                continue
            c = pos[f.indices].mean(axis=0)
            d = minimize_vectors((c - c0)[None, :].astype(np.float32), box)[0]
            pos[f.indices] += (c0 + d) - c
        # wrap solvent/ions around the solute centre, per residue
        other = np.where(self.is_water | self.is_ion)[0]
        if len(other):
            centre = pos[~(self.is_water | self.is_ion)].mean(axis=0)
            _, head_pos, inverse = np.unique(self.u.atoms.resindices[other], return_index=True, return_inverse=True)
            heads = other[head_pos]
            d = minimize_vectors((pos[heads] - centre).astype(np.float32), box)
            shift = (centre + d) - pos[heads]
            pos[other] += shift[inverse].astype(np.float32)
        return pos

    def _align(self, pos: np.ndarray) -> np.ndarray:
        from MDAnalysis.analysis.align import rotation_matrix
        try:
            idx = self.u.select_atoms(self.expand_macros(self.align_selection)).indices
        except Exception:
            return pos
        if len(idx) < 3:
            return pos
        if self._ref is None:
            align, self.align = self.align, False
            ref = self._raw_nocache(0)[idx]
            self.align = align
            self._ref = (ref - ref.mean(axis=0)).astype(np.float64), ref.mean(axis=0)
        ref_c, ref_com = self._ref
        mob = pos[idx].astype(np.float64)
        mob_com = mob.mean(axis=0)
        R, _ = rotation_matrix(mob - mob_com, ref_c)
        return (((pos - mob_com) @ R.T) + ref_com).astype(np.float32)

    def _raw_nocache(self, frame: int) -> np.ndarray:
        ts = self.u.trajectory[frame]
        pos = ts.positions.astype(np.float32).copy()
        if self.pbc_fix and ts.dimensions is not None and np.all(ts.dimensions[:3] > 0):
            pos = self._fix_pbc(pos, ts.dimensions)
        return pos

    def positions(self, frame: int) -> np.ndarray:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        if frame in self._cache:
            self._cache.move_to_end(frame)
            return self._cache[frame]
        if self.smoothing > 1 and self.n_frames > 1:
            half = self.smoothing // 2
            lo, hi = max(0, frame - half), min(self.n_frames - 1, frame + half)
            pos = np.mean([self._raw(f) for f in range(lo, hi + 1)], axis=0).astype(np.float32)
        else:
            pos = self._raw(frame)
        self._cache[frame] = pos
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return pos

    def analysis_positions(self, frame: int) -> np.ndarray:
        """Physical frame, with PBC/fitting but never display smoothing.

        Treat the returned array as read-only. Visual interpolation must not
        affect measured distances, contact occupancy or fluctuations.
        """
        return self._raw(int(np.clip(frame, 0, self.n_frames - 1)))

    def box(self, frame: int):
        return self.u.trajectory[frame].dimensions

    # ------------------------------------------------------------------ analyses used for colouring
    def secondary_structure(self, frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(residue indices of protein residues, per-residue DSSP codes) at a frame."""
        if frame in self._ss_cache:
            return self._ss_cache[frame]
        prot = self.u.select_atoms("protein")
        res = prot.residues
        codes = np.array(["-"] * res.n_residues)
        try:
            codes_arr = self._dssp(prot, frame)
            if len(codes_arr) == res.n_residues:
                codes = codes_arr
            else:
                codes = self._ss_from_ca(res, frame)
        except Exception:
            codes = self._ss_from_ca(res, frame)
        out = (res.resindices, codes)
        self._ss_cache[frame] = out
        return out

    def _dssp(self, prot, frame):
        # Run DSSP on a one-frame copy of the processed coordinates (respects PBC fix / alignment)
        from MDAnalysis.analysis.dssp import DSSP
        sub = mda.Merge(prot)
        sub.atoms.positions = self.analysis_positions(frame)[prot.indices]
        d = DSSP(sub, guess_hydrogens=True,
                 heavyatom_names=("N", "CA", "C", "O O1 OT1 OC1")).run()
        return np.asarray(d.results.dssp[0])

    def _ss_from_ca(self, res, frame):
        """Fallback CA-geometry secondary structure (i,i+3 distance heuristics)."""
        pos = self.analysis_positions(frame)
        ca = []
        for r in res:
            a = r.atoms.select_atoms("name CA")
            ca.append(pos[a[0].index] if len(a) else np.full(3, np.nan))
        ca = np.array(ca)
        n = len(ca)
        codes = np.array(["-"] * n)
        for i in range(n - 3):
            d3 = np.linalg.norm(ca[i + 3] - ca[i])
            if 4.5 < d3 < 5.6:
                codes[i:i + 4] = "H"
        for i in range(n - 2):
            if codes[i] != "H" and np.linalg.norm(ca[i + 2] - ca[i]) > 6.2:
                codes[i:i + 3] = np.where(codes[i:i + 3] == "H", "H", "E")
        return codes

    def rmsf(self, max_frames: int = 200, progress=None) -> np.ndarray:
        """Per-atom RMSF (Angstrom) over the trajectory, protein-aligned."""
        key = (max_frames, self.pbc_fix, self.align_selection)
        if self._rmsf is not None and getattr(self, "_rmsf_key", None) == key:
            return self._rmsf
        if max_frames < 1:
            raise ValueError("RMSF requires at least one sampled frame.")
        idx = self.u.select_atoms(self.expand_macros(self.align_selection)).indices
        if len(idx) < 3:
            raise ValueError("RMSF requires an alignment selection containing at least three atoms.")
        frames = np.linspace(0, self.n_frames - 1, min(self.n_frames, max_frames), dtype=int)
        mean = np.zeros((self.n_atoms, 3))
        m2 = np.zeros_like(mean)
        previous_frame = self.u.trajectory.frame
        try:
            for k, f in enumerate(frames, 1):
                p = self._align(self._raw_nocache(int(f))).astype(np.float64)
                delta = p - mean
                mean += delta / k
                m2 += delta * (p - mean)
                if progress:
                    progress(k, len(frames))
        finally:
            self.u.trajectory[previous_frame]
        self._rmsf = np.sqrt(np.maximum(m2 / len(frames), 0).sum(axis=1)).astype(np.float32)
        self._rmsf_key = key
        return self._rmsf
