"""Set up a ligand torsion PES from an MD run.

A 2D PES needs two coordinates. For a protein-ligand run the useful ones are the ligand's rotatable
torsions: this module finds the ligand's GROMACS topology (e.g. the acpype GAFF files in the run folder),
lists its rotatable bonds, measures every torsion over the trajectory (periodic-box safe), and ranks
them by how much they move in the binding pocket. The two top-ranked torsions are then scanned for the
isolated ligand, and the MD frames are overlaid on the surface.

    setup = setup_from_md(system)          # system: mdviz.system.MolSystem of the complex trajectory
    cfg = PESConfig(molecule="custom", custom_gro=setup.gro, custom_top=setup.top,
                    phi_atoms=setup.torsions[0].quad, psi_atoms=setup.torsions[1].quad)
    result = PESScanner(cfg).prepare().run()
    attach_md_samples(result, setup, system)
"""
from __future__ import annotations

import glob
import os
import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from MDAnalysis.lib.distances import calc_dihedrals

from .pes import read_gro, top_bonds, flatten_topology


@dataclass
class Torsion:
    quad: tuple                  # 0-based atom indices in the ligand topology
    label: str                   # e.g. "O-S-N-C"
    spread: float = 0.0          # circular standard deviation over the MD run (degrees)
    mean: float = 0.0            # circular mean (degrees)
    angles: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class LigandScanSetup:
    gro: str
    top: str
    names: list
    torsions: list               # ranked, most mobile first
    frames: np.ndarray
    times_ps: np.ndarray
    resname: str


def read_molecule(top_path):
    """(atom names, 0-based bonds) of the first moleculetype, following local #includes."""
    import tempfile
    text = flatten_topology(top_path)
    names, section = [], None
    for ln in text.splitlines():
        s = ln.split(";")[0].strip()
        if not s:
            continue
        if s.startswith("["):
            section = s.strip("[] ").lower()
            if section in ("system", "molecules") or (section == "moleculetype" and names):
                break
            continue
        if section == "atoms":
            parts = s.split()
            if len(parts) >= 5 and parts[0].isdigit():
                names.append(parts[4])
    with tempfile.NamedTemporaryFile("w", suffix=".top", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        bonds = top_bonds(tmp)
    finally:
        os.remove(tmp)
    return names, bonds


def find_ligand_topology(run_dir: str, lig_names: list) -> tuple[str, str]:
    """Locate a standalone ligand .gro/.top whose atom names match the ligand in the MD system.

    Prefers a topology whose .itp is the one included by the run's complex topology.
    """
    cands = []
    for top in glob.glob(os.path.join(run_dir, "**", "*.top"), recursive=True):
        if "OPLS" in os.path.basename(top).upper():
            continue
        try:
            names, bonds = read_molecule(top)
        except Exception:
            continue
        if names != list(lig_names) or not bonds:
            continue
        gro = os.path.splitext(top)[0] + ".gro"
        if not os.path.isfile(gro):
            gros = glob.glob(os.path.join(os.path.dirname(top), "*.gro"))
            gro = next((g for g in gros if _gro_names(g) == list(lig_names)), "")
        if gro:
            cands.append((top, gro))
    if not cands:
        raise FileNotFoundError(
            f"No ligand topology with atoms {' '.join(lig_names[:6])}… found under {run_dir}. "
            "Choose the ligand .gro and .top manually (custom molecule).")
    used = set()
    for top in glob.glob(os.path.join(run_dir, "*.top")):
        with open(top, errors="ignore") as fh:
            used |= set(re.findall(r'#include\s+"([^"]+\.itp)"', fh.read()))
    used = {os.path.basename(u) for u in used}

    def score(c):
        itps = re.findall(r"inlined from (\S+\.itp)", flatten_topology(c[0]))
        return (any(os.path.basename(i) in used for i in itps), "DOCKED" in c[0].upper())
    top, gro = max(cands, key=score)
    return gro, top


def _gro_names(path):
    try:
        return [r[2] for r in read_gro(path)[1]]
    except Exception:
        return []


def rotatable_torsions(names, bonds) -> list[Torsion]:
    """Torsions about non-ring single bonds between heavy atoms that both carry other heavy atoms."""
    n = len(names)
    el = [re.sub(r"[^A-Za-z]", "", x)[:1].upper() for x in names]
    nb = [[] for _ in range(n)]
    for a, b in bonds:
        nb[a].append(b)
        nb[b].append(a)

    def in_ring(a, b):
        seen, q = {a}, deque([a])
        while q:
            x = q.popleft()
            for y in nb[x]:
                if {x, y} == {a, b}:
                    continue
                if y == b:
                    return True
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        return False
    out = []
    for a, b in bonds:
        if el[a] == "H" or el[b] == "H" or in_ring(a, b):
            continue
        ha = sorted((x for x in nb[a] if x != b and el[x] != "H"), key=lambda x: (-len(nb[x]), x))
        hb = sorted((x for x in nb[b] if x != a and el[x] != "H"), key=lambda x: (-len(nb[x]), x))
        if not ha or not hb:
            continue
        q = (ha[0], a, b, hb[0])
        out.append(Torsion(q, "-".join(names[i] for i in q)))
    return out


def _circ_stats(deg):
    r = np.radians(deg)
    C, S = np.cos(r).mean(), np.sin(r).mean()
    R = max(min(np.hypot(C, S), 1.0), 1e-12)
    return float(np.degrees(np.sqrt(-2 * np.log(R)))), float(np.degrees(np.arctan2(S, C)))


def md_torsions(system, lig_idx, quads, max_frames=2000):
    """Torsion angles (degrees) of the ligand over the trajectory; minimum-image safe on raw frames."""
    u = system.u
    n = len(u.trajectory)
    stride = max(1, int(np.ceil(n / max_frames)))
    frames = np.arange(0, n, stride)
    idx = np.array([[lig_idx[i] for i in q] for q in quads])  # (k, 4) system atom indices
    out = np.zeros((len(frames), len(quads)))
    times = np.zeros(len(frames))
    for k, f in enumerate(frames):
        ts = u.trajectory[f]
        p = ts.positions
        box = ts.dimensions if ts.dimensions is not None and np.all(ts.dimensions[:3] > 0) else None
        out[k] = np.degrees(calc_dihedrals(p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]], p[idx[:, 3]], box=box))
        times[k] = ts.time
    return frames, times, out


def setup_from_md(system, selection="ligand", max_frames=2000, run_dir=None) -> LigandScanSetup:
    """Find the ligand topology, its rotatable torsions, and rank them by mobility over the MD run."""
    lig = system.select(selection)
    if not len(lig):
        raise ValueError(f"Selection '{selection}' matched no atoms in the MD system.")
    resnames = sorted(set(system.u.atoms[lig].resnames))
    lig_names = list(system.u.atoms[lig].names)
    run_dir = run_dir or os.path.dirname(system.topology)
    gro, top = find_ligand_topology(run_dir, lig_names)
    names, bonds = read_molecule(top)
    tors = rotatable_torsions(names, bonds)
    if len(tors) < 2:
        raise ValueError("The ligand has fewer than two rotatable torsions; a 2D scan is not meaningful.")
    frames, times, ang = md_torsions(system, lig, [t.quad for t in tors], max_frames)
    for k, t in enumerate(tors):
        t.angles = ang[:, k]
        t.spread, t.mean = _circ_stats(ang[:, k])
    tors.sort(key=lambda t: -t.spread)
    return LigandScanSetup(gro, top, names, tors, frames, times, "/".join(resnames))


def attach_md_samples(result, setup: LigandScanSetup, system, selection="ligand"):
    """Store the MD torsion pairs for the scanned torsions on the result (for overlay and strain)."""
    cached = {tuple(t.quad): t.angles for t in setup.torsions if len(t.angles)}
    q1, q2 = tuple(result.phi_atoms), tuple(result.psi_atoms)
    if q1 in cached and q2 in cached:  # already measured when the torsions were ranked
        ang = np.column_stack([cached[q1], cached[q2]])
    else:
        _, _, ang = md_torsions(system, system.select(selection), [q1, q2], len(setup.frames))
    result.md_samples = ang
    names = setup.names
    result.meta["x_label"] = "τ1 (" + "-".join(names[i] for i in result.phi_atoms) + ")"
    result.meta["y_label"] = "τ2 (" + "-".join(names[i] for i in result.psi_atoms) + ")"
    result.meta["md_source"] = os.path.basename(system.trajectories[0]) if system.trajectories else ""
    result.meta["ligand"] = setup.resname
    return result
