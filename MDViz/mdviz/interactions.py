"""Protein-ligand interaction detection (geometric criteria, PLIP-like defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from MDAnalysis.lib.distances import capped_distance

from . import chem

KINDS = ["H-bond", "Salt bridge", "Hydrophobic", "Pi-stacking", "Cation-pi", "Halogen bond"]


@dataclass
class InteractionParams:
    enabled: list = field(default_factory=lambda: list(KINDS))
    hbond_dist: float = 3.5        # donor-acceptor distance (A)
    hbond_angle: float = 120.0     # D-H...A angle (deg)
    hydrophobic_dist: float = 4.0
    salt_dist: float = 4.0
    pi_dist: float = 5.5           # parallel stacking centroid distance
    pi_t_dist: float = 6.0         # T-shaped centroid distance
    pi_offset: float = 2.0
    cation_pi_dist: float = 6.0
    halogen_dist: float = 3.5
    halogen_angle: float = 140.0

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


@dataclass
class Interaction:
    kind: str
    residue: str          # e.g. "ARG485:A"
    resindex: int
    lig_label: str
    rec_label: str
    p1: np.ndarray        # ligand-side point
    p2: np.ndarray        # receptor-side point
    distance: float
    anchor: int = -1      # receptor atom index used to place a residue label
    lig_atoms: tuple = ()
    rec_atoms: tuple = ()

    @property
    def key(self):
        return (self.kind, self.residue, self.lig_label, self.rec_label)


def _angle(a, b, c):
    """Angle at b (degrees) for points a-b-c."""
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def _ring_geom(pos, ring):
    pts = pos[list(ring)]
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[2], np.sqrt(np.mean(((pts - c) @ vt[2]) ** 2))


def _small_rings(atoms, neighbors, allowed, max_size=6):
    """All simple cycles of size 5..max_size among `atoms`."""
    aset = set(int(a) for a in atoms if allowed(a))
    rings = set()

    def dfs(start, cur, path):
        if len(path) > max_size:
            return
        for nb in neighbors[cur]:
            nb = int(nb)
            if nb not in aset:
                continue
            if nb == start and len(path) >= 5:
                rings.add(frozenset(path))
            elif nb not in path and nb > start:
                dfs(start, nb, path + [nb])

    for a in sorted(aset):
        dfs(a, a, [a])
    return [tuple(sorted(r)) for r in rings]


class InteractionDetector:
    def __init__(self, system, ligand_idx, receptor_idx, params: InteractionParams | None = None, ref_frame=0):
        self.ref_frame = ref_frame
        self.s = system
        self.p = params or InteractionParams()
        lig = np.asarray(sorted(set(map(int, ligand_idx))), dtype=np.int64)
        rec = np.asarray(sorted(set(map(int, receptor_idx)) - set(lig.tolist())), dtype=np.int64)
        # ignore solvent on the receptor side
        rec = rec[~(system.is_water[rec] | system.is_ion[rec])] if len(rec) else rec
        self.lig, self.rec = lig, rec
        el = system.elements
        nb = system.neighbors
        u = system.u
        names = u.atoms.names.astype(str)
        resnames = np.char.upper(u.atoms.resnames.astype(str))
        self.names = names
        multi_chain = len(set(system.chains[rec])) > 1 if len(rec) else False
        self._res_label = {}
        for r in u.atoms[rec].residues if len(rec) else []:
            ch = system.chains[r.atoms.indices[0]]
            self._res_label[r.resindex] = f"{r.resname}{r.resid}" + (f":{ch}" if multi_chain else "")
        self.resindex = u.atoms.resindices
        heavy = lambda i: el[i] != "H"
        hyd = lambda i: [j for j in nb[i] if el[j] == "H"]

        def donors(idx):
            return {int(i): hyd(i) for i in idx if el[i] in ("N", "O", "S") and hyd(i)}

        def acceptors(idx, protein):
            out = []
            for i in idx:
                e = el[i]
                if e == "O":
                    out.append(int(i))
                elif e == "N" and not hyd(i):
                    heavy_nb = [j for j in nb[i] if el[j] != "H"]
                    if protein:
                        if resnames[i] in ("HIS", "HID", "HIE", "HSD", "HSE") and names[i] in ("ND1", "NE2"):
                            out.append(int(i))
                    elif len(heavy_nb) <= 2 or not nb[i]:
                        out.append(int(i))
            return np.array(out, dtype=np.int64)

        self.lig_don = donors(lig)
        self.rec_don = donors(rec)
        self.lig_acc = acceptors(lig, False)
        self.rec_acc = acceptors(rec, True)
        has_h = (el[lig] == "H").any()
        self.no_h = not has_h

        def hydrophobic(idx, protein):
            out = []
            for i in idx:
                if el[i] not in ("C", "S", "CL", "BR", "I"):
                    continue
                if protein and names[i] in ("C", "CA", "N", "O"):
                    continue
                if all(el[j] in ("C", "H", "S", "CL", "BR", "I", "F") for j in nb[i]):
                    out.append(int(i))
            return np.array(out, dtype=np.int64)

        self.lig_hyd = hydrophobic(lig, False)
        self.rec_hyd = hydrophobic(rec, True)

        # charged groups: list of (atom indices tuple, sign)
        self.rec_charged = []
        for r in u.atoms[rec].residues if len(rec) else []:
            rn = r.resname.upper()
            for table, sign in ((chem.POSITIVE_GROUPS, 1), (chem.NEGATIVE_GROUPS, -1)):
                if rn in table:
                    at = [a.index for a in r.atoms if a.name in table[rn]]
                    if at:
                        self.rec_charged.append((tuple(at), sign))
            term = [a.index for a in r.atoms if a.name in ("OC1", "OC2", "OXT", "OT1", "OT2")]
            if term:
                self.rec_charged.append((tuple(term), -1))
        self.lig_charged = []
        for i in lig:
            if el[i] == "N" and len(nb[i]) == 4:
                self.lig_charged.append(((int(i),), 1))
            if el[i] in ("C", "P", "S"):
                term_o = [j for j in nb[i] if el[j] == "O" and len(nb[j]) == 1]
                need = {"C": 2, "P": 3, "S": 3}[el[i]]
                if len(term_o) >= need:
                    self.lig_charged.append((tuple(int(j) for j in term_o), -1))
        self.lig_cations = [g[0] for g in self.lig_charged if g[1] > 0]

        # aromatic rings
        ref = system.analysis_positions(ref_frame)
        rings = _small_rings(lig, nb, lambda a: el[a] in ("C", "N", "O", "S") and len(nb[a]) <= 3)
        self.lig_rings = [r for r in rings if _ring_geom(ref, r)[2] < 0.25]
        self.rec_rings = []
        for r in u.atoms[rec].residues if len(rec) else []:
            for ring_names in chem.AROMATIC_RINGS.get(r.resname.upper(), []):
                at = [a.index for a in r.atoms if a.name in ring_names]
                if len(at) == len(ring_names):
                    self.rec_rings.append((tuple(at), r.resindex))
        self.rec_cations = []
        for r in u.atoms[rec].residues if len(rec) else []:
            for nm in chem.CATION_ATOMS.get(r.resname.upper(), ()):
                at = [a.index for a in r.atoms if a.name == nm]
                if at:
                    self.rec_cations.append(at[0])
        self.halogens = [(int(i), int(nb[i][0])) for i in lig if el[i] in ("CL", "BR", "I") and len(nb[i]) == 1]

    # ------------------------------------------------------------------
    def label(self, resindex):
        return self._res_label.get(resindex, str(resindex))

    def _mk(self, kind, lig_atoms, rec_atoms, p1, p2, anchor):
        ri = int(self.resindex[anchor])
        return Interaction(kind, self.label(ri), ri,
                           "/".join(self.names[a] for a in lig_atoms) if len(lig_atoms) <= 2 else "ring:" + self.names[lig_atoms[0]],
                           "/".join(self.names[a] for a in rec_atoms) if len(rec_atoms) <= 2 else "ring",
                           np.asarray(p1, dtype=np.float32), np.asarray(p2, dtype=np.float32),
                           float(np.linalg.norm(np.asarray(p1) - np.asarray(p2))), int(anchor),
                           tuple(map(int, lig_atoms)), tuple(map(int, rec_atoms)))

    def detect(self, pos: np.ndarray) -> list[Interaction]:
        p = self.p
        out: list[Interaction] = []
        if not len(self.lig) or not len(self.rec):
            return out
        en = set(p.enabled)
        reach = max(p.pi_t_dist, p.cation_pi_dist, p.hydrophobic_dist, p.hbond_dist, p.salt_dist) + 4.0
        pairs = capped_distance(pos[self.lig], pos[self.rec], reach, return_distances=False)
        near = np.zeros(len(pos), bool)
        near[self.rec[np.unique(pairs[:, 1])]] = True
        if not near.any():
            return out

        # ---- hydrogen bonds
        if "H-bond" in en:
            def hb(donors: dict, acc: np.ndarray, lig_is_donor: bool):
                if not donors or not len(acc):
                    return
                d_idx = np.array(list(donors.keys()))
                if not lig_is_donor:
                    d_idx = d_idx[near[d_idx]]
                    a_idx = acc
                else:
                    a_idx = acc[near[acc]]
                if not len(d_idx) or not len(a_idx):
                    return
                pr, dist = capped_distance(pos[d_idx], pos[a_idx], p.hbond_dist)
                for (i, j), dd in zip(pr, dist):
                    D, A = int(d_idx[i]), int(a_idx[j])
                    if D == A:
                        continue
                    hs = donors[D]
                    ang = max((_angle(pos[D], pos[h], pos[A]) for h in hs), default=180.0)
                    if ang < p.hbond_angle:
                        continue
                    lig_atom, rec_atom = (D, A) if lig_is_donor else (A, D)
                    k = self._mk("H-bond", [lig_atom], [rec_atom], pos[lig_atom], pos[rec_atom], rec_atom)
                    k.rec_label += " (acc)" if lig_is_donor else " (don)"
                    out.append(k)
            hb(self.lig_don, self.rec_acc, True)
            hb(self.rec_don, self.lig_acc, False)

        # ---- salt bridges
        if "Salt bridge" in en:
            for lg, ls in self.lig_charged:
                for rg, rs in self.rec_charged:
                    if ls * rs >= 0 or not near[list(rg)].any():
                        continue
                    d = np.linalg.norm(pos[list(lg)][:, None] - pos[list(rg)][None], axis=2)
                    i, j = np.unravel_index(np.argmin(d), d.shape)
                    if d[i, j] <= p.salt_dist:
                        out.append(self._mk("Salt bridge", [lg[i]], [rg[j]], pos[lg[i]], pos[rg[j]], rg[j]))

        # ---- hydrophobic contacts (closest pair per residue)
        if "Hydrophobic" in en and len(self.lig_hyd) and len(self.rec_hyd):
            rh = self.rec_hyd[near[self.rec_hyd]]
            if len(rh):
                pr, dist = capped_distance(pos[self.lig_hyd], pos[rh], p.hydrophobic_dist)
                best = {}
                for (i, j), dd in zip(pr, dist):
                    ri = self.resindex[rh[j]]
                    if ri not in best or dd < best[ri][0]:
                        best[ri] = (dd, int(self.lig_hyd[i]), int(rh[j]))
                for ri, (dd, a, b) in best.items():
                    out.append(self._mk("Hydrophobic", [a], [b], pos[a], pos[b], b))

        # ---- aromatic interactions
        lig_geo = [(_ring_geom(pos, r), r) for r in self.lig_rings]
        if "Pi-stacking" in en:
            for (lc, ln, _), lr in lig_geo:
                for rr, ri in self.rec_rings:
                    if not near[list(rr)].any():
                        continue
                    rc, rn, _ = _ring_geom(pos, rr)
                    v = rc - lc
                    d = np.linalg.norm(v)
                    if d > p.pi_t_dist:
                        continue
                    ang = np.degrees(np.arccos(np.clip(abs(np.dot(ln, rn)), 0, 1)))
                    off = min(np.sqrt(max(d * d - np.dot(v, ln) ** 2, 0)), np.sqrt(max(d * d - np.dot(v, rn) ** 2, 0)))
                    ok = (ang < 30 and d <= p.pi_dist and off <= p.pi_offset) or \
                         (ang > 60 and d <= p.pi_t_dist and off <= p.pi_offset)
                    if ok:
                        k = self._mk("Pi-stacking", list(lr), list(rr), lc, rc, rr[0])
                        k.rec_label = "parallel" if ang < 30 else "T-shaped"
                        out.append(k)
        if "Cation-pi" in en:
            for (lc, ln, _), lr in lig_geo:
                for c in self.rec_cations:
                    if not near[c]:
                        continue
                    v = pos[c] - lc
                    d = np.linalg.norm(v)
                    off = np.sqrt(max(d * d - np.dot(v, ln) ** 2, 0))
                    if d <= p.cation_pi_dist and off <= 2.5:
                        out.append(self._mk("Cation-pi", list(lr), [c], lc, pos[c], c))
            for c in self.lig_cations:
                for rr, ri in self.rec_rings:
                    if not near[list(rr)].any():
                        continue
                    rc, rn, _ = _ring_geom(pos, rr)
                    v = pos[c] - rc
                    d = np.linalg.norm(v)
                    off = np.sqrt(max(d * d - np.dot(v, rn) ** 2, 0))
                    if d <= p.cation_pi_dist and off <= 2.5:
                        out.append(self._mk("Cation-pi", [c], list(rr), pos[c], rc, rr[0]))

        # ---- halogen bonds
        if "Halogen bond" in en and self.halogens:
            acc = np.concatenate([self.rec_acc, np.array([i for i in self.rec if self.s.elements[i] == "S"], dtype=np.int64)])
            acc = acc[near[acc]] if len(acc) else acc
            for X, C in self.halogens:
                if not len(acc):
                    break
                d = np.linalg.norm(pos[acc] - pos[X], axis=1)
                for j in np.where(d <= p.halogen_dist)[0]:
                    A = int(acc[j])
                    if _angle(pos[C], pos[X], pos[A]) >= p.halogen_angle:
                        out.append(self._mk("Halogen bond", [X], [A], pos[X], pos[A], A))
        return out


@dataclass
class OccupancyResult:
    frames: list
    times_ps: list
    residue_keys: list          # (kind, residue)
    residue_matrix: np.ndarray  # (n_keys, n_frames) bool
    detail_keys: list           # full keys
    detail_counts: dict
    metadata: dict = field(default_factory=dict)

    def save_metadata(self, path):
        import json
        with open(str(path) + ".metadata.json", "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2)

    def residue_occupancy(self):
        occ = self.residue_matrix.mean(axis=1) * 100 if self.residue_matrix.size else np.zeros(0)
        return sorted(zip(self.residue_keys, occ), key=lambda kv: -kv[1])

    def to_csv(self, path):
        self.save_metadata(path)
        import csv
        n = len(self.frames)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["level", "type", "residue", "ligand_atoms", "receptor_atoms", "frames_present", "total_frames", "occupancy_percent"])
            for (kind, res), occ in self.residue_occupancy():
                cnt = int(round(occ * n / 100))
                w.writerow(["residue", kind, res, "", "", cnt, n, f"{occ:.1f}"])
            for key, cnt in sorted(self.detail_counts.items(), key=lambda kv: -kv[1]):
                kind, res, la, ra = key
                w.writerow(["atom", kind, res, la, ra, cnt, n, f"{100 * cnt / max(n, 1):.1f}"])

    def timeline_csv(self, path):
        self.save_metadata(path)
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_ns"] + [f"{k}|{r}" for k, r in self.residue_keys])
            for j, (f, t) in enumerate(zip(self.frames, self.times_ps)):
                w.writerow([f, f"{t / 1000:.4f}"] + [int(v) for v in self.residue_matrix[:, j]])


def occupancy(system, detector: InteractionDetector, frames, progress=None) -> OccupancyResult:
    frames = list(frames)
    if not frames:
        raise ValueError("Choose a non-empty frame range (start must not exceed end).")
    requested_count = len(frames)
    per_frame = []
    detail = {}
    times = []
    for k, f in enumerate(frames):
        inter = detector.detect(system.analysis_positions(f))
        per_frame.append({(i.kind, i.residue) for i in inter})
        for key in {i.key for i in inter}:
            detail[key] = detail.get(key, 0) + 1
        times.append(system.frame_time(f))
        if progress and not progress(k + 1, len(frames)):
            break
    frames = list(frames)[:len(per_frame)]
    keys = sorted(set().union(*per_frame) if per_frame else set(), key=lambda kr: (KINDS.index(kr[0]), kr[1]))
    mat = np.zeros((len(keys), len(per_frame)), bool)
    kidx = {k: i for i, k in enumerate(keys)}
    for j, s in enumerate(per_frame):
        for key in s:
            mat[kidx[key], j] = True
    metadata = {"topology": system.topology, "trajectories": system.trajectories,
                "frames": [int(f) for f in frames], "smoothing": 1,
                "pbc_fix": system.pbc_fix, "align": system.align,
                "align_selection": system.align_selection, "interaction_parameters": detector.p.to_dict(),
                "ligand_indices": detector.lig.tolist(), "receptor_indices": detector.rec.tolist(),
                "partial": len(per_frame) < requested_count}
    return OccupancyResult(frames, times, keys, mat, list(detail.keys()), detail, metadata)
