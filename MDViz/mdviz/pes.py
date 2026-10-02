"""Potential energy surface (PES) scans over backbone dihedrals (phi, psi).

Pipeline (no Qt here, so it can run from the GUI, a thread or the command line):

    cfg = PESConfig(step=10)
    scanner = PESScanner(cfg)
    scanner.prepare(log=print)                 # build Ace-Ala-Nme, pdb2gmx, relax reference
    result = scanner.run(progress=cb)          # restrained EM at every (phi, psi) grid point
    result.save("ala2_pes.npz")
    points = find_stationary_points(result)    # basins and saddle points

Each grid point: the reference structure is rotated rigidly about N-CA (phi) and CA-C (psi) to the
target angles, both dihedrals are held by strong GROMACS [ dihedral_restraints ], every other degree of
freedom is minimised, and E(phi, psi) = potential energy - restraint energy (kJ/mol).
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

FORCE_FIELDS = ["amber99sb-ildn", "amber99sb", "amber99", "amber03", "amber94", "amber96"]

# Vacuum minimisation: large periodic box with cut-offs longer than the molecule, so every intramolecular
# pair is inside the cut-off and no periodic image is. The plain cut-off only adds a conformation-independent
# constant to the energy, which drops out of the relative surface.
EM_MDP = """; MDViz PES scan: restrained vacuum energy minimisation
integrator      = {integrator}
emtol           = {emtol}
emstep          = 0.01
nsteps          = {nsteps}
nstenergy       = {nsteps}
nstxout         = 0
cutoff-scheme   = Verlet
pbc             = xyz
nstlist         = 10
coulombtype     = Cut-off
rcoulomb        = {cutoff}
epsilon-r       = {epsilon_r}
vdwtype         = Cut-off
rvdw            = {cutoff}
DispCorr        = no
constraints     = none
"""


# ---------------------------------------------------------------------------------------- geometry
def dihedral(p0, p1, p2, p3) -> float:
    """IUPAC dihedral angle in degrees (same convention as GROMACS)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _place(a, b, c, bond, angle, torsion):
    """NeRF: position of atom d bonded to c, with angle b-c-d and torsion a-b-c-d (degrees)."""
    angle, torsion = np.radians(angle), np.radians(torsion)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(angle), bond * np.sin(angle) * np.cos(torsion), bond * np.sin(angle) * np.sin(torsion)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


# heavy atoms of Ace-Ala-Nme in PDB order: (resname, resid, atom name)
DIPEPTIDE_ATOMS = [("ACE", 1, "CH3"), ("ACE", 1, "C"), ("ACE", 1, "O"),
                   ("ALA", 2, "N"), ("ALA", 2, "CA"), ("ALA", 2, "C"), ("ALA", 2, "O"), ("ALA", 2, "CB"),
                   ("NME", 3, "N"), ("NME", 3, "CH3")]


def build_dipeptide(phi=-60.0, psi=135.0, omega=180.0) -> np.ndarray:
    """Heavy-atom coordinates (Angstrom) of L-Ace-Ala-Nme from standard peptide geometry."""
    ch3a = np.array([0.0, 0.0, 0.0])
    c1 = np.array([1.52, 0.0, 0.0])
    pseudo = np.array([-0.5, 1.4, 0.0])
    n2 = _place(pseudo, ch3a, c1, 1.33, 116.6, 180.0)
    ca = _place(ch3a, c1, n2, 1.46, 121.7, omega)
    c2 = _place(c1, n2, ca, 1.52, 111.2, phi)
    cb = _place(n2, c2, ca, 1.53, 109.5, 122.686)      # L-chirality (PeptideBuilder convention)
    n3 = _place(n2, ca, c2, 1.33, 116.2, psi)
    o2 = _place(n3, ca, c2, 1.23, 120.5, 180.0)
    ch3n = _place(ca, c2, n3, 1.46, 121.7, omega)
    o1 = _place(ca, n2, c1, 1.23, 122.7, 0.0)
    return np.array([ch3a, c1, o1, n2, ca, c2, o2, cb, n3, ch3n])


def write_pdb(path, atoms, coords):
    with open(path, "w") as fh:
        for i, ((rn, ri, name), p) in enumerate(zip(atoms, coords), start=1):
            nm = f" {name:<3s}" if len(name) < 4 else name
            fh.write(f"ATOM  {i:5d} {nm} {rn:3s} A{ri:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00"
                     f"          {name[0]:>2s}\n")
        fh.write("TER\nEND\n")


def read_gro(path):
    """Return (title, atom records [(resid, resname, name)], coords in Angstrom, box line)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[1].strip())
    recs, xyz = [], []
    for ln in lines[2:2 + n]:
        recs.append((int(ln[0:5]), ln[5:10].strip(), ln[10:15].strip()))
        xyz.append([float(ln[20:28]), float(ln[28:36]), float(ln[36:44])])
    return lines[0], recs, np.array(xyz) * 10.0, lines[2 + n]


def write_gro(path, recs, coords_A, box_nm=5.0, title="MDViz PES"):
    with open(path, "w") as fh:
        fh.write(f"{title}\n{len(recs)}\n")
        for i, ((ri, rn, name), p) in enumerate(zip(recs, np.asarray(coords_A) / 10.0), start=1):
            fh.write(f"{ri:5d}{rn:<5s}{name:>5s}{i % 100000:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
        fh.write(f"{box_nm:10.5f}{box_nm:10.5f}{box_nm:10.5f}\n")


def top_bonds(top_path) -> list[tuple[int, int]]:
    """0-based bonds of the first inline moleculetype in a GROMACS topology."""
    bonds, section = [], None
    with open(top_path) as fh:
        for ln in fh:
            s = ln.split(";")[0].strip()
            if not s:
                continue
            if s.startswith("["):
                section = s.strip("[] ").lower()
                if section == "system":
                    break
                continue
            if section == "bonds":
                parts = s.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
    return bonds


def flatten_topology(top_path: str, _depth=0) -> str:
    """Topology text with local #include files inlined (force-field includes are left alone).

    acpype and pdb2gmx often keep the molecule in an .itp; the scanner needs its [ bonds ] and a place to
    add [ dihedral_restraints ] inside the [ moleculetype ].
    """
    base = os.path.dirname(os.path.abspath(top_path))
    with open(top_path) as fh:
        text = fh.read()

    def inline(m):
        cand = os.path.join(base, m.group(1))
        if _depth < 5 and os.path.isfile(cand):
            return f"; ---- inlined from {m.group(1)}\n" + flatten_topology(cand, _depth + 1)
        return m.group(0)
    return re.sub(r'#include\s+"([^"]+)"', inline, text)


def moving_side(n_atoms, bonds, b, c) -> np.ndarray:
    """Atoms on the c-side of bond b-c (rigidly rotated when the dihedral about b-c changes)."""
    nb = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        nb[i].append(j)
        nb[j].append(i)
    seen = {c}
    q = deque([c])
    while q:
        a = q.popleft()
        for x in nb[a]:
            if x == b and a == c:
                continue
            if x not in seen:
                if x == b:
                    raise ValueError("Dihedral bond is part of a ring; it cannot be rotated rigidly.")
                seen.add(x)
                q.append(x)
    mask = np.zeros(n_atoms, bool)
    mask[list(seen)] = True
    return mask


def rotate_dihedral(coords, quad, target, mask) -> np.ndarray:
    """Rotate atoms in `mask` about the quad[1]->quad[2] axis so that dihedral(quad) == target."""
    x = coords.copy()
    cur = dihedral(*x[list(quad)])
    delta = np.radians(wrap180(target - cur))
    b, c = x[quad[1]], x[quad[2]]
    k = (c - b) / np.linalg.norm(c - b)
    v = x[mask] - c
    cos, sin = np.cos(delta), np.sin(delta)
    x[mask] = c + v * cos + np.cross(k, v) * sin + np.outer(v @ k, k) * (1 - cos)  # Rodrigues
    return x


def find_backbone_dihedrals(recs):
    """(phi quad, psi quad) 0-based from atom names: C(i-1)-N-CA-C and N-CA-C-N(i+1) of the central residue."""
    by_res = {}
    for i, (ri, rn, name) in enumerate(recs):
        by_res.setdefault(ri, {})[name] = i
    resids = sorted(by_res)
    for k in range(1, len(resids) - 1):
        prev, cur, nxt = by_res[resids[k - 1]], by_res[resids[k]], by_res[resids[k + 1]]
        if all(n in cur for n in ("N", "CA", "C")) and "C" in prev and "N" in nxt:
            return ((prev["C"], cur["N"], cur["CA"], cur["C"]), (cur["N"], cur["CA"], cur["C"], nxt["N"]))
    raise ValueError("Could not find C-N-CA-C / N-CA-C-N atoms; set the dihedral atom indices manually.")


# ---------------------------------------------------------------------------------------- engine
class GromacsEngine:
    """Runs gmx natively or inside WSL (Windows paths are translated to /mnt/<drive>/...)."""

    def __init__(self, gmx="gmx", use_wsl=None, distro=""):
        self.gmx = gmx
        if use_wsl is None:
            use_wsl = os.name == "nt" and shutil.which(gmx) is None
        self.use_wsl = bool(use_wsl)
        self.distro = distro

    @staticmethod
    def to_posix(path: str) -> str:
        p = os.path.abspath(path)
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
        if not m:
            return p.replace("\\", "/")
        return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")

    def _prefix(self):
        if not self.use_wsl:
            return []
        return ["wsl.exe"] + (["-d", self.distro] if self.distro else [])

    def bash(self, script: str, cwd: str, **popen_kw):
        """Start a bash script in `cwd` (returns Popen)."""
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if self.use_wsl:
            cmd = self._prefix() + ["--cd", cwd, "-e", "bash", "-c", script]
        else:
            cmd = ["bash", "-c", script]
        return subprocess.Popen(cmd, cwd=cwd if not self.use_wsl else None, creationflags=flags, **popen_kw)

    def run(self, args: list[str], cwd: str, stdin: str | None = None, timeout=600) -> subprocess.CompletedProcess:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if self.use_wsl:
            cmd = self._prefix() + ["--cd", cwd, "-e", self.gmx] + args
        else:
            cmd = [self.gmx] + args
        r = subprocess.run(cmd, cwd=None if self.use_wsl else cwd, input=stdin, capture_output=True, text=True,
                           timeout=timeout, creationflags=flags)
        if r.returncode != 0:
            tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-25:])
            raise RuntimeError(f"gmx {args[0]} failed:\n{tail}")
        return r

    def check(self) -> str:
        r = self.run(["--version"], cwd=tempfile.gettempdir(), timeout=120)
        m = re.search(r"GROMACS version:\s*(\S+)", r.stdout)
        return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------------------- config / result
@dataclass
class PESConfig:
    molecule: str = "ala2"            # "ala2" (build Ace-Ala-Nme) or "custom" (.gro + .top below)
    forcefield: str = "amber99sb-ildn"
    custom_gro: str = ""
    custom_top: str = ""
    phi_atoms: list = field(default_factory=list)   # 0-based quads; auto-detected when empty
    psi_atoms: list = field(default_factory=list)
    step: float = 10.0                # grid spacing (degrees)
    restraint_k: float = 5000.0       # kJ/mol/rad^2
    integrator: str = "l-bfgs"        # l-bfgs | steep
    emtol: float = 1.0                # kJ/mol/nm
    nsteps: int = 5000
    cutoff: float = 1.9               # nm; box is 2*cutoff + 1.2 nm
    epsilon_r: float = 1.0            # dielectric: 1 vacuum, ~4 protein interior, ~80 water-like screening
    workers: int = 0                  # 0 = auto
    gmx: str = "gmx"
    use_wsl: bool | None = None
    wsl_distro: str = ""
    workdir: str = ""                 # empty = temp folder

    def grid(self):
        n = int(round(360.0 / self.step))
        return np.array([-180.0 + i * self.step for i in range(n)])


@dataclass
class PESResult:
    phi: np.ndarray                   # (n,) grid values (degrees)
    psi: np.ndarray                   # (m,)
    energy: np.ndarray                # (n, m) kJ/mol, NaN where missing
    phi_actual: np.ndarray            # (n, m) achieved dihedrals after minimisation
    psi_actual: np.ndarray
    coords: np.ndarray | None         # (n, m, atoms, 3) Angstrom, or None
    atoms: list                       # [(resid, resname, name)]
    phi_atoms: tuple
    psi_atoms: tuple
    meta: dict
    bonds: list = field(default_factory=list)   # 0-based atom pairs (for the molecule view)
    md_samples: np.ndarray | None = None        # (N, 2) torsion pairs sampled in an MD run (degrees)

    @property
    def labels(self):
        """Axis names: phi/psi for peptides, torsion names for ligands."""
        return self.meta.get("x_label", "φ"), self.meta.get("y_label", "ψ")

    @property
    def rel(self) -> np.ndarray:
        """Energies relative to the global minimum."""
        return self.energy - np.nanmin(self.energy) if np.isfinite(self.energy).any() else self.energy

    def index(self, phi, psi):
        i = int(np.argmin(np.abs(wrap180(self.phi - phi))))
        j = int(np.argmin(np.abs(wrap180(self.psi - psi))))
        return i, j

    def save(self, path):
        np.savez_compressed(path, phi=self.phi, psi=self.psi, energy=self.energy, phi_actual=self.phi_actual,
                            psi_actual=self.psi_actual, coords=self.coords if self.coords is not None else np.zeros(0),
                            atoms=np.array([f"{a}|{b}|{c}" for a, b, c in self.atoms]),
                            phi_atoms=np.array(self.phi_atoms), psi_atoms=np.array(self.psi_atoms),
                            meta=np.array(repr(self.meta)), bonds=np.array(self.bonds, dtype=np.int64).reshape(-1, 2),
                            md_samples=self.md_samples if self.md_samples is not None else np.zeros((0, 2)))

    def to_csv(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phi_deg", "psi_deg", "E_kJmol", "dE_kJmol", "phi_actual", "psi_actual"])
            rel = self.rel
            for i, p in enumerate(self.phi):
                for j, q in enumerate(self.psi):
                    w.writerow([p, q, f"{self.energy[i, j]:.4f}", f"{rel[i, j]:.4f}",
                                f"{self.phi_actual[i, j]:.2f}", f"{self.psi_actual[i, j]:.2f}"])

    @classmethod
    def load(cls, path) -> "PESResult":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            import ast
            d = np.load(path, allow_pickle=False)
            coords = d["coords"] if d["coords"].size else None
            atoms = [(int(s.split("|")[0]), s.split("|")[1], s.split("|")[2]) for s in d["atoms"].tolist()]
            return cls(d["phi"], d["psi"], d["energy"], d["phi_actual"], d["psi_actual"], coords, atoms,
                       tuple(int(x) for x in d["phi_atoms"]), tuple(int(x) for x in d["psi_atoms"]),
                       ast.literal_eval(str(d["meta"])),
                       [tuple(map(int, b)) for b in d["bonds"]] if "bonds" in d.files else [],
                       d["md_samples"] if "md_samples" in d.files and d["md_samples"].size else None)
        return cls.from_table(path)

    @classmethod
    def from_table(cls, path) -> "PESResult":
        """Pre-computed PES: CSV/TSV/.dat/.xvg with columns phi, psi, E (any order of rows).

        Conformations are not stored in such files, so an idealised Ace-Ala-Nme heavy-atom model is used for
        the molecule view.
        """
        rows = []
        with open(path) as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s[0] in "#@;":
                    continue
                parts = re.split(r"[,\s;]+", s)
                try:
                    rows.append([float(x) for x in parts[:3]])
                except ValueError:
                    continue  # header
        if not rows:
            raise ValueError(f"No numeric (phi, psi, E) rows in {path}")
        a = np.array(rows)
        phi = np.unique(np.round(wrap180(a[:, 0]), 4))
        psi = np.unique(np.round(wrap180(a[:, 1]), 4))
        E = np.full((len(phi), len(psi)), np.nan)
        for p, q, e in a:
            E[np.searchsorted(phi, round(float(wrap180(p)), 4)), np.searchsorted(psi, round(float(wrap180(q)), 4))] = e
        P, Q = np.meshgrid(phi, psi, indexing="ij")
        coords = np.array([[build_dipeptide(p, q) for q in psi] for p in phi])
        atoms = [(ri, rn, nm) for rn, ri, nm in DIPEPTIDE_ATOMS]
        bonds = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6), (4, 7), (5, 8), (8, 9)]
        return cls(phi, psi, E, P.copy(), Q.copy(), coords, atoms, (1, 3, 4, 5), (3, 4, 5, 8),
                   {"source": os.path.abspath(path), "idealised_geometry": True}, bonds)


# ---------------------------------------------------------------------------------------- scanner
class PESScanner:
    def __init__(self, cfg: PESConfig):
        self.cfg = cfg
        self.engine = GromacsEngine(cfg.gmx, cfg.use_wsl, cfg.wsl_distro)
        self.workdir = os.path.abspath(cfg.workdir) if cfg.workdir else tempfile.mkdtemp(prefix="mdviz_pes_")
        os.makedirs(self.workdir, exist_ok=True)
        self.ref_coords = None
        self.recs = None
        self.bonds = None
        self.top_text = None
        self._procs: list = []
        self._cancel = threading.Event()

    # -------------------------------------------------------------- preparation
    def prepare(self, log=print):
        cfg, wd = self.cfg, self.workdir
        log(f"Work folder: {wd}")
        log(f"GROMACS {self.engine.check()} ({'WSL' if self.engine.use_wsl else 'native'})")
        ref = os.path.join(wd, "reference")
        os.makedirs(ref, exist_ok=True)
        if cfg.molecule == "ala2":
            write_pdb(os.path.join(ref, "ala2_heavy.pdb"), DIPEPTIDE_ATOMS, build_dipeptide(-80, 80))
            log(f"pdb2gmx: Ace-Ala-Nme with {cfg.forcefield}")
            self.engine.run(["pdb2gmx", "-f", "ala2_heavy.pdb", "-o", "conf.gro", "-p", "topol.top",
                             "-ff", cfg.forcefield, "-water", "none", "-ignh"], cwd=ref)
        else:
            shutil.copy(cfg.custom_gro, os.path.join(ref, "conf.gro"))
            with open(os.path.join(ref, "topol.top"), "w") as fh:
                fh.write(flatten_topology(os.path.abspath(cfg.custom_top)))
        _, recs, xyz, _ = read_gro(os.path.join(ref, "conf.gro"))
        self.recs = recs
        extent_nm = float(np.max(np.linalg.norm(xyz[:, None] - xyz[None], axis=2))) / 10.0
        if extent_nm + 0.6 > cfg.cutoff:
            cfg.cutoff = round(extent_nm + 0.6, 2)
            log(f"Molecule spans {extent_nm:.2f} nm: using {cfg.cutoff} nm cut-offs (still vacuum-exact)")
        with open(os.path.join(ref, "topol.top")) as fh:
            self.top_text = fh.read()
        self.bonds = top_bonds(os.path.join(ref, "topol.top"))
        if not self.bonds:
            raise ValueError("No [ bonds ] found in the molecule topology.")
        if cfg.phi_atoms and cfg.psi_atoms:
            self.phi_q, self.psi_q = tuple(cfg.phi_atoms), tuple(cfg.psi_atoms)
        else:
            self.phi_q, self.psi_q = find_backbone_dihedrals(recs)
        self.phi_mask = moving_side(len(recs), self.bonds, self.phi_q[1], self.phi_q[2])
        self.psi_mask = moving_side(len(recs), self.bonds, self.psi_q[1], self.psi_q[2])
        names = lambda q: "-".join(f"{recs[i][2]}{recs[i][0]}" for i in q)
        log(f"{len(recs)} atoms · phi = {names(self.phi_q)} · psi = {names(self.psi_q)}")
        # relax the starting structure without restraints -> reference geometry for all grid points
        box = 2 * cfg.cutoff + 1.2
        write_gro(os.path.join(ref, "start.gro"), recs, xyz - xyz.mean(axis=0) + box * 5, box)
        self._write_mdp(os.path.join(wd, "em.mdp"))
        log("Relaxing the reference structure …")
        self.engine.run(["grompp", "-f", "../em.mdp", "-c", "start.gro", "-p", "topol.top", "-o", "ref.tpr",
                         "-maxwarn", "5"], cwd=ref)
        self.engine.run(["mdrun", "-deffnm", "ref", "-ntmpi", "1", "-ntomp", "1", "-nb", "cpu"], cwd=ref)
        _, _, self.ref_coords, _ = read_gro(os.path.join(ref, "ref.gro"))
        e = read_energy(os.path.join(ref, "ref.edr"))
        log(f"Reference: phi = {dihedral(*self.ref_coords[list(self.phi_q)]):.1f}°, "
            f"psi = {dihedral(*self.ref_coords[list(self.psi_q)]):.1f}°, E = {e[0]:.2f} kJ/mol")
        return self

    def _write_mdp(self, path):
        c = self.cfg
        with open(path, "w") as fh:
            fh.write(EM_MDP.format(integrator=c.integrator, emtol=c.emtol, nsteps=c.nsteps, cutoff=c.cutoff,
                                   epsilon_r=c.epsilon_r))

    def _restrained_top(self, phi, psi) -> str:
        k = self.cfg.restraint_k
        block = ("\n[ dihedral_restraints ]\n; ai aj ak al type phi dphi kfac (MDViz PES)\n"
                 + "".join(f"{q[0] + 1} {q[1] + 1} {q[2] + 1} {q[3] + 1} 1 {ang:.3f} 0 {k}\n"
                           for q, ang in ((self.phi_q, phi), (self.psi_q, psi))) + "\n")
        lines = self.top_text.splitlines(keepends=True)
        # insert inside the first moleculetype: before its position-restraint include, the next
        # moleculetype/#include after it, or [ system ]
        seen_mol = False
        for n, ln in enumerate(lines):
            s = ln.strip().lower()
            if s.startswith("[") and "moleculetype" in s:
                if seen_mol:
                    return "".join(lines[:n]) + block + "".join(lines[n:])
                seen_mol = True
                continue
            if seen_mol and (s.startswith("; include position restraint") or s.startswith("#ifdef posres")
                             or s.startswith("; include water") or s.startswith("; include topology for ions")
                             or re.match(r"\[\s*system\s*\]", s)):
                return "".join(lines[:n]) + block + "".join(lines[n:])
        raise ValueError("Could not locate the molecule definition in the topology.")

    # -------------------------------------------------------------- scan
    def cancel(self):
        self._cancel.set()
        try:
            open(os.path.join(self.workdir, "STOP"), "w").close()
        except OSError:
            pass
        for p in list(self._procs):
            try:
                p.kill()
            except Exception:
                pass

    def run(self, progress=None, order="snake") -> PESResult:
        """Run restrained minimisations over the whole grid.

        progress(done, total, i, j, energy, coords) is called from worker threads as points finish.
        Points whose outputs already exist in the work folder are reused (resume after cancel).
        """
        if self.ref_coords is None:
            raise RuntimeError("Call prepare() first")
        cfg = self.cfg
        grid = cfg.grid()
        n = len(grid)
        E = np.full((n, n), np.nan)
        PA = np.full((n, n), np.nan)
        QA = np.full((n, n), np.nan)
        X = np.full((n, n, len(self.recs), 3), np.nan, dtype=np.float32)
        stop = os.path.join(self.workdir, "STOP")
        if os.path.exists(stop):
            os.remove(stop)
        self._cancel.clear()
        box = 2 * cfg.cutoff + 1.2
        todo, done = [], 0
        lock = threading.Lock()

        def collect(i, j, d):
            nonlocal done
            try:
                e, _ = read_energy(os.path.join(d, "em.edr"))
                _, _, xyz, _ = read_gro(os.path.join(d, "em.gro"))
            except Exception:
                return False
            with lock:
                E[i, j] = e
                X[i, j] = xyz
                PA[i, j] = dihedral(*xyz[list(self.phi_q)])
                QA[i, j] = dihedral(*xyz[list(self.psi_q)])
                done += 1
                d_ = done
            if progress:
                progress(d_, n * n, i, j, e, xyz)
            return True

        pts = os.path.join(self.workdir, "points")
        os.makedirs(pts, exist_ok=True)
        for i, phi in enumerate(grid):
            for j, psi in enumerate(grid):
                d = os.path.join(pts, f"p{i:03d}_{j:03d}")
                if os.path.exists(os.path.join(d, "em.gro")) and collect(i, j, d):
                    continue
                os.makedirs(d, exist_ok=True)
                x = rotate_dihedral(self.ref_coords, self.phi_q, phi, self.phi_mask)
                x = rotate_dihedral(x, self.psi_q, psi, self.psi_mask)
                write_gro(os.path.join(d, "conf.gro"), self.recs, x - x.mean(axis=0) + box * 5, box)
                with open(os.path.join(d, "topol.top"), "w") as fh:
                    fh.write(self._restrained_top(phi, psi))
                todo.append((i, j, d))
        workers = cfg.workers or max(1, min(12, (os.cpu_count() or 4) - 2))
        batches = [todo[w::workers] for w in range(workers)]
        errors = []

        def worker(batch):
            if not batch or self._cancel.is_set():
                return
            names = " ".join(os.path.basename(d) for _, _, d in batch)
            script = (
                f"cd points; for p in {names}; do "
                "[ -f ../STOP ] && exit 0; "
                "( cd $p && gmx grompp -f ../../em.mdp -c conf.gro -p topol.top -o em.tpr -maxwarn 5 "
                ">grompp.log 2>&1 && gmx mdrun -deffnm em -ntmpi 1 -ntomp 1 -nb cpu >mdrun.log 2>&1 ); "
                "echo \"DONE $p $?\"; done")
            if cfg.gmx != "gmx":
                script = script.replace("gmx ", f"{cfg.gmx} ")
            lookup = {os.path.basename(d): (i, j, d) for i, j, d in batch}
            p = self.engine.bash(script, self.workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._procs.append(p)
            for line in p.stdout:
                m = re.match(r"DONE (\S+) (\d+)", line.strip())
                if m and m.group(1) in lookup:
                    i, j, d = lookup[m.group(1)]
                    if m.group(2) != "0" or not collect(i, j, d):
                        errors.append(d)
            p.wait()

        threads = [threading.Thread(target=worker, args=(b,), daemon=True) for b in batches]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._procs.clear()
        meta = dict(forcefield=cfg.forcefield if cfg.molecule == "ala2" else os.path.basename(cfg.custom_top),
                    step=cfg.step, restraint_k=cfg.restraint_k, integrator=cfg.integrator, emtol=cfg.emtol,
                    epsilon_r=cfg.epsilon_r,
                    workdir=self.workdir, seconds=round(time.time() - t0, 1), failed=len(errors),
                    cancelled=self._cancel.is_set(), molecule=cfg.molecule)
        if errors:
            meta["first_failure_log"] = os.path.join(errors[0], "mdrun.log")
        return PESResult(grid, grid.copy(), E, PA, QA, X, self.recs, self.phi_q, self.psi_q, meta, list(self.bonds))


def read_energy(edr_path):
    """(potential - restraint energy, restraint energy) at the last frame of a GROMACS .edr (kJ/mol)."""
    import pyedr
    d = pyedr.edr_to_dict(edr_path, verbose=False)
    pot = float(d["Potential"][-1])
    rest = 0.0
    for key in d:
        if key.lower().startswith("dih. rest"):
            rest = float(d[key][-1])
    return pot - rest, rest


# ---------------------------------------------------------------------------------------- stationary points
def periodic_spline(res: PESResult):
    """Bicubic spline of E(phi, psi), tiled 3x3 so it is smooth across +-180."""
    from scipy.interpolate import RectBivariateSpline
    E = res.energy.copy()
    if np.isnan(E).any():  # gaps (failed points) are filled high for fitting only
        E = np.where(np.isnan(E), np.nanmax(E), E)
    P = np.concatenate([res.phi - 360, res.phi, res.phi + 360])
    Q = np.concatenate([res.psi - 360, res.psi, res.psi + 360])
    return RectBivariateSpline(P, Q, np.tile(E, (3, 3)), kx=3, ky=3, s=0)


def energy_at(res: PESResult, points) -> np.ndarray:
    """Interpolated dE (kJ/mol, relative to the surface minimum) at (phi, psi) points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    sp = periodic_spline(res)
    return sp.ev(wrap180(pts[:, 0]), wrap180(pts[:, 1])) - np.nanmin(res.energy)


def md_strain_summary(res: PESResult) -> dict | None:
    """How far up the gas-phase surface the MD-sampled torsions sit (a 2D conformational-strain estimate)."""
    if res.md_samples is None or not len(res.md_samples):
        return None
    e = np.clip(energy_at(res, res.md_samples), 0, None)
    return dict(n=len(e), mean=float(e.mean()), median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                within_5=float((e <= 5).mean() * 100), within_10=float((e <= 10).mean() * 100))


def _basin_merge_tree(E: np.ndarray, threshold: float):
    """0-dimensional persistence on the periodic grid (8-neighbour), elder rule.

    Returns significant minima {root: grid index} and saddles [(grid index, root_a, root_b)], where a
    minimum is significant when its basin must climb >= threshold to reach a lower basin, and each saddle
    is the lowest crossing point that joins two significant basins.
    """
    n, m = E.shape
    flat = E.ravel()
    order = [k for k in np.argsort(flat, kind="stable") if np.isfinite(flat[k])]
    parent = -np.ones(n * m, dtype=np.int64)
    emin = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    significant, saddles = set(), []
    for k in order:
        i, j = divmod(int(k), m)
        roots = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    nb = ((i + di) % n) * m + (j + dj) % m
                    if parent[nb] != -1:
                        roots.add(find(nb))
        if not roots:
            parent[k] = k
            emin[k] = flat[k]
            continue
        roots = sorted(roots, key=lambda r: emin[r])
        keep = roots[0]
        parent[k] = keep
        for r in roots[1:]:
            if flat[k] - emin[r] >= threshold:
                significant.update((r, keep))
                saddles.append((int(k), r, keep))
            parent[r] = keep
    glob = min(emin, key=emin.get)
    significant.add(glob)
    return {r: r for r in significant}, saddles


def find_stationary_points(res: PESResult, min_persistence=2.0, max_saddles=10):
    """Energy basins (minima) and the saddle points (lowest barriers) that connect them.

    Basins come from a persistence analysis of the grid: a local minimum is kept only if its basin has to
    climb at least `min_persistence` kJ/mol to reach a lower basin, so shallow bumps of a relaxed scan are
    ignored. Each saddle is the lowest crossing between two basins ("M1-M3"). Positions and energies are
    then refined on a periodic bicubic spline (Newton on the gradient, classified by the Hessian).
    """
    if not np.isfinite(res.energy).any():
        return []
    n, m = res.energy.shape
    step = max(360.0 / n, 360.0 / m)
    sp = periodic_spline(res)
    f = lambda x, y, dx=0, dy=0: float(sp.ev(x, y, dx=dx, dy=dy))
    roots, merges = _basin_merge_tree(res.energy, min_persistence)

    def refine(x, y, want):
        x0, y0 = x, y
        for _ in range(25):
            g = np.array([f(x, y, 1, 0), f(x, y, 0, 1)])
            H = np.array([[f(x, y, 2, 0), f(x, y, 1, 1)], [f(x, y, 1, 1), f(x, y, 0, 2)]])
            try:
                d = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            x, y = x - d[0], y - d[1]
            if np.hypot(wrap180(x - x0), wrap180(y - y0)) > 1.5 * step:
                break
            if np.abs(d).max() < 1e-4:
                H = np.array([[f(x, y, 2, 0), f(x, y, 1, 1)], [f(x, y, 1, 1), f(x, y, 0, 2)]])
                det, tr = np.linalg.det(H), np.trace(H)
                if (want == "minimum" and det > 0 and tr > 0) or (want == "saddle" and det < 0):
                    x, y = float(wrap180(x)), float(wrap180(y))
                    return x, y, f(x, y)
                break
        # spline and grid disagree (rugged surface): keep the grid point
        return float(x0), float(y0), float(res.energy.ravel()[res.index(x0, y0)[0] * m + res.index(x0, y0)[1]])

    mins = []
    for r in roots:
        i, j = divmod(r, m)
        x, y, e = refine(res.phi[i], res.psi[j], "minimum")
        mins.append((e, x, y, r))
    mins.sort()
    label = {r: f"M{k}" for k, (_, _, _, r) in enumerate(mins, 1)}
    e0 = min([np.nanmin(res.energy)] + [e for e, _, _, _ in mins])
    region = res.meta.get("x_label") is None
    out = [dict(type="minimum", label=label[r], phi=x, psi=y, energy=e, rel=e - e0,
                region=ramachandran_region(x, y) if region else "basin") for e, x, y, r in mins]
    sads = []
    for k, a, b in merges:
        i, j = divmod(k, m)
        x, y, e = refine(res.phi[i], res.psi[j], "saddle")
        la, lb = sorted((label.get(a, "?"), label.get(b, "?")), key=lambda t: int(t[1:]) if t[1:].isdigit() else 99)
        sads.append((e, x, y, f"{la}↔{lb}"))
    sads.sort()
    for k, (e, x, y, link) in enumerate(sads[:max_saddles], 1):
        out.append(dict(type="saddle", label=f"TS{k}", phi=x, psi=y, energy=e, rel=e - e0, region=link))
    return out


def ramachandran_region(phi, psi) -> str:
    """Conventional alanine-dipeptide conformer names."""
    if phi < 0:
        if psi > 120 or psi < -150:
            return "β / C5"
        if 30 < psi <= 120:
            return "C7eq / PII"
        if -120 < psi <= 30:
            return "αR"
        return "bridge"
    if -150 < psi <= -20:
        return "C7ax"
    if -20 < psi <= 100:
        return "αL"
    return "αʹ / other"


# ---------------------------------------------------------------------------------------- viewer bridge
def write_topology_pdb(path, atoms, coords, bonds):
    """PDB with CONECT records so the molecule viewer gets the force-field bonds."""
    from .chem import element_from_name
    with open(path, "w") as fh:
        for k, ((ri, rn, name), p) in enumerate(zip(atoms, coords), start=1):
            el = element_from_name(name, rn)
            nm = f" {name:<3s}" if len(name) < 4 else name
            fh.write(f"ATOM  {k:5d} {nm} {rn:<3s} A{ri:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00"
                     f"          {el.capitalize():>2s}\n")
        nb = {}
        for i, j in bonds:
            nb.setdefault(i, []).append(j)
            nb.setdefault(j, []).append(i)
        for i in sorted(nb):
            for s in range(0, len(nb[i]), 4):
                fh.write("CONECT" + f"{i + 1:5d}" + "".join(f"{j + 1:5d}" for j in nb[i][s:s + 4]) + "\n")
        fh.write("END\n")


def viewer_frames(res: PESResult):
    """(frames (n*m, atoms, 3), per-frame labels, reference coords). Frame index = i * len(psi) + j."""
    n, m = res.energy.shape
    X = np.asarray(res.coords, dtype=np.float32).reshape(n * m, -1, 3).copy()
    ok = np.isfinite(X).all(axis=(1, 2))
    ref = X[ok][0] if ok.any() else np.zeros(X.shape[1:], np.float32)
    X[~ok] = ref
    rel = res.rel
    xl, yl = (lab.split(" (")[0] for lab in res.labels)   # "\u03c41 (N-C-C1-C2)" -> "\u03c41"
    labels = []
    for i in range(n):
        for j in range(m):
            e = rel[i, j]
            etxt = f"\u0394E = {e:.2f} kJ/mol" if np.isfinite(e) else "not computed"
            labels.append(f"{xl} = {res.phi[i]:.0f}\u00b0, {yl} = {res.psi[j]:.0f}\u00b0  \u00b7  {etxt}")
    return X, labels, ref
