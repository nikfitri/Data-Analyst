"""Chemistry tables: elements, radii, colours and residue classes."""
from __future__ import annotations

import re

import numpy as np

# Element -> (mass, van der Waals radius in Angstrom, CPK-style colour)
ELEMENTS = {
    "H": (1.008, 1.10, "#f2f2f2"),
    "C": (12.011, 1.70, "#909090"),
    "N": (14.007, 1.55, "#3050f8"),
    "O": (15.999, 1.52, "#ff0d0d"),
    "F": (18.998, 1.47, "#90e050"),
    "NA": (22.990, 2.27, "#ab5cf2"),
    "MG": (24.305, 1.73, "#8aff00"),
    "P": (30.974, 1.80, "#ff8000"),
    "S": (32.06, 1.80, "#e6c828"),
    "CL": (35.45, 1.75, "#1ff01f"),
    "K": (39.098, 2.75, "#8f40d4"),
    "CA": (40.078, 2.31, "#3dff00"),
    "MN": (54.938, 2.00, "#9c7ac7"),
    "FE": (55.845, 2.00, "#e06633"),
    "CO": (58.933, 2.00, "#f090a0"),
    "NI": (58.693, 1.63, "#50d050"),
    "CU": (63.546, 1.40, "#c88033"),
    "ZN": (65.38, 1.39, "#7d80b0"),
    "SE": (78.971, 1.90, "#ffa100"),
    "BR": (79.904, 1.85, "#a62929"),
    "I": (126.904, 1.98, "#940094"),
    "X": (0.0, 1.50, "#ff1493"),
}

_MASS_TABLE = sorted((v[0], k) for k, v in ELEMENTS.items() if v[0] > 0)
ION_RESNAMES = {"NA", "CL", "K", "MG", "CA", "ZN", "SOD", "CLA", "POT", "CAL", "NA+", "CL-", "K+", "MN", "FE", "CU", "LI", "CS", "RB"}
WATER_RESNAMES = {"SOL", "WAT", "HOH", "TIP3", "TIP3P", "TIP4P", "TIP4", "TIP5P", "SPC", "SPCE", "T3P", "T4P", "H2O", "OPC"}

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
    "CYX", "CYM", "ASH", "GLH", "LYN", "ACE", "NME", "NMA",
}
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR", "CYS", "CYX"}
ACIDIC_RES = {"ASP", "GLU"}
BASIC_RES = {"ARG", "LYS", "HIP", "HSP"}
POLAR_RES = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "HID", "HIE", "HSD", "HSE", "GLY"}

# Kyte-Doolittle hydropathy
KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9, "ALA": 1.8,
    "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3, "PRO": -1.6,
    "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}

SS_COLORS = {"H": "#e0569b", "G": "#8b5cd6", "I": "#b04fd0", "E": "#f2c12e", "B": "#c8a020",
             "T": "#62c3d6", "S": "#d8d8d8", "-": "#d8d8d8", "C": "#d8d8d8"}

CHAIN_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
                 "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

# Colours used for interaction types (dashed lines, legends, plots)
INTERACTION_COLORS = {          # validated categorical palette, fixed order (CVD-safe)
    "H-bond": "#2a78d6",
    "Salt bridge": "#eb6834",
    "Hydrophobic": "#1baf7a",
    "Pi-stacking": "#eda100",
    "Cation-pi": "#e87ba4",
    "Halogen bond": "#008300",
}

AROMATIC_RINGS = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TRP": [("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"), ("CG", "CD1", "NE1", "CE2", "CD2")],
    "HIS": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HID": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HIE": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HIP": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HSD": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HSE": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HSP": [("CG", "ND1", "CD2", "CE1", "NE2")],
}
POSITIVE_GROUPS = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIP": ("ND1", "NE2"), "HSP": ("ND1", "NE2")}
NEGATIVE_GROUPS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
CATION_ATOMS = {"LYS": ("NZ",), "ARG": ("CZ",)}


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_float(color: str) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in hex_to_rgb(color))


def element_from_mass(mass: float) -> str | None:
    if mass <= 0:
        return None
    if mass < 4.2:  # plain or mass-repartitioned hydrogen
        return "H"
    best = min(_MASS_TABLE, key=lambda mk: abs(mk[0] - mass))
    return best[1] if abs(best[0] - mass) < 0.6 else None


def element_from_name(name: str, resname: str = "", n_res_atoms: int = 2) -> str:
    letters = re.sub(r"[^A-Za-z]", "", name).upper()
    if not letters:
        return "X"
    # Single-atom residues are ions: CA/CL/NA etc. are the element itself
    if n_res_atoms == 1 and letters[:2] in ELEMENTS:
        return letters[:2]
    # Ligands: halogens are named Cl1/Br2; other two-letter names (NA, CO...) stay N/C
    if resname.upper() not in AMINO_ACIDS and letters[:2] in {"CL", "BR"}:
        return letters[:2]
    return letters[0] if letters[0] in ELEMENTS else "X"


def guess_elements(names, resnames, masses=None, existing=None, res_sizes=None) -> np.ndarray:
    """Best-effort element assignment: existing valid elements > masses > names."""
    n = len(names)
    out = np.empty(n, dtype=object)
    for i in range(n):
        el = None
        if existing is not None:
            e = str(existing[i]).strip().upper()
            if e in ELEMENTS and e != "X":
                el = e
        if el is None and masses is not None:
            el = element_from_mass(float(masses[i]))
        if el is None:
            el = element_from_name(str(names[i]), str(resnames[i]), 2 if res_sizes is None else int(res_sizes[i]))
        out[i] = el
    return out


def element_colors(elements, carbon_color: str | None = None) -> np.ndarray:
    cols = np.array([hex_to_rgb(ELEMENTS.get(e, ELEMENTS["X"])[2]) for e in elements], dtype=np.uint8).reshape(-1, 3)
    if carbon_color:
        cols[np.asarray(elements) == "C"] = hex_to_rgb(carbon_color)
    return cols


def vdw_radii(elements) -> np.ndarray:
    return np.array([ELEMENTS.get(e, ELEMENTS["X"])[1] for e in elements], dtype=np.float32)
