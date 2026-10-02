"""Tests for the PES backend. GROMACS-dependent tests run only with MDVIZ_GMX_TESTS=1."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdviz.pes import (DIPEPTIDE_ATOMS, GromacsEngine, PESConfig, PESResult, PESScanner, build_dipeptide,  # noqa: E402
                       dihedral, find_stationary_points, moving_side, rotate_dihedral, viewer_frames, wrap180)

PHI = (1, 3, 4, 5)   # C(ACE)-N-CA-C in DIPEPTIDE_ATOMS order
PSI = (3, 4, 5, 8)   # N-CA-C-N(NME)
BONDS = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6), (4, 7), (5, 8), (8, 9)]


def test_dihedral_sign_convention():
    # IUPAC: looking along b->c, positive = clockwise rotation of the far atom
    p = [np.array(v, float) for v in ([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1])]
    assert dihedral(*p) == pytest.approx(90.0)
    p[3] = np.array([0, -1, 1.0])
    assert dihedral(*p) == pytest.approx(-90.0)


@pytest.mark.parametrize("phi,psi", [(-60, -45), (-150, 150), (60, 40), (179, -179), (0, 0)])
def test_builder_hits_requested_angles(phi, psi):
    x = build_dipeptide(phi, psi)
    assert dihedral(*x[list(PHI)]) == pytest.approx(phi, abs=1e-6)
    assert wrap180(dihedral(*x[list(PSI)]) - psi) == pytest.approx(0, abs=1e-6)


def test_builder_is_L_alanine():
    x = build_dipeptide()
    # N-C-CA-CB improper is about +122 deg for L-amino acids (checked against the 3M3L crystal structure)
    assert dihedral(x[3], x[5], x[4], x[7]) == pytest.approx(122.7, abs=1.0)


def test_moving_side_and_rotation_preserve_bonds():
    x = build_dipeptide(-80, 70)
    m_phi = moving_side(10, BONDS, PHI[1], PHI[2])
    m_psi = moving_side(10, BONDS, PSI[1], PSI[2])
    assert m_phi.tolist() == [False, False, False, False, True, True, True, True, True, True]
    assert np.where(m_psi)[0].tolist() == [5, 6, 8, 9]
    y = rotate_dihedral(rotate_dihedral(x, PHI, 65.0, m_phi), PSI, -40.0, m_psi)
    assert dihedral(*y[list(PHI)]) == pytest.approx(65.0, abs=1e-6)
    assert dihedral(*y[list(PSI)]) == pytest.approx(-40.0, abs=1e-6)
    d = lambda z: np.array([np.linalg.norm(z[i] - z[j]) for i, j in BONDS])
    np.testing.assert_allclose(d(x), d(y), atol=1e-9)


def test_moving_side_rejects_ring_bond():
    ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    with pytest.raises(ValueError):
        moving_side(4, ring, 1, 2)


def _analytic_result(step=10.0, two_basins=True):
    g = np.arange(-180, 180, step, dtype=float)
    P, Q = np.meshgrid(g, g, indexing="ij")
    if two_basins:  # minima (0,0) and (180,0) at -16; lowest crossing between them at (+-90, 0) = +4
        E = -10 * np.cos(np.radians(2 * P)) - 6 * np.cos(np.radians(Q))
    else:
        E = -10 * np.cos(np.radians(P)) - 6 * np.cos(np.radians(Q))
    coords = np.array([[build_dipeptide(p, q) for q in g] for p in g])
    return PESResult(g, g.copy(), E, P.copy(), Q.copy(), coords,
                     [(ri, rn, nm) for rn, ri, nm in DIPEPTIDE_ATOMS], PHI, PSI, {"test": True}, BONDS)


def test_basins_and_connecting_saddle():
    pts = find_stationary_points(_analytic_result())
    mins = [p for p in pts if p["type"] == "minimum"]
    sads = [p for p in pts if p["type"] == "saddle"]
    assert sorted(round(abs(p["phi"])) for p in mins) == [0, 180]
    assert all(abs(p["psi"]) < 0.5 and abs(p["rel"]) < 1e-6 for p in mins)
    assert len(sads) == 1 and sads[0]["region"] == "M1↔M2"
    assert abs(sads[0]["phi"]) == pytest.approx(90, abs=0.5) and sads[0]["psi"] == pytest.approx(0, abs=0.5)
    assert sads[0]["rel"] == pytest.approx(20.0, abs=0.05)


def test_shallow_bumps_are_not_basins():
    r = _analytic_result(two_basins=False)
    rng = np.random.default_rng(0)
    r.energy = r.energy + rng.uniform(0, 0.8, r.energy.shape)   # relaxed-scan roughness
    mins = [p for p in find_stationary_points(r, min_persistence=2.0) if p["type"] == "minimum"]
    assert len(mins) == 1 and abs(mins[0]["phi"]) < 15 and abs(mins[0]["psi"]) < 15


def test_energy_at_interpolates_grid():
    from mdviz.pes import energy_at
    r = _analytic_result(two_basins=False)
    e = energy_at(r, [[0, 0], [90, 0], [45, 45]])
    exact = lambda p, q: -10 * np.cos(np.radians(p)) - 6 * np.cos(np.radians(q)) + 16
    np.testing.assert_allclose(e, [exact(0, 0), exact(90, 0), exact(45, 45)], atol=0.05)


def test_flatten_and_rotatable_torsions(tmp_path):
    from mdviz.ligand_pes import read_molecule, rotatable_torsions
    from mdviz.pes import flatten_topology
    # C1-C2 attached to a C3..C6 ring, with O1 on the ring
    atoms = [f"{i} c3 1 MOL {n} {i} 0 12" for i, n in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "O1"], 1)]
    bonds = ["1 2 1", "2 3 1", "3 4 1", "4 5 1", "5 6 1", "6 3 1", "4 7 1"]
    (tmp_path / "mol.itp").write_text("\n".join(["[ moleculetype ]", "MOL 3", "[ atoms ]"] + atoms
                                                + ["[ bonds ]"] + bonds) + "\n")
    (tmp_path / "mol.top").write_text("\n".join(['[ defaults ]', '1 2 yes 0.5 0.8333', '#include "mol.itp"',
                                                 '#include "amber99sb-ildn.ff/ions.itp"', '[ system ]', 'x',
                                                 '[ molecules ]', 'MOL 1']) + "\n")
    flat = flatten_topology(str(tmp_path / "mol.top"))
    assert "[ bonds ]" in flat and '#include "amber99sb-ildn.ff/ions.itp"' in flat
    names, bonds = read_molecule(str(tmp_path / "mol.top"))
    assert names[:2] == ["C1", "C2"] and len(bonds) == 7
    tors = rotatable_torsions(names, bonds)
    # only C2-C3 is rotatable: C1-C2 has no heavy atom beyond C1, ring bonds are excluded, C4-O1 is terminal
    assert [t.quad[1:3] for t in tors] == [(1, 2)]


def test_save_load_roundtrip(tmp_path):
    r = _analytic_result(30.0)
    f = tmp_path / "pes.npz"
    r.save(f)
    q = PESResult.load(str(f))
    np.testing.assert_allclose(q.energy, r.energy)
    np.testing.assert_allclose(q.coords, r.coords, atol=1e-5)
    assert q.atoms == r.atoms and q.bonds == BONDS and q.phi_atoms == PHI and q.meta == {"test": True}


def test_table_import(tmp_path):
    f = tmp_path / "pes.csv"
    rows = ["phi,psi,E"] + [f"{p},{q},{p * 0.01 + q * 0.001}" for p in range(-180, 180, 30) for q in range(-180, 180, 30)]
    f.write_text("\n".join(rows))
    r = PESResult.load(str(f))
    assert r.energy.shape == (12, 12) and np.isfinite(r.energy).all()
    assert r.energy[r.index(60, -30)] == pytest.approx(0.6 - 0.03)
    assert r.meta["idealised_geometry"] and r.coords.shape == (12, 12, 10, 3)


def test_viewer_frames_order_and_labels():
    r = _analytic_result(30.0)
    X, labels, _ = viewer_frames(r)
    i, j = r.index(60, -90)
    f = i * len(r.psi) + j
    assert dihedral(*X[f][list(PHI)]) == pytest.approx(60, abs=1e-4)
    assert labels[f].startswith("\u03c6 = 60\u00b0, \u03c8 = -90\u00b0")


def test_restraints_go_inside_the_molecule(tmp_path):
    top = """#include "amber99sb-ildn.ff/forcefield.itp"
[ moleculetype ]
Protein 3
[ atoms ]
     1 CT 1 ACE CH3 1 0 12
[ bonds ]
1 2 1
; Include Position restraint file
#ifdef POSRES
#include "posre.itp"
#endif
; Include topology for ions
#include "amber99sb-ildn.ff/ions.itp"
[ system ]
x
[ molecules ]
Protein 1
"""
    sc = PESScanner.__new__(PESScanner)
    sc.cfg = PESConfig(restraint_k=1234)
    sc.top_text = top
    sc.phi_q, sc.psi_q = (4, 6, 8, 14), (6, 8, 14, 16)
    out = sc._restrained_top(-80.0, 70.0)
    block = out.index("[ dihedral_restraints ]")
    assert out.index("[ bonds ]") < block < out.index("; Include Position restraint file")
    assert "5 7 9 15 1 -80.000 0 1234" in out and "7 9 15 17 1 70.000 0 1234" in out


def test_wsl_path_translation():
    assert GromacsEngine.to_posix(r"C:\Users\Nik Fitri\OneDrive\x y\z.gro") == "/mnt/c/Users/Nik Fitri/OneDrive/x y/z.gro"


@pytest.mark.skipif(os.environ.get("MDVIZ_GMX_TESTS") != "1", reason="set MDVIZ_GMX_TESTS=1 to run GROMACS")
def test_gromacs_coarse_scan(tmp_path):
    cfg = PESConfig(step=60.0, workdir=str(tmp_path / "scan"))
    res = PESScanner(cfg).prepare(log=lambda *_: None).run()
    assert np.isfinite(res.energy).all() and res.meta["failed"] == 0
    dev = np.abs(wrap180(res.phi_actual - res.phi[:, None]))
    assert dev.max() < 3.0
    # C7eq / beta region (phi < 0) is lower than the phi = +120 region in vacuum AMBER
    assert res.energy[res.index(-60, 60)] < res.energy[res.index(120, 60)]
