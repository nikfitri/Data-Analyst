"""Batch publication renders for every complex that exists in the thesis folder.

Run:  ..\\.venv\\Scripts\\python.exe examples\\render_figures.py        (from the MDViz folder)
Output goes to  MD simulation/Figures/<ligand>/ .
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

from mdviz.render import MovieOptions, render_image, render_movie  # noqa: E402
from mdviz.state import Session, apply_style  # noqa: E402
from mdviz.system import MolSystem  # noqa: E402

J = lambda *p: os.path.join(ROOT, *p)
COMPLEXES = [
    # name, topology, trajectories, source note, make movie
    ("52E5", J("GROMACS", "runs", "52E5", "md_10ns_52E5.tpr"), [J("GROMACS", "runs", "52E5", "md_10ns_52E5.xtc")],
     "GROMACS MD", True),
    ("52E17", J("GROMACS", "runs", "52E17", "nvt.tpr"), [J("GROMACS", "runs", "52E17", "52E17_video_centered.xtc")],
     "GROMACS NVT", False),
    ("52E17_YASARA", J("52E17", "Run 1 52E17 GluA2 Protein Complex Soln 5_last.pdb"), [], "YASARA last frame", False),
    ("52E2_YASARA", J("MD simulation", "52E2", "ADT 52E2 AD2404 Complex_last.pdb"), [], "YASARA last frame", False),
]
only = set(sys.argv[1:])


def discover_gromacs_runs():
    """Every GROMACS/runs/<ligand> folder with a production trajectory (skips nvt/npt/em)."""
    import glob
    found = []
    for run in sorted(glob.glob(J("GROMACS", "runs", "*", ""))):
        name = os.path.basename(os.path.normpath(run))
        xtcs = [x for x in glob.glob(os.path.join(run, "*.xtc"))
                if not os.path.basename(x).lower().startswith(("nvt", "npt", "em", "."))
                and "video" not in os.path.basename(x).lower()]
        if not xtcs:
            continue
        xtc = max(xtcs, key=os.path.getsize)
        stem = os.path.splitext(xtc)[0]
        tprs = [stem + ".tpr"] if os.path.exists(stem + ".tpr") else sorted(glob.glob(os.path.join(run, "md*.tpr")))
        if tprs:
            found.append((name, tprs[0], [xtc], "GROMACS MD", True))
    return found


known = {c[0] for c in COMPLEXES}
COMPLEXES += [c for c in discover_gromacs_runs() if c[0] not in known]


def session_for(system, style):
    s = Session(topology=system.topology, trajectories=system.trajectories)
    if system.n_frames > 1 and system.detect_pbc_problems()["broken"]:
        s.pbc_fix = True
        s.align = True
    apply_style(s, system, style)
    system.set_processing(s.pbc_fix, s.align, s.align_selection, s.smoothing)
    return s


for name, top, trajs, note, movie in COMPLEXES:
    if only and name not in only:
        continue
    if not os.path.exists(top):
        print(f"skip {name}: {top} not found")
        continue
    out = J("Figures", name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    system = MolSystem(top, trajs)
    last = system.n_frames - 1
    print(f"{name}: {system.summary()}")
    s = session_for(system, "Active-site close-up (publication)")
    s.render.title = f"{name.split('_')[0]} · GluA2 ({note})"
    s.render.slab = 26.0
    for frame, tag in ((0, "start"), (last, "end")) if last else ((0, "pose"),):
        s.frame = frame
        render_image(system, s, os.path.join(out, f"{name}_closeup_{tag}.png"), 3000, 2000, 300, frame, 2)
    ov = session_for(system, "Overview (chains + ligand)")
    ov.render.title = s.render.title
    render_image(system, ov, os.path.join(out, f"{name}_overview.png"), 3000, 2000, 300, last, 2)
    print(f"  stills done in {time.time() - t0:.0f}s")
    if movie and system.n_frames > 1:
        stride = max(1, system.n_frames // 900)
        s.frame = 0

        def prog(i, n):
            if i % 100 == 0 or i == n:
                print(f"  movie frame {i}/{n}", flush=True)
            return True
        render_movie(system, s, os.path.join(out, f"{name}_closeup.mp4"),
                     MovieOptions(width=1920, height=1080, fps=30, stride=stride, crf=16, hold_last=1.5), prog)
        print(f"  movie done in {time.time() - t0:.0f}s")
print("outputs in", J("Figures"))
