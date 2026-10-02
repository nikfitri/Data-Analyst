"""Entry point.

GUI:      python -m mdviz [structure] [trajectory ...]
          python -m mdviz session.mdviz.json
Batch:    python -m mdviz --session s.json --image fig.png [--frame 100 --size 3000x2000 --dpi 300]
          python -m mdviz --session s.json --movie out.mp4 [--fps 30 --size 1920x1080 --stride 2]
          python -m mdviz --session s.json --occupancy table.csv [--plots]
Re-use one session's look for another run with --top / --traj (e.g. every ligand in GROMACS/runs).
PES:      python -m mdviz --pes-scan ala2.npz [--pes-step 10 --pes-ff amber99sb-ildn --pes-workdir DIR]
          python -m mdviz --pes-figure ala2.npz [--pes-cap 60]      (2D map + 3D surface PNGs, stationary points)
          python -m mdviz --pes-ligand lig.npz --top md.tpr --traj md.xtc [--pes-torsions "4 6 7 8" "21 23 24 25"]
                   (ligand torsion PES: the two most mobile torsions in the MD run, with MD frames overlaid)
"""
from __future__ import annotations

import argparse
import os
import sys


def _size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def _headless(args):
    from .state import Session, default_reps
    from .system import MolSystem

    sess = Session.load(args.session) if args.session else Session()
    top = args.top or sess.topology
    trajs = args.traj if args.traj is not None else sess.trajectories
    if not top:
        sys.exit("No structure given (use --session or --top).")
    system = MolSystem(top, trajs)
    if args.pbc:
        sess.pbc_fix = True
    if args.align:
        sess.align = True
    if system.n_frames > 1 and not sess.pbc_fix and system.detect_pbc_problems()["broken"]:
        print("molecules are split across the periodic box: fixing periodic boundaries and fitting frames")
        sess.pbc_fix = True
        sess.align = sess.align or bool(system.is_protein.any())
    if args.style:
        from .state import STYLE_PRESETS, apply_style
        name = next((n for n in STYLE_PRESETS if n.lower().startswith(args.style.lower())), None)
        if name is None:
            sys.exit(f"unknown --style {args.style}; choose from: {', '.join(STYLE_PRESETS)}")
        apply_style(sess, system, name)
    system.set_processing(sess.pbc_fix, sess.align, sess.align_selection, sess.smoothing)
    if not sess.reps:
        sess.reps = default_reps(system)
    if args.title is not None:
        sess.render.title = args.title
    if args.frame is not None:
        sess.frame = args.frame if args.frame >= 0 else system.n_frames + args.frame
    print(system.summary())

    if args.image:
        from .render import render_image
        w, h = _size(args.size or "3000x2000")
        render_image(system, sess, args.image, w, h, args.dpi, sess.frame, args.supersample, args.transparent)
        print("wrote", args.image)
    if args.movie:
        from .render import MovieOptions, render_movie
        w, h = _size(args.size or "1920x1080")
        opt = MovieOptions(width=w, height=h, fps=args.fps, start=args.start, stop=args.stop, stride=args.stride,
                           mode="turntable" if args.turntable else "trajectory", spin_degrees=args.spin,
                           turntable_seconds=args.turntable or 12.0, crf=args.crf, supersample=args.supersample)

        def prog(i, n):
            if i % 10 == 0 or i == n:
                print(f"\r  frame {i}/{n}", end="", flush=True)
            return True
        render_movie(system, sess, args.movie, opt, prog)
        print("\nwrote", args.movie)
    if args.occupancy:
        from .interactions import InteractionDetector, InteractionParams, occupancy
        st = sess.interactions
        det = InteractionDetector(system, system.select(st.ligand), system.select(st.receptor),
                                  InteractionParams.from_dict(st.params))
        stop = system.n_frames - 1 if args.stop < 0 else args.stop
        res = occupancy(system, det, range(args.start, stop + 1, args.stride))
        res.to_csv(args.occupancy)
        print("wrote", args.occupancy)
        if args.plots:
            from .plots import plot_occupancy, plot_timeline
            base = os.path.splitext(args.occupancy)[0]
            name = os.path.basename(os.path.dirname(system.topology))
            plot_occupancy(res, base + "_occupancy.png", title=name)
            plot_timeline(res, base + "_timeline.png", title=name)
            print("wrote", base + "_occupancy.png", base + "_timeline.png")


def _pes(args):
    from .pes import PESConfig, PESResult, PESScanner, find_stationary_points, md_strain_summary
    if args.pes_ligand:
        from .ligand_pes import attach_md_samples, setup_from_md
        from .system import MolSystem
        if not args.top:
            sys.exit("--pes-ligand needs --top (and --traj) of the MD run")
        system = MolSystem(args.top, args.traj or [])
        setup = setup_from_md(system)
        print(f"ligand topology: {setup.top}")
        for t in setup.torsions:
            print(f"  {t.label:<18s} atoms {' '.join(str(i + 1) for i in t.quad):<14s} MD spread {t.spread:6.1f} deg")
        if args.pes_torsions:
            quads = [[int(x) - 1 for x in q.split()] for q in args.pes_torsions]
        else:
            quads = [list(setup.torsions[0].quad), list(setup.torsions[1].quad)]
        cfg = PESConfig(molecule="custom", custom_gro=setup.gro, custom_top=setup.top, phi_atoms=quads[0],
                        psi_atoms=quads[1], step=args.pes_step, restraint_k=args.pes_k, epsilon_r=args.pes_eps,
                        workdir=args.pes_workdir or os.path.splitext(os.path.abspath(args.pes_ligand))[0] + "_work")
        res = PESScanner(cfg).prepare(log=print).run()
        attach_md_samples(res, setup, system)
        res.save(args.pes_ligand)
        st = md_strain_summary(res)
        print(f"wrote {args.pes_ligand}  ({res.meta['seconds']} s, {res.meta['failed']} failed)")
        print(f"MD frames sit {st['mean']:.1f} kJ/mol (median {st['median']:.1f}) above the surface minimum; "
              f"{st['within_5']:.0f}% within 5 kJ/mol")
        args.pes_figure = args.pes_figure or args.pes_ligand
    if args.pes_scan:
        cfg = PESConfig(step=args.pes_step, forcefield=args.pes_ff, restraint_k=args.pes_k, epsilon_r=args.pes_eps,
                        workdir=args.pes_workdir or os.path.splitext(os.path.abspath(args.pes_scan))[0] + "_work")

        def prog(done, total, *_):
            if done % 50 == 0 or done == total:
                print(f"\r  {done}/{total} minimisations", end="", flush=True)
        res = PESScanner(cfg).prepare(log=print).run(progress=prog)
        res.save(args.pes_scan)
        print(f"\nwrote {args.pes_scan}  ({res.meta['seconds']} s, {res.meta['failed']} failed)")
        args.pes_figure = args.pes_figure or args.pes_scan
    if args.pes_figure:
        from .plots import plot_pes_map, plot_pes_surface
        res = PESResult.load(args.pes_figure)
        pts = find_stationary_points(res)
        for p in pts:
            xl, yl = (lab.split(" (")[0] for lab in res.labels)
            print(f"  {p['label']:<4s} {p['type']:<8s} {xl} {p['phi']:7.1f}  {yl} {p['psi']:7.1f}  "
                  f"dE {p['rel']:6.2f} kJ/mol  {p['region']}")
        base = os.path.splitext(args.pes_figure)[0]
        plot_pes_map(res, pts, base + "_map.png", cap=args.pes_cap)
        plot_pes_surface(res, pts, base + "_surface.png", cap=args.pes_cap)
        res.to_csv(base + ".csv")
        print("wrote", base + "_map.png", base + "_surface.png", base + ".csv")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # never crash on Greek letters / arrows in a cp1252 console
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="mdviz", description="MD trajectory visualisation and publication rendering")
    ap.add_argument("files", nargs="*", help="structure [trajectories...] or a session .json (GUI mode)")
    ap.add_argument("--session", help="session file (.json) saved from the GUI")
    ap.add_argument("--top", help="override structure/topology")
    ap.add_argument("--traj", nargs="*", help="override trajectories")
    ap.add_argument("--image", help="render a still image (png/jpg/tif) and exit")
    ap.add_argument("--movie", help="render an MP4 movie and exit")
    ap.add_argument("--occupancy", help="write interaction occupancy CSV and exit")
    ap.add_argument("--plots", action="store_true", help="with --occupancy: also save PNG plots")
    ap.add_argument("--frame", type=int, help="frame for --image (negative counts from the end)")
    ap.add_argument("--size", help="WIDTHxHEIGHT in pixels")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--title")
    ap.add_argument("--style", help="style preset: default | active-site | overview")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--spin", type=float, default=0.0, help="camera spin in degrees over the movie")
    ap.add_argument("--turntable", type=float, help="turntable movie of the chosen frame, duration in seconds")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--pbc", action="store_true", help="fix periodic boundaries")
    ap.add_argument("--align", action="store_true", help="fit frames to frame 0")
    ap.add_argument("--pes-scan", help="run a phi/psi PES scan of alanine dipeptide in GROMACS, save .npz")
    ap.add_argument("--pes-step", type=float, default=10.0, help="PES grid spacing in degrees")
    ap.add_argument("--pes-ff", default="amber99sb-ildn", help="GROMACS force field for the PES scan")
    ap.add_argument("--pes-k", type=float, default=10000.0, help="dihedral restraint force constant (kJ/mol/rad^2)")
    ap.add_argument("--pes-workdir", help="PES scan work folder (re-use to resume)")
    ap.add_argument("--pes-figure", help="make publication PES figures from a .npz/.csv surface")
    ap.add_argument("--pes-ligand", help="ligand torsion PES from the MD run given by --top/--traj, save .npz")
    ap.add_argument("--pes-torsions", nargs=2, help='two torsions as 1-based ligand atom numbers, e.g. "4 6 7 8" "21 23 24 25"')
    ap.add_argument("--pes-eps", type=float, default=1.0, help="dielectric constant for the scan (1 = vacuum)")
    ap.add_argument("--pes-cap", type=float, default=60.0, help="upper energy limit of PES figures (kJ/mol)")
    args = ap.parse_args(argv)

    if args.pes_scan or args.pes_figure or args.pes_ligand:
        return _pes(args)

    if args.image or args.movie or args.occupancy:
        if args.supersample == 2 and args.movie and not args.image:
            args.supersample = 1
        return _headless(args)

    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication
    import PySide6.QtWebEngineWidgets  # noqa: F401  (must be imported before QApplication exists)
    from .gui.main_window import MainWindow

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)  # VTK and Qt WebEngine in one window
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MDViz")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    files = args.files
    if args.session:
        files = [args.session]
    if files:
        from .state import Session
        if files[0].lower().endswith(".json"):
            win._guard(win._load_session_file, files[0])
        else:
            win.last_dir = os.path.dirname(os.path.abspath(files[0]))
            win._guard(win.load_system, files[0], files[1:], Session(), True)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
