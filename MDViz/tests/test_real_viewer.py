"""Opt-in integration checks using the local 52E5 trajectory and actual VTK exports.

Run with MDVIZ_VIEWER_TESTS=1. Artifacts remain in MDViz/validation/.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MDVIZ_VIEWER_TESTS") != "1",
                                reason="set MDVIZ_VIEWER_TESTS=1 for local trajectory/render checks")


def test_real_trajectory_and_exports():
    from mdviz.system import MolSystem
    from mdviz.state import Session, apply_style
    from mdviz.interactions import InteractionDetector, InteractionParams, occupancy
    from mdviz.render import OffscreenRenderer, render_movie, MovieOptions
    root = Path(__file__).resolve().parents[2]
    run = root / "GROMACS/runs/52E5"
    system = MolSystem(str(run / "md_10ns_52E5.tpr"), [str(run / "md_10ns_52E5.xtc")])
    out = root / "MDViz/validation"
    out.mkdir(exist_ok=True)
    session = Session(topology=system.topology, trajectories=system.trajectories, pbc_fix=True, align=True)
    apply_style(session, system, "Active-site close-up (publication)")
    system.set_processing(True, True, smoothing=1)
    st = session.interactions
    detector = InteractionDetector(system, system.select(st.ligand), system.select(st.receptor),
                                   InteractionParams.from_dict(st.params))
    frames = np.linspace(0, system.n_frames - 1, 12, dtype=int).tolist()
    first = occupancy(system, detector, frames)
    system.set_processing(smoothing=5)
    smooth = occupancy(system, detector, frames)
    assert first.residue_keys == smooth.residue_keys
    np.testing.assert_array_equal(first.residue_matrix, smooth.residue_matrix)
    first.to_csv(out / "52E5_validation_occupancy.csv")
    first.timeline_csv(out / "52E5_validation_timeline.csv")
    session.smoothing = 5
    session.render.title = "52E5 · updated pocket view"
    timings = []
    renderer = OffscreenRenderer(system, session, 1500, 1000, supersample=1)
    try:
        view = renderer.scene.capture_view()
        for frame, filename in [(0, "52E5_updated_start.png"), (system.n_frames-1, "52E5_updated_end.png")]:
            start = time.perf_counter()
            image = renderer.grab(frame)
            timings.append(time.perf_counter() - start)
            pixels = np.asarray(image)
            assert pixels.std() > 10
            assert ((pixels[..., :3] < 230).any(axis=2)).mean() > 0.01
            image.save(out / filename)
            assert renderer.scene._callouts
            active = [a for a in renderer.scene._callout_actors[::2] if a.GetVisibility()]
            assert len(active) == len(renderer.scene._callouts)
            for actor in active:
                x, y = actor.GetPosition()
                assert 0 <= x < 1500 and 0 <= y < 1000
            expected = detector.detect(system.analysis_positions(frame))
            assert {i.key for i in expected} == {i.key for i in renderer.scene.current_interactions}
        renderer.scene.ren.GetActiveCamera().Azimuth(60)
        renderer.grab(0).save(out / "52E5_updated_rotated.png")
    finally:
        renderer.close()
    # Same camera, reconstructed old style; this is a style comparison, not an old-code benchmark.
    old = session.copy()
    old.view = view
    old.interactions.callout_labels = False
    old.render.label_box = False
    old.render.title = "52E5 · previous pocket styling"
    old.reps[0].uniform_color, old.reps[0].size, old.reps[0].material = "#8ca9f7", 0.8, "soft"
    old.reps[1].carbon_color, old.reps[1].size = "#8fb0f5", 0.72
    renderer = OffscreenRenderer(system, old, 1500, 1000, supersample=1)
    try:
        renderer.grab(0).save(out / "52E5_previous_style.png")
    finally:
        renderer.close()
    session.frame = 0
    session.save(str(out / "52E5_updated.mdviz.json"))
    movie = out / "52E5_validation.mp4"
    render_movie(system, session, str(movie), MovieOptions(width=640, height=420, fps=5,
                 start=0, stop=4, stride=1, supersample=1, hold_last=0))
    assert movie.stat().st_size > 1000
    import imageio_ffmpeg
    decoder = imageio_ffmpeg.read_frames(str(movie), pix_fmt="rgb24")
    video_metadata = next(decoder)
    decoded = list(decoder)
    assert len(decoded) == 5
    assert video_metadata["size"] == (640, 420)
    assert np.frombuffer(decoded[0], dtype=np.uint8).std() > 10
    report = {"atoms": system.n_atoms, "trajectory_frames": system.n_frames,
              "sampled_frames": frames, "occupancy_smoothing_invariant": True,
              "export_frame_seconds": timings, "movie_bytes": movie.stat().st_size,
              "decoded_movie_frames": len(decoded), "labels_visible": True}
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    from PySide6.QtWidgets import QApplication
    from mdviz.gui.contact_timeline import ContactTimeline
    app = QApplication.instance() or QApplication([])
    timeline = ContactTimeline()
    timeline.set_result(first)
    timeline.figure.savefig(out / "52E5_contact_timeline.png", dpi=160)
    timeline.close()
    app.processEvents()


def test_main_window_timeline_and_processing():
    from PySide6.QtWidgets import QApplication
    from mdviz.gui.main_window import MainWindow
    from mdviz.state import Session
    from mdviz.interactions import OccupancyResult
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    root = Path(__file__).resolve().parents[2]
    run = root / "GROMACS/runs/52E5"
    try:
        window.load_system(str(run / "md_10ns_52E5.tpr"), [str(run / "md_10ns_52E5.xtc")],
                           Session(pbc_fix=True, align=True))
        result = OccupancyResult([0, 2, 4], [0, 20, 40], [("H-bond", "SER105")],
                                 np.array([[True, False, True]]), [], {})
        window.contact_timeline.set_result(result)
        window.contact_timeline.frameRequested.emit(2)
        assert window.scene.frame == 2 and window.traj.frame() == 2
        window.traj.toggle_play()
        assert window.scene.quality is window.scene.QUALITY["playback"]
        window.traj.stop()
        assert window.scene.quality is window.scene.QUALITY["screen"]
        window._processing_changed()
        assert window.contact_timeline.result is None
        window.contact_timeline.set_result(result)
        window.vtk.resize(1100, 700)
        window.renwin.SetSize(1100, 700)
        window.apply_style("Active-site close-up (publication)")
        window._update_text_scale()
        assert window.contact_timeline.result is None
        assert window.scene._callout_boxes
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        key, x, y, _, _ = window.scene._callout_boxes[0]
        w, h = window.renwin.GetSize()
        def event(kind, xx, yy):
            return QMouseEvent(kind, QPointF(xx * window.vtk.width()/w,
                                            (h-yy)*window.vtk.height()/h),
                               Qt.LeftButton, Qt.LeftButton, Qt.AltModifier)
        assert window.eventFilter(window.vtk, event(QEvent.MouseButtonPress, x+2, y+2))
        assert window._drag_label == key
        assert window.eventFilter(window.vtk, event(QEvent.MouseMove, w*0.25, h*0.5))
        assert window.eventFilter(window.vtk, event(QEvent.MouseButtonRelease, w*0.25, h*0.5))
        assert window.session.interactions.label_positions[key] == pytest.approx([0.25, 0.5])
        assert window._drag_label is None
    finally:
        window.close()
        app.processEvents()
