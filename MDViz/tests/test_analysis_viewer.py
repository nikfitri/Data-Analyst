"""Regressions for coordinate provenance, rendering updates and contact navigation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mdviz.system import MolSystem
from mdviz.interactions import Interaction, InteractionParams, OccupancyResult, occupancy


@pytest.fixture
def system(tmp_path):
    path = tmp_path / "test.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       0.000   4.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  ALA A   3       0.000   0.000   4.000  1.00  0.00           C\n"
        "HETATM    4  C1  LIG B   4       2.000   0.000   0.000  1.00  0.00           C\nEND\n")
    xyz = np.array([[[0, 0, 0], [0, 4, 0], [0, 0, 4], [x, 0, 0]] for x in (2, 8, 2)], np.float32)
    return MolSystem(str(path), coordinates=xyz)


class ThresholdDetector:
    p = InteractionParams()
    lig, rec = np.array([3]), np.array([0])

    def detect(self, pos):
        if np.linalg.norm(pos[3] - pos[0]) > 3.5:
            return []
        return [Interaction("Hydrophobic", "ALA1", 0, "C1", "CA", pos[3], pos[0], 2)]


def test_smoothing_never_changes_occupancy(system, tmp_path):
    a = occupancy(system, ThresholdDetector(), range(3))
    system.set_processing(smoothing=3)
    assert not np.allclose(system.positions(1), system.analysis_positions(1))
    b = occupancy(system, ThresholdDetector(), range(3))
    np.testing.assert_array_equal(a.residue_matrix, [[True, False, True]])
    np.testing.assert_array_equal(a.residue_matrix, b.residue_matrix)
    assert b.residue_occupancy()[0][1] == pytest.approx(200 / 3)
    b.to_csv(tmp_path / "occ.csv")
    metadata = json.loads((tmp_path / "occ.csv.metadata.json").read_text())
    assert metadata["smoothing"] == 1
    assert metadata["frames"] == [0, 1, 2]


def test_rmsf_is_unsmoothed_and_does_not_change_display(system):
    expected = np.sqrt(8)
    a = system.rmsf()
    assert a[3] == pytest.approx(expected, abs=1e-5)
    system.set_processing(smoothing=3)
    before = system.positions(1).copy()
    b = system.rmsf()
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_array_equal(before, system.positions(1))
    assert system.align is False
    assert system.smoothing == 3


def test_processing_invalidates_all_analysis_caches(system):
    system.rmsf()
    system._ss_cache[0] = "old"
    system._interacting_cache = {"old": {1}}
    system.set_processing(pbc_fix=True)
    assert system._rmsf is None
    assert not system._ss_cache
    assert not system._interacting_cache


def test_rmsf_sampling_and_failure_restore_state(system):
    samples = []
    system.rmsf(max_frames=2, progress=lambda i, n: samples.append((i, n)))
    assert samples == [(1, 2), (2, 2)]
    system.u.trajectory[1]
    def fail(i, n):
        raise RuntimeError("cancel")
    with pytest.raises(RuntimeError):
        system.rmsf(max_frames=3, progress=fail)
    assert system.u.trajectory.frame == 1
    assert not system.align
    system.set_processing(align_selection="name MISSING")
    with pytest.raises(ValueError, match="three atoms"):
        system.rmsf()


def test_dynamic_selection_does_not_overwrite_memory_trajectory(system):
    original = system.u.trajectory.coordinate_array.copy()
    system.set_processing(smoothing=3)
    system.select("around 5 ligand", 1)
    np.testing.assert_array_equal(system.u.trajectory.coordinate_array, original)


def test_cancelled_occupancy_marks_partial_and_empty_range_rejected(system):
    result = occupancy(system, ThresholdDetector(), range(3), lambda i, n: i < 1)
    assert result.frames == [0]
    assert result.metadata["partial"] is True
    with pytest.raises(ValueError, match="non-empty"):
        occupancy(system, ThresholdDetector(), [])


def test_label_layout_bounds_and_overlap():
    from mdviz.labels import layout_labels
    anchors = [(400, 250), (410, 250), (420, 250), (430, 250), (440, 250)]
    out = layout_labels(anchors, [80]*5, 800, 500, 20)
    for x, y in out:
        assert 0 <= x <= 720 and 0 <= y <= 480
    assert out == layout_labels(anchors, [80]*5, 800, 500, 20)
    for i, (x, y) in enumerate(out):
        for xx, yy in out[i+1:]:
            assert abs(x-xx) >= 80 or abs(y-yy) >= 20


def test_glyph_updates_reuse_objects_and_accept_selection_changes():
    from mdviz.scene import _glyph_actor, _sphere_source, _set_glyph_data
    from vtkmodules.util.numpy_support import vtk_to_numpy
    _, mapper = _glyph_actor(_sphere_source(8))
    colors = np.array([[255, 0, 0]], np.uint8)
    _set_glyph_data(mapper, [[0, 0, 0]], [1], colors)
    original = mapper.GetInput()
    _set_glyph_data(mapper, [[2, 3, 4]], [2], colors)
    assert mapper.GetInput() is original
    np.testing.assert_array_equal(vtk_to_numpy(original.GetPoints().GetData()), [[2, 3, 4]])
    _set_glyph_data(mapper, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), np.uint8))
    assert mapper.GetInput().GetNumberOfPoints() == 0


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_contact_timeline_uses_real_frame_ids(app):
    from mdviz.gui.contact_timeline import ContactTimeline
    timeline = ContactTimeline()
    result = OccupancyResult([10, 20, 50], [100, 200, 500], [("H-bond", "ALA1")],
                             np.array([[True, False, True]]), [], {})
    timeline.set_result(result)
    selected = []
    timeline.frameRequested.connect(selected.append)
    timeline._click(SimpleNamespace(inaxes=timeline.ax, xdata=1.8, button=1))
    assert selected == [50]
    timeline.set_frame(20)
    assert list(timeline.marker.get_xdata()) == [1, 1]
    assert timeline.marker.get_linestyle() == "-"
    timeline.set_frame(21)
    assert timeline.marker.get_linestyle() == "--"
    timeline.clear()
    assert timeline.result is None
    timeline.close()


def test_worker_owns_reader_and_keeps_gui_frame(app, system):
    from PySide6.QtCore import QEventLoop, QTimer
    from mdviz.gui.analysis_worker import OccupancyWorker
    from mdviz.interactions import InteractionDetector
    detector = InteractionDetector(system, [3], [0, 1, 2])
    system.u.trajectory[1]
    xyz = system.u.trajectory.coordinate_array.copy()
    job = OccupancyWorker(system, detector, [0, 1, 2])
    results, errors = [], []
    job.resultReady.connect(results.append)
    job.failed.connect(errors.append)
    loop = QEventLoop()
    job.finished.connect(loop.quit)
    QTimer.singleShot(10000, loop.quit)
    job.start()
    loop.exec()
    assert job.wait(1000)
    assert not errors
    assert len(results) == 1
    assert results[0].frames == [0, 1, 2]
    assert system.u.trajectory.frame == 1
    np.testing.assert_array_equal(system.u.trajectory.coordinate_array, xyz)


def test_scene_measurements_surface_cache_and_pinned_labels(system):
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow
    from mdviz.scene import Scene
    from mdviz.state import Session, Representation
    win, ren = vtkRenderWindow(), vtkRenderer()
    win.SetOffScreenRendering(1)
    win.SetSize(800, 600)
    win.AddRenderer(ren)
    session = Session(reps=[Representation("Ligand", "ligand", "surface")], measurements=[[0, 3]])
    session.interactions.show = False
    system.set_processing(smoothing=3)
    scene = Scene(ren, system, session)
    try:
        scene.rebuild()
        scene.set_frame(1)
        assert scene._label_actors[0].GetInput() == "8.00 Å"
        mesh = scene.visuals[0].mesh_mapper.GetInput()
        scene.set_frame(1)
        assert scene.visuals[0].mesh_mapper.GetInput() is mesh
        scene._callouts = [("ALA1", [0, 0, 0])]
        scene._callout_keys = ["0"]
        scene.move_label("0", 200, 100)
        scene._layout_callouts()
        assert scene.label_at(210, 105) == "0"
        assert scene._callout_actors[0].GetPosition() == (200, 100)
        restored = Session.from_dict(json.loads(session.to_json()))
        assert restored.interactions.label_positions["0"] == pytest.approx([0.25, 1/6])
        scene.set_playback_quality(True)
        assert scene.quality["surface_spacing"] > Scene.QUALITY["screen"]["surface_spacing"]
        scene.set_playback_quality(False)
        assert scene.quality is Scene.QUALITY["screen"]
    finally:
        scene.dispose()
        win.Finalize()


def test_occupancy_dialog_background_completion_and_close(app, system):
    from PySide6.QtCore import QEventLoop, QTimer
    from mdviz.gui.dialogs import OccupancyDialog
    from mdviz.interactions import InteractionDetector
    detector = InteractionDetector(system, [3], [0, 1, 2])
    dialog = OccupancyDialog(None, system, detector)
    loop = QEventLoop()
    dialog.run()
    dialog.worker.finished.connect(loop.quit)
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    assert dialog.worker is None
    assert dialog.result.frames == [0, 1, 2]
    assert dialog.timeline_button.isEnabled()
    assert "unsmoothed" in dialog.summary.text()
    dialog.run()
    dialog.worker.finished.connect(loop.quit)
    dialog.reject()
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    assert dialog.worker is None
    dialog.close()


def test_worker_failure_is_reported(app, system):
    from PySide6.QtCore import QEventLoop, QTimer
    from mdviz.gui.analysis_worker import OccupancyWorker
    from mdviz.interactions import InteractionDetector
    detector = InteractionDetector(system, [3], [0, 1, 2])
    worker = OccupancyWorker(system, detector, [0])
    worker.topology += ".missing"
    errors = []
    worker.failed.connect(errors.append)
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(10000, loop.quit)
    worker.start()
    loop.exec()
    assert worker.wait(1000)
    assert errors


def test_pocket_sampling_is_explicit_and_independent_of_viewed_frame(system):
    from mdviz.scene import Scene
    from mdviz.state import Session
    scene = Scene.__new__(Scene)
    scene.sys, scene.sess, scene.frame = system, Session(), 1
    scene.sess.interactions.label_mode = "pocket"
    scene.sess.interactions.pocket_samples = 2
    contact = Interaction("Hydrophobic", "ALA1", 0, "C1", "CA", np.zeros(3), np.ones(3), 2,
                          lig_atoms=(3,), rec_atoms=(0,))
    scene.detector = SimpleNamespace(detect=lambda p: [contact] if p[3, 0] > 4 else [])
    scene.compute_interacting()
    assert not system.interacting_resindices  # preview samples endpoints, never adds viewed frame
    scene.sess.interactions.pocket_samples = 0
    scene.compute_interacting()
    assert system.interacting_resindices == {0}
    assert 0 in system.contact_atom_indices
    assert scene.pocket_kinds == {"Hydrophobic"}
