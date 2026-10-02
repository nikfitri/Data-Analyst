"""Occupancy jobs own their trajectory reader; never share GUI coordinates."""
import copy
from PySide6.QtCore import QThread, Signal
from ..interactions import InteractionDetector, InteractionParams, occupancy
from ..system import MolSystem


class OccupancyWorker(QThread):
    resultReady = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, system, detector, frames, parent=None):
        super().__init__(parent)
        self.topology, self.trajectories = system.topology, list(system.trajectories)
        self.processing = (system.pbc_fix, system.align, system.align_selection)
        self.lig, self.rec = detector.lig.copy(), detector.rec.copy()
        self.params = copy.deepcopy(detector.p.to_dict())
        self.frames = list(frames)
        self.ref_frame = detector.ref_frame
        from MDAnalysis.coordinates.memory import MemoryReader
        trj = system.u.trajectory
        self.memory = ((trj.coordinate_array.copy(), trj.dimensions_array.copy(), trj.dt)
                       if isinstance(trj, MemoryReader) else None)

    def run(self):
        system = None
        try:
            if self.isInterruptionRequested():
                return
            system = MolSystem(self.topology, self.trajectories,
                               coordinates=self.memory[0] if self.memory else None)
            if self.memory:
                from MDAnalysis.coordinates.memory import MemoryReader
                xyz, box, dt = self.memory
                system.u.load_new(xyz, format=MemoryReader, dimensions=box, dt=dt)
            system.set_processing(*self.processing, smoothing=1)
            detector = InteractionDetector(system, self.lig, self.rec,
                                           InteractionParams.from_dict(self.params), self.ref_frame)

            def progress(i, n):
                self.progress.emit(i, n)
                return not self.isInterruptionRequested()

            if self.isInterruptionRequested():
                return
            self.resultReady.emit(occupancy(system, detector, self.frames, progress))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if system is not None:
                system.u.trajectory.close()
