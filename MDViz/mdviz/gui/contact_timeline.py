"""Contact heatmap linked to actual sampled frame IDs."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ContactTimeline(QWidget):
    frameRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.marker = None
        self.figure = Figure(figsize=(8, 2.5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.info = QLabel("Run trajectory occupancy analysis to populate the timeline.")
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._click)

    def clear(self):
        self.result = None
        self.marker = None
        self.figure.clear()
        self.info.setText("Run occupancy analysis again for the current system and settings.")
        self.canvas.draw_idle()

    def set_result(self, result):
        self.result = result
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        scores = result.residue_matrix.mean(axis=1) if len(result.residue_keys) else np.array([])
        order = np.argsort(-scores, kind="stable")[:30]
        if len(order):
            self.ax.imshow(result.residue_matrix[order], aspect="auto", interpolation="nearest",
                           cmap="Blues", vmin=0, vmax=1)
            self.ax.set_yticks(range(len(order)),
                              [f"{result.residue_keys[i][1]} · {result.residue_keys[i][0]} ({scores[i]*100:.0f}%)"
                               for i in order], fontsize=8)
        else:
            self.ax.set_xlim(-0.5, len(result.frames) - 0.5)
            self.ax.set_yticks([])
            self.ax.text(0.5, 0.5, "No contacts detected", ha="center", transform=self.ax.transAxes)
        ticks = np.unique(np.linspace(0, len(result.frames)-1, min(8, len(result.frames)), dtype=int))
        self.ax.set_xticks(ticks, [f"{result.times_ps[i]/1000:.3g}\n#{result.frames[i]}" for i in ticks])
        self.ax.set_xlabel("Sampled time (ns) / frame — click a column to inspect it")
        self.marker = self.ax.axvline(0, color="#da6a22", linewidth=1.5)
        partial = "Partial result · " if result.metadata.get("partial") else ""
        self.info.setText(f"{partial}{len(result.frames)} frames · top {len(order)} of {len(scores)} contacts · "
                          "unsmoothed coordinates; columns are samples, not continuous time")
        self.canvas.draw_idle()

    def _click(self, event):
        if self.result is None or event.inaxes is not self.ax or event.xdata is None or event.button != 1:
            return
        i = int(np.clip(np.floor(event.xdata + 0.5), 0, len(self.result.frames)-1))
        self.frameRequested.emit(int(self.result.frames[i]))

    def set_frame(self, frame):
        if self.result is None or self.marker is None:
            return
        frames = np.asarray(self.result.frames)
        index = int(np.argmin(np.abs(frames - frame)))
        self.marker.set_xdata([index, index])
        self.marker.set_linestyle("-" if frames[index] == frame else "--")
        self.canvas.draw_idle()
