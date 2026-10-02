"""Off-screen publication rendering: still images (PNG/JPEG/TIFF) and H.264 MP4 movies."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image

import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter

from .scene import Scene
from .state import Session, ViewState

MAX_RENDER_DIM = 8192
# Ambient occlusion needs several full-size GPU buffers; beyond ~4k px integrated GPUs return black frames.
MAX_RENDER_DIM_SSAO = 4096


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
        return exe


class OffscreenRenderer:
    """Owns an off-screen VTK window with its own Scene built from a session copy."""

    def __init__(self, system, session: Session, width: int, height: int, supersample: int = 2,
                 transparent: bool = False, reference_height: int | None = None):
        self.sys = system
        self.sess = session.copy()
        self.out_w, self.out_h = int(width), int(height)
        ss = max(1, int(supersample))
        limit = MAX_RENDER_DIM_SSAO if session.render.ssao else MAX_RENDER_DIM
        while ss > 1 and max(width, height) * ss > limit:
            ss -= 1
        self.ss = ss
        self.w, self.h = self.out_w * ss, self.out_h * ss
        self.transparent = transparent
        self.win = vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.SetMultiSamples(0)
        self.win.SetAlphaBitPlanes(1)
        self.win.SetSize(self.w, self.h)
        self.ren = vtkRenderer()
        self.win.AddRenderer(self.ren)
        if transparent:
            self.sess.render.background2 = ""
        self.scene = Scene(self.ren, system, self.sess, quality="export")
        # text/line sizes are defined for a 1080-px-high image
        self.scene.text_scale = self.h / 1080.0
        if transparent:
            self.ren.SetBackgroundAlpha(0.0)
        self.scene.frame = int(self.sess.frame)
        self.scene.rebuild()
        if self.sess.view.auto:
            self.scene.auto_view()
        else:
            self.scene.apply_view(self.sess.view)
        self.sess.view = self.scene.capture_view()

    def set_view(self, view: ViewState):
        self.scene.apply_view(view)

    def grab(self, frame: int | None = None) -> Image.Image:
        if frame is not None and frame != self.scene.frame:
            self.scene.set_frame(frame)
        self.ren.ResetCameraClippingRange()
        self.win.Render()
        f = vtkWindowToImageFilter()
        f.SetInput(self.win)
        f.SetInputBufferTypeToRGBA() if self.transparent else f.SetInputBufferTypeToRGB()
        f.ReadFrontBufferOff()
        f.ShouldRerenderOff()
        f.Update()
        img = f.GetOutput()
        w, h, _ = img.GetDimensions()
        nc = 4 if self.transparent else 3
        arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, nc)[::-1]
        im = Image.fromarray(np.ascontiguousarray(arr), "RGBA" if self.transparent else "RGB")
        if self.ss > 1:
            im = im.resize((self.out_w, self.out_h), Image.LANCZOS)
        return im

    def close(self):
        self.scene.dispose()
        self.win.Finalize()


def is_blank(im: Image.Image) -> bool:
    """True when the GPU returned an empty (single-colour) frame."""
    a = np.asarray(im.convert("RGB"))
    return bool(a.max() == a.min())


def safe_renderer(system, session: Session, width, height, supersample=2, transparent=False, frame=None):
    """OffscreenRenderer whose first frame is verified; falls back to lighter settings if the GPU gives up.

    Returns (renderer, first image).
    """
    attempts = [(supersample, session.render.ssao), (1, session.render.ssao), (1, False)]
    for ss, ssao in attempts:
        sess = session.copy()
        sess.render.ssao = ssao
        r = OffscreenRenderer(system, sess, width, height, ss, transparent)
        im = r.grab(sess.frame if frame is None else frame)
        if not is_blank(im):
            return r, im
        r.close()
    raise RuntimeError("The graphics card returned an empty image even at reduced settings; "
                       "try a smaller image size.")


def save_image(im: Image.Image, path: str, dpi: int = 300, quality: int = 95):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        if im.mode == "RGBA":  # JPEG has no alpha: composite on white
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            im = bg
        im.save(path, quality=quality, subsampling=0, dpi=(dpi, dpi), optimize=True)
    elif ext in (".tif", ".tiff"):
        im.save(path, dpi=(dpi, dpi), compression="tiff_lzw")
    else:
        im.save(path, dpi=(dpi, dpi), optimize=True)


def render_image(system, session: Session, path: str, width=3000, height=2000, dpi=300, frame=None,
                 supersample=2, transparent=False, quality=95):
    r, im = safe_renderer(system, session, width, height, supersample, transparent, frame)
    try:
        save_image(im, path, dpi, quality)
    finally:
        r.close()
    return path


@dataclass
class MovieOptions:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    start: int = 0
    stop: int = -1                 # inclusive; -1 = last frame
    stride: int = 1
    mode: str = "trajectory"       # trajectory | turntable
    spin_degrees: float = 0.0      # extra rotation about the vertical axis spread over the movie
    turntable_seconds: float = 12.0
    view_from: str = ""            # saved view names for a camera fly-through
    view_to: str = ""
    crf: int = 16                  # lower = higher quality (0 = lossless)
    supersample: int = 1
    hold_last: float = 0.0         # seconds to hold the final frame
    codec: str = "libx264"


def _interp_view(a: ViewState, b: ViewState, t: float) -> ViewState:
    t = 0.5 - 0.5 * np.cos(np.pi * t)  # ease in/out
    lerp = lambda x, y: list((1 - t) * np.asarray(x) + t * np.asarray(y))
    up = np.asarray(lerp(a.view_up, b.view_up))
    up /= max(np.linalg.norm(up), 1e-6)
    return ViewState(lerp(a.position, b.position), lerp(a.focal_point, b.focal_point), list(up),
                     (1 - t) * a.view_angle + t * b.view_angle, a.parallel,
                     (1 - t) * a.parallel_scale + t * b.parallel_scale)


def movie_frames(system, opt: MovieOptions, current_frame: int) -> list[int]:
    if opt.mode == "turntable":
        n = max(2, int(round(opt.turntable_seconds * opt.fps)))
        return [current_frame] * n
    stop = system.n_frames - 1 if opt.stop < 0 else min(opt.stop, system.n_frames - 1)
    return list(range(max(0, opt.start), stop + 1, max(1, opt.stride)))


def render_movie(system, session: Session, path: str, opt: MovieOptions, progress=None) -> str:
    """Render an MP4. `progress(i, n)` may return False to cancel."""
    w, h = opt.width - opt.width % 2, opt.height - opt.height % 2
    frames = movie_frames(system, opt, session.frame)
    n = len(frames)
    r, _ = safe_renderer(system, session, w, h, opt.supersample, False, frames[0])
    base_view = r.sess.view
    va = session.saved_views.get(opt.view_from)
    vb = session.saved_views.get(opt.view_to)
    va = ViewState(**va) if isinstance(va, dict) else va
    vb = ViewState(**vb) if isinstance(vb, dict) else vb
    total_spin = 360.0 if opt.mode == "turntable" and not opt.spin_degrees else opt.spin_degrees
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(opt.fps), "-i", "-", "-an", "-c:v", opt.codec]
    if opt.codec == "libx264":
        cmd += ["-preset", "slow", "-crf", str(opt.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-profile:v", "high"]
    cmd += [path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    cancelled = False
    try:
        last = None
        for k, f in enumerate(frames):
            t = k / max(n - 1, 1)
            view = _interp_view(va, vb, t) if (va and vb) else base_view
            r.set_view(view)
            if total_spin:
                step = total_spin * (k / n if opt.mode == "turntable" else t)
                r.ren.GetActiveCamera().Azimuth(step)
            im = r.grab(f)
            last = im.tobytes()
            proc.stdin.write(last)
            if progress and progress(k + 1, n) is False:
                cancelled = True
                break
        if last and opt.hold_last > 0 and not cancelled:
            for _ in range(int(opt.hold_last * opt.fps)):
                proc.stdin.write(last)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        r.close()
    if cancelled:
        try:
            os.remove(path)
        except OSError:
            pass
        raise RuntimeError("Movie export cancelled")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:500]}")
    return path
