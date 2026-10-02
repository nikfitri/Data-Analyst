"""Publication plots for interaction occupancy (matplotlib, static PNG/SVG/PDF/TIFF)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .chem import INTERACTION_COLORS  # noqa: E402

INK = "#1f1f1f"
INK_2 = "#5c5c5c"
GRID = "#e4e4e2"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelcolor=INK, width=0.8, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _pretty(res: str) -> str:
    return res[:1] + res[1:3].lower() + res[3:]


def _legend(ax, kinds, **kw):
    handles = [Patch(facecolor=INTERACTION_COLORS[k], edgecolor="none", label=k) for k in kinds]
    leg = ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK, **kw)
    return leg


def plot_occupancy(result, path, min_occupancy=5.0, max_rows=30, title="", dpi=300):
    """Horizontal bar chart: occupancy (% of frames) per residue and interaction type."""
    rows = [(k, o) for k, o in result.residue_occupancy() if o >= min_occupancy][:max_rows]
    rows = rows[::-1]  # largest at top
    n = max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(6.0, 0.8 + 0.26 * n), dpi=dpi)
    _style(ax)
    if rows:
        y = np.arange(len(rows))
        occ = [o for _, o in rows]
        cols = [INTERACTION_COLORS[k] for (k, _), _ in rows]
        ax.barh(y, occ, height=0.72, color=cols, edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{_pretty(res)}  ·  {kind}" for (kind, res), _ in rows])
        kinds = [k for k in INTERACTION_COLORS if any(kk == k for (kk, _), _ in rows)]
        _legend(ax, kinds, loc="lower right")
    else:
        ax.text(0.5, 0.5, f"No interactions above {min_occupancy:.0f}% occupancy", ha="center", va="center",
                transform=ax.transAxes, color=INK_2)
        ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Occupancy (% of analysed frames)")
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_timeline(result, path, min_occupancy=5.0, max_rows=30, title="", dpi=300):
    """Barcode plot: presence of each residue interaction over simulation time."""
    occ = dict(result.residue_occupancy())
    keys = [k for k in result.residue_keys if occ.get(k, 0) >= min_occupancy]
    keys = sorted(keys, key=lambda k: -occ[k])[:max_rows]
    n = max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(7.2, 0.8 + 0.26 * n), dpi=dpi)
    _style(ax)
    t = np.asarray(result.times_ps) / 1000.0
    if keys and len(t):
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
        kidx = {k: i for i, k in enumerate(result.residue_keys)}
        for row, key in enumerate(keys[::-1]):
            present = result.residue_matrix[kidx[key]]
            # merge consecutive frames into runs so the plot stays light
            runs, start = [], None
            for j, p in enumerate(present):
                if p and start is None:
                    start = j
                if (not p or j == len(present) - 1) and start is not None:
                    end = j if p else j - 1
                    runs.append((t[start] - dt / 2, (t[end] - t[start]) + dt))
                    start = None
            ax.broken_barh(runs, (row - 0.36, 0.72), facecolors=INTERACTION_COLORS[key[0]], zorder=3)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f"{_pretty(res)}  ·  {kind}  ({occ[(kind, res)]:.0f}%)" for kind, res in keys[::-1]])
        ax.set_xlim(t[0] - dt / 2, t[-1] + dt / 2)
        kinds = [k for k in INTERACTION_COLORS if any(kk == k for kk, _ in keys)]
        _legend(ax, kinds, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    else:
        ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Time (ns)")
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------- PES figures
PES_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _pes_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pes_blue", PES_RAMP)


def _periodic_closed(res, cap):
    """Grid extended to +180 (periodic copy) and clipped at `cap` for plotting."""
    rel = np.clip(res.rel, 0, cap)
    phi = np.append(res.phi, res.phi[0] + 360)
    psi = np.append(res.psi, res.psi[0] + 360)
    Z = np.vstack([rel, rel[:1]])
    Z = np.hstack([Z, Z[:, :1]])
    return phi, psi, Z


def _stat_markers(ax, points, three_d=False, zfun=None):
    from matplotlib.lines import Line2D
    for p in points:
        is_min = p["type"] == "minimum"
        kw = dict(marker="o" if is_min else "X", s=46 if is_min else 52, color=INK, edgecolors="white",
                  linewidths=1.2, zorder=6)
        if three_d:
            ax.scatter([p["phi"]], [p["psi"]], [zfun(p)], depthshade=False, **kw)
            if is_min:  # saddles are named on the 2D map; in 3D their labels collide in projection
                ax.text(p["phi"], p["psi"], zfun(p) + 2.5, p["label"], color=INK, fontsize=8, fontweight="bold", zorder=7)
        else:
            ax.scatter([p["phi"]], [p["psi"]], **kw)
            ax.annotate(p["label"], (p["phi"], p["psi"]), xytext=(5, 5), textcoords="offset points", color=INK,
                        fontsize=8.5, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    return [Line2D([], [], marker="o", color=INK, lw=0, markeredgecolor="white", label="Minimum (M)"),
            Line2D([], [], marker="X", color=INK, lw=0, markeredgecolor="white", markersize=8,
                   label="Saddle point (TS)")]


def plot_pes_map(res, points, path, cap=60.0, step=None, title="", dpi=300):
    """2D contour map of dE(phi, psi) with minima and saddle points annotated."""
    phi, psi, Z = _periodic_closed(res, cap)
    step = step or (2.0 if cap <= 30 else 4.0 if cap <= 60 else 8.0)
    levels = np.arange(0, cap + step, step)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=dpi)
    _style(ax)
    cf = ax.contourf(phi, psi, Z.T, levels=levels, cmap=_pes_cmap(), extend="max")
    ax.contour(phi, psi, Z.T, levels=levels, colors="white", linewidths=0.5, alpha=0.7)
    handles = _stat_markers(ax, points)
    xl, yl = getattr(res, "labels", ("φ", "ψ"))
    md = getattr(res, "md_samples", None)
    if md is not None and len(md):
        from matplotlib.lines import Line2D
        ax.scatter(md[:, 0], md[:, 1], s=3, color=INK, alpha=0.25, linewidths=0, zorder=4, rasterized=True)
        handles = handles + [Line2D([], [], marker="o", color=INK, alpha=0.5, lw=0, markersize=3,
                                    label=f"MD frames ({len(md)})")]
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    ax.set_aspect("equal")
    ax.set_xlabel(f"{xl} (°)")
    ax.set_ylabel(f"{yl} (°)")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ΔE (kJ/mol)", color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    if points or (md is not None and len(md)):
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
                  fontsize=8.5, labelcolor=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pes_surface(res, points, path, cap=60.0, title="", dpi=300, elev=38, azim=-58):
    """3D surface of dE(phi, psi), coloured by energy, with stationary points marked."""
    phi, psi, Z = _periodic_closed(res, cap)
    P, Q = np.meshgrid(phi, psi, indexing="ij")
    fig = plt.figure(figsize=(6.0, 4.8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False  # keep stationary-point markers on top of the surface
    ax.plot_surface(P, Q, Z, cmap=_pes_cmap(), vmin=0, vmax=cap, rstride=1, cstride=1, linewidth=0.15,
                    edgecolor=(1, 1, 1, 0.35), antialiased=True, zorder=1)
    _stat_markers(ax, points, three_d=True, zfun=lambda p: min(p["rel"], cap) + 1.0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_zlim(0, cap)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-90, 181, 90))
    xl, yl = getattr(res, "labels", ("φ", "ψ"))
    ax.set_xlabel(f"{xl} (°)", color=INK)
    ax.set_ylabel(f"{yl} (°)", color=INK)
    ax.set_zlabel("ΔE (kJ/mol)", color=INK)
    ax.tick_params(colors=INK_2, labelsize=8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
