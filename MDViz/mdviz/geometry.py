"""Mesh generation for cartoon / tube backbones and molecular surfaces (numpy only)."""
from __future__ import annotations

import numpy as np


def _catmull_rom(p: np.ndarray, sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform Catmull-Rom spline through points p (n,3). Returns samples and their fractional residue index."""
    n = len(p)
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0, 1, sub, endpoint=False)
    t2, t3 = t * t, t * t * t
    b0 = -0.5 * t3 + t2 - 0.5 * t
    b1 = 1.5 * t3 - 2.5 * t2 + 1
    b2 = -1.5 * t3 + 2 * t2 + 0.5 * t
    b3 = 0.5 * t3 - 0.5 * t2
    pts, u = [], []
    for i in range(n - 1):
        P0, P1, P2, P3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        seg = b0[:, None] * P0 + b1[:, None] * P1 + b2[:, None] * P2 + b3[:, None] * P3
        pts.append(seg)
        u.append(i + t)
    pts.append(p[-1:])
    u.append(np.array([n - 1.0]))
    return np.vstack(pts), np.concatenate(u)


def _lerp_vectors(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    i0 = np.clip(np.floor(u).astype(int), 0, len(v) - 1)
    i1 = np.clip(i0 + 1, 0, len(v) - 1)
    f = (u - i0)[:, None]
    out = v[i0] * (1 - f) + v[i1] * f
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)


def backbone_segments(ca: np.ndarray, chain_ids, max_gap: float = 4.3):
    """Split residue list into continuous segments (by chain and CA-CA distance)."""
    segs, start = [], 0
    for i in range(1, len(ca) + 1):
        brk = i == len(ca) or chain_ids[i] != chain_ids[i - 1] or np.linalg.norm(ca[i] - ca[i - 1]) > max_gap
        if brk:
            if i - start >= 2:
                segs.append((start, i))
            start = i
    return segs


def cartoon_mesh(ca, o, ss, colors, chain_ids, style="cartoon", scale=1.0, sub=8, sides=12):
    """Build a ribbon/tube mesh.

    ca, o: (n,3) CA and carbonyl O positions per residue (o may contain NaN)
    ss: per-residue codes ('H','G','I' helix; 'E','B' strand; other coil)
    colors: (n,3) uint8 per residue
    Returns points, normals, triangles, point colours.
    """
    P, N, T, C = [], [], [], []
    offset = 0
    ss = np.asarray(ss)
    helix = np.isin(ss, ["H", "G", "I"])
    strand = np.isin(ss, ["E", "B"])
    if style == "tube":
        helix[:] = False
        strand[:] = False
    for s0, s1 in backbone_segments(ca, chain_ids):
        pca = ca[s0:s1].astype(np.float64).copy()
        n = len(pca)
        hx, st = helix[s0:s1], strand[s0:s1]
        # smooth strand zig-zag
        sm = pca.copy()
        for i in range(1, n - 1):
            if st[i]:
                sm[i] = 0.25 * pca[i - 1] + 0.5 * pca[i] + 0.25 * pca[i + 1]
        pts, u = _catmull_rom(sm, sub)
        m = len(pts)
        # tangents
        tan = np.gradient(pts, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
        # guide vectors: CA->O, sign-continuous; fall back to curvature-based
        guide = o[s0:s1] - ca[s0:s1]
        bad = ~np.isfinite(guide).all(axis=1)
        if bad.any():
            guide[bad] = np.cross(np.gradient(pca, axis=0), [0.31, 0.72, 0.62])[bad]
        for i in range(1, n):
            if np.dot(guide[i], guide[i - 1]) < 0 and not hx[i]:
                guide[i] = -guide[i]
        g = _lerp_vectors(guide, u)
        nrm = g - (np.sum(g * tan, axis=1, keepdims=True)) * tan
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        bin_ = np.cross(tan, nrm)
        # cross-section dimensions per sample
        ridx = np.clip(np.rint(u).astype(int), 0, n - 1)
        w = np.full(m, 0.30)
        h = np.full(m, 0.30)
        if style == "cartoon":
            w[hx[ridx]], h[hx[ridx]] = 1.20, 0.25
            w[st[ridx]], h[st[ridx]] = 1.10, 0.25
            # arrow heads at the end of each strand
            for i in range(n):
                if st[i] and (i == n - 1 or not st[i + 1]):
                    sel = (u >= i - 0.5) & (u <= i + 0.5)
                    frac = np.clip((u[sel] - (i - 0.5)), 0, 1)
                    w[sel] = 1.7 * (1 - frac) + 0.30 * frac
                    h[sel] = 0.25
            # soften helix/coil transitions (keep arrow heads sharp)
            ker = np.ones(5) / 5
            is_arrow = np.zeros(m, bool)
            for i in range(n):
                if st[i] and (i == n - 1 or not st[i + 1]):
                    is_arrow |= (u >= i - 0.5) & (u <= i + 0.5)
            ws = np.convolve(np.pad(w, 2, mode="edge"), ker, mode="valid")
            hs = np.convolve(np.pad(h, 2, mode="edge"), ker, mode="valid")
            w = np.where(is_arrow, w, ws)
            h = np.where(is_arrow, h, hs)
        w *= scale
        h *= scale
        th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        ct, stn = np.cos(th), np.sin(th)
        # superellipse-ish profile: flat faces for ribbons, round for coil
        x = w[:, None] * ct[None, :]
        y = h[:, None] * stn[None, :]
        nx = ct[None, :] / np.maximum(w[:, None], 1e-3)
        ny = stn[None, :] / np.maximum(h[:, None], 1e-3)
        ring = pts[:, None, :] + x[..., None] * nrm[:, None, :] + y[..., None] * bin_[:, None, :]
        rn = nx[..., None] * nrm[:, None, :] + ny[..., None] * bin_[:, None, :]
        rn /= np.maximum(np.linalg.norm(rn, axis=2, keepdims=True), 1e-6)
        P.append(ring.reshape(-1, 3))
        N.append(rn.reshape(-1, 3))
        col = colors[s0:s1][ridx]
        C.append(np.repeat(col, sides, axis=0))
        # triangles
        a = np.arange(m - 1)[:, None] * sides + np.arange(sides)[None, :]
        b = np.arange(m - 1)[:, None] * sides + (np.arange(sides)[None, :] + 1) % sides
        tri1 = np.stack([a, b, a + sides], axis=-1).reshape(-1, 3)
        tri2 = np.stack([b, b + sides, a + sides], axis=-1).reshape(-1, 3)
        T.append(np.vstack([tri1, tri2]) + offset)
        base = offset
        offset += m * sides
        # end caps (fan around the centre point)
        for end, sign in ((0, -1), (m - 1, 1)):
            centre_idx = offset
            P.append(pts[end][None])
            N.append((sign * tan[end])[None])
            C.append(col[end][None])
            ring_idx = base + end * sides + np.arange(sides)
            nxt = np.roll(ring_idx, -1)
            cap = np.stack([np.full(sides, centre_idx), nxt, ring_idx] if sign < 0 else
                           [np.full(sides, centre_idx), ring_idx, nxt], axis=-1)
            T.append(cap)
            offset += 1
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3), np.uint8)
    return (np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32),
            np.vstack(T).astype(np.int64), np.vstack(C).astype(np.uint8))


def gaussian_surface_grid(pos: np.ndarray, radii: np.ndarray, spacing: float = 0.7, probe: float = 1.4,
                          kappa: float = 1.0):
    """Gaussian density grid (Grant-Pickup) approximating a molecular surface; contour at level 1.0."""
    r_eff = radii + probe * 0.35
    pad = r_eff.max() * 2.4 + spacing
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    grid = np.zeros(dims[::-1], dtype=np.float32)  # z, y, x ordering for vtkImageData
    for p, r in zip(pos, r_eff):
        cut = r * 2.3
        i0 = np.maximum(np.floor((p - cut - lo) / spacing).astype(int), 0)
        i1 = np.minimum(np.ceil((p + cut - lo) / spacing).astype(int) + 1, dims)
        xs = lo[0] + np.arange(i0[0], i1[0]) * spacing - p[0]
        ys = lo[1] + np.arange(i0[1], i1[1]) * spacing - p[1]
        zs = lo[2] + np.arange(i0[2], i1[2]) * spacing - p[2]
        d2 = zs[:, None, None] ** 2 + ys[None, :, None] ** 2 + xs[None, None, :] ** 2
        grid[i0[2]:i1[2], i0[1]:i1[1], i0[0]:i1[0]] += np.exp(-kappa * (d2 / (r * r) - 1.0))
    return grid, lo, spacing, dims
