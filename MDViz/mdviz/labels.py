"""Deterministic screen-space placement for pocket residue labels."""
import numpy as np


def layout_labels(anchors, widths, width, height, font_size):
    """Two columns, ordered by anchor height; keep labels inside the viewport.

    Dense scenes may exceed available height; users can reduce label size or
    the selected residue set. Input order is preserved in the return value.
    """
    if not len(anchors):
        return []
    anchors = np.asarray(anchors)
    centre = float(np.median(anchors[:, 0]))
    margin = max(8, font_size)
    bottom, top = margin * 2, max(margin * 2, height - margin * 3)
    out = [None] * len(anchors)
    for left in (True, False):
        ids = [i for i, a in enumerate(anchors) if (a[0] < centre) == left]
        ids.sort(key=lambda i: anchors[i, 1])
        if not ids:
            continue
        gap = min(font_size * 1.55, (top - bottom) / max(1, len(ids) - 1))
        ys = []
        for i in ids:
            ys.append(max(bottom, min(top, anchors[i, 1]), ys[-1] + gap if ys else bottom))
        if ys[-1] > top:
            ys[-1] = top
            for k in range(len(ys) - 2, -1, -1):
                ys[k] = min(ys[k], ys[k + 1] - gap)
        for i, y in zip(ids, ys):
            x = centre - width * 0.22 - widths[i] if left else centre + width * 0.22
            out[i] = (max(margin, min(x, width - margin - widths[i])), y)
    return out
