# MDViz viewer validation — 29 September 2026

Final combined run: **34 passed, 1 skipped**, 51.77 seconds.

```powershell
$env:MDVIZ_VIEWER_TESTS = "1"
..\.venv\Scripts\python.exe -m pytest tests -q
```

The skipped test is the optional real GROMACS PES scan (`MDVIZ_GMX_TESTS=1`); no new simulation was run.
The run emitted VTK/Qt deprecation warnings and an MDAnalysis trajectory-offset refresh warning.
These did not fail the tests. The local trajectory grew during this work, so sampled frame IDs are
recorded explicitly in `validation.json` and the CSV metadata, rather than assuming a fixed file length.

## Checked

- Known synthetic contact occupancy and analytical RMSF, invariant to display smoothing.
- Cache invalidation, sample-count limits, and preservation of in-memory source coordinates.
- Cancellation, partial-result labelling, worker errors and independent worker trajectory readers.
- Screen-space label placement, saved label positions and Alt-drag input at a realistic viewport size.
- Exact mapping from timeline columns to sampled trajectory frame IDs; clearing stale results.
- Playback quality restoration, glyph-object reuse and surface-cache reuse.
- Real 52E5 data: 80,681 atoms and 3,601 frames at the time of the final export test. Contact occupancy
  matched for smoothing windows 1 and 5 on 12 frames distributed across the loaded trajectory.
- PNG exports of start, end and rotated views; visible labels and nonblank images. These views were
  also inspected visually during development.
- A five-frame, 640×420 MP4, decoded again to verify frame count, dimensions and nonblank content.
- Existing PES tests.

## Try the changes

Restart MDViz and open `52E5_updated.mdviz.json` in this folder, or choose
**View > Style preset > Active-site close-up (publication)** for your own system.
Alt-drag a residue label to pin it, then save the session to keep its placement.

For the timeline, choose **Interactions > Trajectory occupancy analysis**, run the analysis and click
**Show clickable timeline in viewer**. Click any column to inspect its actual sampled frame.
Choose a stride of 1 for every frame; the supplied validation timeline is only a 12-frame test sample.

`52E5_previous_style.png` reconstructs the previous styling using the current renderer and the same
camera. It is a style comparison, not an old-version performance benchmark.

## Limits

This validates the tested workflows, not every supported topology, GPU or chemical interaction rule.
The Gaussian surface remains approximate. Very dense pockets can require manual label placement.
No frame-rate speedup is claimed: rendering work was reduced and object reuse was tested, but a
controlled before/after playback benchmark was not performed. The real-data comparison sampled 12
frames; it was not a full-trajectory scientific validation. Additional RMSD/torsion plot tracks, an
overview inset and synchronised multi-run comparison remain separate enhancements.
