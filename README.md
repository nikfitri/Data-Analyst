Theme: System (Light)Light Theme: Default
MDViz
A desktop viewer for MD trajectories that reads the same files as VMD. Use it to inspect protein–ligand interactions and to export figures and movies for publication.

Viewer improvements (September 2026)
Measurements use physical frames. Smoothing changes the drawing only. Interaction detection, occupancy, distance labels, secondary structure and RMSF use unsmoothed coordinates, with the chosen periodic-boundary correction and fitting. Contact dashes follow displayed atoms, while their presence and the distances in the interaction table are measured on the unsmoothed frame.
Reliable recalculation. Changing processing settings invalidates RMSF, secondary-structure and pocket caches. Distance-based selections no longer overwrite in-memory trajectory coordinates. RMSF fits the selected atoms to frame 0, reports an error if fewer than three are selected, and samples at most 200 evenly spaced frames by default. It does not alter the viewer's fitting or smoothing settings.
Clearer pocket figures. View > Style preset > Active-site close-up uses thinner, muted protein ribbons, stronger pocket side chains, and includes backbone contact atoms with their heavy neighbours. Residue names are arranged in two columns with leader lines. Alt+drag a label to pin it; positions are saved in the session and reused in exports. Interactions > Reset label positions restores automatic placement. Very dense pockets may still need fewer labels or smaller text.
Explicit pocket sampling. Interactions > Stable pocket residues selects a deterministic 40-frame preview or all frames. The preview can miss short contacts. All-frame pocket discovery may take longer; this setting is separate from the occupancy analysis frame range. Pocket legends stay fixed even in frames without contacts.
Clickable contact timeline. Open Interactions > Trajectory occupancy analysis, choose the start, end and stride, then Run analysis. Calculation uses a separate trajectory reader in a background thread. Cancel preserves completed frames and marks the result as partial. Choose Show clickable timeline in viewer when finished. Click a column to jump to that sampled frame; normal playback moves the orange marker. A dashed marker means the displayed frame was not sampled. The plot shows the 30 most occupied contact rows; CSV exports retain every row. Columns are samples, not equal physical time intervals. Reopen via Analysis > Show contact timeline. Changing the system or analysis settings clears the old timeline so it is not mistaken for a new result.
Playback and export. Playback uses coarser cartoons and approximate Gaussian surfaces, restoring screen detail on pause. Exports keep their existing publication detail. Glyph buffers, labels and unchanged legends are reused; surfaces retain a bounded two-entry cache; hidden representations skip frame work. The Gaussian surface is a visual approximation, not a solvent-excluded surface calculation.
Analysis provenance. Occupancy and timeline CSVs now have an adjacent .metadata.json with source paths, frame IDs, atom selections, interaction thresholds and processing settings. Keep the source files and a saved MDViz session with exported figures. The metadata does not embed or hash trajectory data.
For a quick demonstration, reopen validation/52E5_updated.mdviz.json after running the optional real-data validation. That folder also contains example images, a contact timeline and a five-frame MP4.

Validation commands (from MDViz in PowerShell):

..\.venv\Scripts\python.exe -m pytest tests -q
$env:MDVIZ_VIEWER_TESTS = "1"
..\.venv\Scripts\python.exe -m pytest tests/test_real_viewer.py -q
The optional viewer tests require the local 52E5 topology/trajectory. They compare occupancy with smoothing off/on, render start/end/rotated views, verify labels, encode and decode a short movie, and exercise GUI frame navigation and label dragging. They do not run a new molecular dynamics simulation.

Start
Double-click MDViz.bat, or run it from a terminal.

Which folder: it works from any folder, and relative paths are resolved from where you run it.
PowerShell (the default VS Code terminal): type .\MDViz.bat, or the full path, instead of MDViz.bat.
MDViz.bat                                              # empty window
MDViz.bat ..\GROMACS\runs\52E5\md_10ns_52E5.tpr ..\GROMACS\runs\52E5\52E5_video_centered.xtc
MDViz.bat my_figure.mdviz.json                         # reopen a saved session
You can also drag a structure, a trajectory or a session file onto the window.

Files it reads
Structure / topology: .tpr .gro .pdb .psf .prmtop/.parm7 .top .mol2 .pqr .pdbqt .xyz .crd .cif .sdf/.mol .dms .gsd .mmtf
Trajectory: .xtc .trr .dcd .nc .mdcrd .lammpstrj .trj .inpcrd/.rst7 .h5md .tng, plus multi-model .pdb/.gro
You can load several trajectory files at once. They are joined in order, like mol addfile in VMD.

Use a .tpr (or .psf/.prmtop) as the topology when you have one. These contain the real bonds and masses. With a .gro or .pdb, bonds are guessed.

Main features
Area	What you can do
Representations	cartoon, tube, licorice, ball+stick, VDW, lines, surface
Colouring	element (with a custom carbon colour), secondary structure (DSSP), chain, residue type, rainbow, hydrophobicity, RMSF, B-factor, uniform
Other settings	opacity, size, material; any number of representations
Selections	VMD/MDAnalysis syntax (protein and within 5 of ligand, byres (around 4 ligand), resid 100 to 120, …)
Selection macros	ligand water ions solute heavy hydrogen polarH nonpolarH
View	mouse rotate / zoom / pan / roll; buttons for zoom and ±X/±Y/±Z rotation with a set step; reset; focus on the ligand or any selection; orthographic or perspective; named saved views; clipping slab to see into buried pockets
Trajectory	play, step and loop; fix periodic boundaries; fit frames to remove tumbling; moving-average smoothing; secondary structure recomputed per frame (optional)
Interactions	H-bonds, salt bridges, hydrophobic contacts, π-stacking (parallel and T-shaped), cation–π and halogen bonds, found per frame; drawn as coloured dashes with residue labels; cut-offs are editable
Occupancy analysis	% of frames each interaction is present across the trajectory; saves a CSV table, a per-frame timeline CSV, and occupancy/timeline plots (PNG/SVG/PDF/TIFF, 300 dpi)
Measurements	Ctrl+click atoms to label them, measure distances, or centre the view
Images	PNG / JPEG / TIFF, up to 8192 px, DPI metadata, 2–3× supersampling, transparent background, ambient-occlusion shading
Movies	H.264 MP4 at 720p, 1080p, 4K or square; any frame range and stride; camera spin; 360° turntable; fly-through between two saved views; title, time stamp and legend overlays
Style presets (View ▸ Style preset, or --style on the command line):

Active-site close-up (publication): a ChimeraX-like binding-site figure.
Thin, muted blue-grey cartoon, purple-carbon ligand in ball-and-stick.
Blue side chains of residues making H-bonds, salt bridges or π/halogen contacts in the selected pocket sample, plus backbone contact atoms and nearby ions. Choose all frames for exhaustive pocket discovery.
Dashed interaction lines and clean residue labels (e.g. Ser105).
A clipping slab and a camera looking into the pocket.
Residues and labels stay fixed through a movie, so they don't flicker.
Overview (chains + ligand): the whole complex coloured by chain, with the ligand space-filling.
Default (VMD-like).
Periodic boundaries: raw GROMACS trajectories keep atoms inside the box, so molecules drift across the box edge and get split. Drawn naively, that gives long streaks across the image and detached protein pieces.

When you open a trajectory, MDViz checks for split molecules. If it finds any, it switches on Fix periodic boundaries and fit to frame 0 automatically, and says so in the status bar.
Bonds longer than 3 Å are never drawn, as a safety net.
You no longer need gmx trjconv -pbc whole -center before rendering.
Batch figures: ..\.venv\Scripts\python.exe examples\render_figures.py renders close-ups, overviews and a close-up MP4 for every complex in the thesis folder. Output goes to MD simulation/Figures/.

Text sizes are defined relative to a 1080-px-high image. An exported image or movie therefore has the same layout as the screen, at any resolution.

Controls
Mouse: left-drag rotates, the wheel or right-drag zooms, and middle-drag or Shift+left-drag pans. Ctrl+left-drag rolls the view.
Ctrl+click an atom: the action is set by the Ctrl+click mode in the toolbar.
Keys: Space plays or pauses, ←/→ steps one frame, and Home/End jump to the first/last frame. +/− zoom, R resets the view, F focuses the ligand, and Esc clears labels and measurements.
Batch rendering (no window)
Save a session from the GUI (File ▸ Save session). You can then render from it on the command line. Use --top and --traj to apply the same look to another ligand's run:

MDViz.bat --session pocket.mdviz.json --image fig_52E5.png --frame -1 --size 3000x2000 --dpi 300
MDViz.bat --session pocket.mdviz.json --movie 52E5.mp4 --size 1920x1080 --fps 30 --stride 1 --spin 30
MDViz.bat --session pocket.mdviz.json --movie turn.mp4 --turntable 12
MDViz.bat --session pocket.mdviz.json --occupancy 52E5_occ.csv --plots
MDViz.bat --session pocket.mdviz.json --top ..\GROMACS\runs\52E17\md.tpr --traj ..\GROMACS\runs\52E17\md.xtc --image fig_52E17.png
The ligand macro picks up whatever ligand residue each run contains. You don't need to edit the selections when switching runs.

Interaction criteria (defaults)
The defaults are geometric and close to PLIP's.

Type	Criterion (default)
H-bond	donor–acceptor ≤ 3.5 Å and D–H···A ≥ 120° (explicit hydrogens)
Salt bridge	oppositely charged groups ≤ 4.0 Å. Protein: Lys, Arg, Asp, Glu, termini. Ligand charges come from topology (4-connected N, carboxylate, phosphate, sulfonate).
Hydrophobic	carbon/sulfur/halogen atoms bonded only to C/H, ≤ 4.0 Å; one closest pair per residue
π-stacking	ring centroids ≤ 5.5 Å at < 30° (parallel), or ≤ 6.0 Å at > 60° (T-shaped); lateral offset ≤ 2.0 Å
Cation–π	Lys NZ / Arg CZ / ligand cation to ring centroid ≤ 6.0 Å, offset ≤ 2.5 Å
Halogen bond	Cl/Br/I···O/N/S ≤ 3.5 Å and C–X···A ≥ 140°
Water bridges are not detected. To see them, show bridging waters with the preset Waters within 3.5 Å of ligand (dynamic).

Potential energy surface (φ, ψ)
The PES (φ, ψ) tab sits next to Interactions (also under Analysis ▸ Potential energy surface). It maps the energy of alanine dipeptide (Ace-Ala-Nme), or your own small molecule, as a function of its backbone dihedrals.

How a scan works:

The molecule is built and parameterised with pdb2gmx. AMBER99SB-ILDN is the default, and the other AMBER force fields can be chosen.
For every grid point, the relaxed reference structure is rotated to the target φ and ψ.
Both dihedrals are held with GROMACS [ dihedral_restraints ], with k = 10,000 kJ/mol/rad² by default.
All other degrees of freedom are minimised in vacuum.
E(φ, ψ) is the potential energy minus the restraint energy, in kJ/mol. The panel also records the φ/ψ actually reached (within about 1.5° of target).
Speed and background running:

Minimisations run in parallel inside WSL (GROMACS 2021.4 on this machine). A 10° grid (1,296 points) takes about 80 s; a 5° grid (5,184 points) takes about 5 min.
Scans run in the background, and the surface fills in live while they run.
Cancel keeps every finished point. Running again with the same work folder resumes the scan.
Loading and saving:

Load PES opens a finished scan (.npz, which includes every minimised conformation).
It also opens a pre-computed table (.csv/.dat/.xvg with φ, ψ, E columns). A table has no conformations, so the molecule is then drawn with idealised geometry.
examples/ala2_amber99sb-ildn_10deg.npz is a ready-made 10° surface.
Interacting with the surface:

The bottom dock shows an interactive 3D surface and a 2D contour map (the Ramachandran projection). You can rotate and zoom both.
Hovering or clicking any point shows that minimised conformation in the 3D viewer, with its exact E and ΔE. The trajectory slider steps through the grid points and moves the marker on the plot.
Stationary points are found on a periodic bicubic spline of the surface: minima M1…, and first-order saddle points TS1…, classified by the Hessian. They are marked on both plots and listed in a table; click a row to go to that point.
Publication figure… saves a 2D map and a 3D surface (PNG/SVG/PDF/TIFF, 300 dpi). Save PES… writes .npz or .csv.
Return the 3D viewer to the MD system brings back the trajectory you had open before.
Results for AMBER99SB-ILDN in vacuum (10° grid):

Point	Type	φ, ψ (°)	ΔE (kJ/mol)	Region
M1	minimum	−78, 54	0.0	C7eq
M2	minimum	−147, 159	2.3	C5 / β
M3	minimum	60, −41	5.7	C7ax
TS1	saddle	−82, 120	8.0	C7eq ↔ C5
TS3	saddle	−2, −26	36.2	C7eq ↔ C7ax (φ ≈ 0 crossing)
Ligand PES from an MD run: this scans the ligand's torsions, not the protein.

Open the run: md_10ns_52E5.tpr + md_10ns_52E5.xtc.
In the PES tab, choose Ligand from the loaded MD run.
Click Find ligand torsions in the MD run. It finds the ligand's GROMACS topology in the run folder (the acpype GAFF2 files), lists the rotatable bonds, and ranks them by how much they move during the MD.
The two most mobile torsions are pre-selected; change them if you like.
Pick a grid and a dielectric (εr = 4 is the default here; 1 = vacuum, 80 ≈ water screening), then press Run PES scan.
What you get:

The MD frames overlaid on the surface.
A readout of how far above the ligand's gas-phase minimum the bound torsions sit. This is a 2D conformational-strain estimate.
How minima and saddles are chosen: a minimum must be at least Minimum basin depth (default 2 kJ/mol) below the barrier to any lower basin. Saddles are the lowest crossings between basins, labelled e.g. M1↔M3.

From the command line:

MDViz.bat --pes-ligand 52E5_lig.npz --top ..\GROMACS\runs\52E5\md_10ns_52E5.tpr --traj ..\GROMACS\runs\52E5\md_10ns_52E5.xtc --pes-eps 4
Custom molecules: choose Custom molecule (.gro + .top). The molecule can be written in the .top itself or pulled in from a local .itp via #include; it must have a [ bonds ] section. φ and ψ are found automatically from C–N–CA–C / N–CA–C–N, or you can type four atom numbers for each.

From the command line:

MDViz.bat --pes-scan ala2.npz --pes-step 10 --pes-ff amber99sb-ildn     # scan + figures + CSV
MDViz.bat --pes-figure ala2.npz --pes-cap 60                            # figures from an existing surface
Tests: ..\.venv\Scripts\python -m pytest tests. Set MDVIZ_GMX_TESTS=1 to also run a real GROMACS scan.

Installing on another machine
python -m venv .venv
.venv\Scripts\pip install -r MDViz\requirements.txt
ffmpeg comes bundled with imageio-ffmpeg, so no separate install is needed.
