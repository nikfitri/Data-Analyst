"""Session state: representations, view, render settings (JSON serialisable)."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

STYLES = ["cartoon", "tube", "licorice", "ball+stick", "vdw", "lines", "surface"]
COLOR_SCHEMES = ["element", "secondary", "chain", "residue type", "residue index", "hydrophobicity",
                 "uniform", "rmsf", "bfactor"]
MATERIALS = ["default", "matte", "glossy", "shiny", "flat", "soft"]


@dataclass
class Representation:
    name: str = "Protein"
    selection: str = "protein"
    style: str = "cartoon"
    color: str = "secondary"
    uniform_color: str = "#4e79a7"
    carbon_color: str = ""          # element scheme: override carbon colour ("" = grey)
    opacity: float = 1.0
    size: float = 1.0
    material: str = "default"
    visible: bool = True
    update_every_frame: bool = False   # re-evaluate distance-based selections each frame
    selection_frame: int = 0           # frame used for distance-based selections otherwise


@dataclass
class ViewState:
    position: list = field(default_factory=lambda: [0.0, 0.0, 100.0])
    focal_point: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    view_up: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    view_angle: float = 30.0
    parallel: bool = True
    parallel_scale: float = 30.0
    auto: bool = True               # True = no camera saved yet; frame the scene automatically
    auto_mode: str = "overview"     # automatic framing: "overview" (whole system) or "pocket" (ligand close-up)


@dataclass
class RenderSettings:
    background: str = "#ffffff"
    background2: str = ""           # gradient bottom colour ("" = none)
    ssao: bool = True               # ambient occlusion
    fxaa: bool = True
    light_intensity: float = 1.0
    show_axes: bool = False
    title: str = ""
    title_size: int = 28
    show_time: bool = True
    time_format: str = "t = {ns:.2f} ns"
    text_color: str = "#202020"
    show_legend: bool = True
    font: str = "arial"
    slab: float = 0.0               # clipping slab thickness in Angstrom around the focal point (0 = off)
    label_box: bool = True          # draw a soft background box behind 3D labels


@dataclass
class InteractionSettings:
    show: bool = True
    ligand: str = "ligand"
    receptor: str = "protein"
    labels: bool = True
    label_size: int = 18
    label_color: str = "#202020"
    dash_radius: float = 0.08
    params: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)   # per-type colour overrides, e.g. {"H-bond": "#4f7fe0"}
    label_chain: bool = True                      # "Arg371:A" (True) or "Arg371" (False)
    label_mode: str = "current"                   # "current": residues interacting now; "pocket": the
                                                  # residues selected by `interacting` (stable in movies)
    pocket_samples: int = 40       # 0 = all frames; otherwise a deterministic preview sample
    callout_labels: bool = False   # screen-space pocket labels with leader lines
    label_positions: dict = field(default_factory=dict)  # residue index -> normalised screen position


@dataclass
class Session:
    topology: str = ""
    trajectories: list = field(default_factory=list)
    reps: list = field(default_factory=list)
    view: ViewState = field(default_factory=ViewState)
    render: RenderSettings = field(default_factory=RenderSettings)
    interactions: InteractionSettings = field(default_factory=InteractionSettings)
    pbc_fix: bool = False
    align: bool = False
    align_selection: str = "protein and name CA"
    smoothing: int = 1
    ss_every_frame: bool = False
    frame: int = 0
    saved_views: dict = field(default_factory=dict)
    measurements: list = field(default_factory=list)   # [[i, j], ...] atom index pairs
    atom_labels: list = field(default_factory=list)    # [i, ...]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls()
        for k, v in d.items():
            if k == "reps":
                s.reps = [Representation(**{kk: vv for kk, vv in r.items() if kk in Representation.__dataclass_fields__}) for r in v]
            elif k == "view":
                s.view = ViewState(**{kk: vv for kk, vv in v.items() if kk in ViewState.__dataclass_fields__})
            elif k == "render":
                s.render = RenderSettings(**{kk: vv for kk, vv in v.items() if kk in RenderSettings.__dataclass_fields__})
            elif k == "interactions":
                s.interactions = InteractionSettings(**{kk: vv for kk, vv in v.items() if kk in InteractionSettings.__dataclass_fields__})
            elif hasattr(s, k):
                setattr(s, k, v)
        return s

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path, "r") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str):
        with open(path, "w") as fh:
            fh.write(self.to_json())

    def copy(self) -> "Session":
        return copy.deepcopy(self)


def default_reps(system) -> list:
    """VMD-like publication defaults: cartoon protein, pocket side chains, ligand sticks."""
    reps = []
    has_protein = system.is_protein.any()
    if has_protein:
        reps.append(Representation("Protein", "protein", "cartoon", "secondary"))
    if system.ligand_resnames:
        if has_protein:
            reps.append(Representation("Binding pocket",
                                       "protein and not name N C O H HN and not nonpolarH and byres (around 5 ligand)",
                                       "licorice", "element", carbon_color="#b8b8b8", size=0.8))
        reps.append(Representation("Ligand", "ligand and not nonpolarH", "licorice", "element",
                                   carbon_color="#35b779", size=1.2))
    if not reps:
        reps.append(Representation("All", "all", "licorice", "element"))
    return reps


# ---------------------------------------------------------------------- style presets
STYLE_PRESETS = ["Default (VMD-like)", "Active-site close-up (publication)", "Overview (chains + ligand)"]

CLOSEUP_COLORS = {"H-bond": "#3f7be6", "Salt bridge": "#e0a92a", "Hydrophobic": "#1baf7a",
                  "Pi-stacking": "#c2569b", "Cation-pi": "#e87ba4", "Halogen bond": "#008300"}


def apply_style(session, system, name: str):
    """Replace representations / interaction / render settings with a named look (camera re-framed)."""
    lig = bool(system.ligand_resnames)
    it = session.interactions
    r = session.render
    if name.startswith("Active-site"):
        # ChimeraX-like active-site figure: soft blue cartoon, purple ligand, blue interacting residues,
        # dashed H-bonds / salt bridges, clean residue labels, clipping slab around the ligand.
        session.reps = [
            Representation("Protein", "protein", "cartoon", "uniform", uniform_color="#c1ccdd", material="matte",
                           size=0.48),
            Representation("Interacting residues",
                           "protein and ((interacting and not name N C O H HN and not nonpolarH) or contact_atoms)",
                           "ball+stick", "element", carbon_color="#6996bc", size=0.82, material="soft"),
        ]
        if lig:
            session.reps += [
                Representation("Ligand", "ligand and not nonpolarH", "ball+stick", "element",
                               carbon_color="#a07ee0", size=0.85, material="soft"),
                Representation("Ions near ligand", "ions and around 7 ligand", "vdw", "element", size=0.45,
                               update_every_frame=True),
            ]
        it.show = lig
        it.labels = True
        it.label_chain = False
        it.label_mode = "pocket"
        it.label_size = 24
        it.callout_labels = True
        it.label_color = "#262626"
        it.dash_radius = 0.07
        it.colors = dict(CLOSEUP_COLORS)
        params = dict(it.params)
        params["enabled"] = ["H-bond", "Salt bridge", "Pi-stacking", "Cation-pi", "Halogen bond"]
        it.params = params
        r.background, r.background2 = "#ffffff", ""
        r.ssao, r.fxaa, r.light_intensity = True, True, 1.2
        r.label_box = True
        r.show_legend = True
        r.show_time = system.n_frames > 1
        r.title_size = 30
        r.slab = 26.0 if lig else 0.0
        session.smoothing = 3 if system.n_frames > 20 else 1
        session.view = ViewState(parallel=True, auto=True, auto_mode="pocket" if lig else "overview")
    elif name.startswith("Overview"):
        session.reps = [Representation("Protein", "protein", "cartoon", "chain", material="glossy")]
        if lig:
            session.reps.append(Representation("Ligand", "ligand and not nonpolarH", "vdw", "element",
                                               carbon_color="#a585e0", size=0.9))
        it.show = False
        r.label_box, r.show_legend, r.ssao, r.slab = False, False, True, 0.0
        session.view = ViewState(parallel=True, auto=True, auto_mode="overview")
    else:
        session.reps = default_reps(system)
        it.show = lig
        it.colors = {}
        it.label_chain, it.label_mode = True, "current"
        it.callout_labels = False
        it.params = {}
        r.label_box, r.slab, r.show_legend = True, 0.0, True
        session.view = ViewState(parallel=True, auto=True, auto_mode="overview")
    return session
