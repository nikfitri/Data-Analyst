"""Interactive PES plots: Plotly 3D surface + 2D contour map inside a QWebEngineView.

Hover/click events travel back to Python through QWebChannel as (phi, psi) in degrees.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
from PySide6.QtCore import QFile, QIODevice, QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

# Sequential single-hue ramp (light = low energy, dark = high energy)
PES_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SELECT_COLOR = "#eb6834"
INK, INK_2, GRID = "#1f1f1f", "#5c5c5c", "#e4e4e2"

_PAGE_JS = r"""
const INK='%INK%', INK2='%INK2%', GRID='%GRID%', SEL='%SEL%';
const RAMP=%RAMP%;
let bridge=null, D=null, lastKey=null, pending=null, bound=false;
new QWebChannel(qt.webChannelTransport, ch => { bridge = ch.objects.bridge; bridge.ready(); });
const cfg = {displaylogo:false, responsive:true, toImageButtonOptions:{format:'png', scale:3},
             modeBarButtonsToRemove:['lasso2d','select2d']};
let HOVER = '';
function axis(title){ return {title:{text:title, font:{color:INK, size:13}}, range:[-180,180], dtick:90,
  tickfont:{color:INK2}, gridcolor:GRID, zeroline:false, linecolor:INK2, ticks:'outside', tickcolor:INK2}; }
function statTraces(is3d){
  const mk = (type, symbol, name) => {
    const P = D.points.filter(p => p.type===type);
    const t = {x:P.map(p=>p.phi), y:P.map(p=>p.psi), text:P.map(p=>p.label), name:name,
      mode:'markers+text', textposition:'top center', textfont:{color:INK, size:is3d?11:12},
      customdata:P.map(p=>p.rel), hovertemplate:'%{text}  φ %{x:.1f}°  ψ %{y:.1f}°<br>ΔE %{customdata:.2f} kJ/mol<extra></extra>',
      showlegend:!is3d};
    if (is3d){ t.type='scatter3d'; t.z=P.map(p=>Math.min(p.rel, D.cap)+D.cap*0.015);
               t.marker={symbol:symbol, size:5, color:INK, line:{color:'#ffffff', width:1}}; }
    else { t.type='scatter'; t.marker={symbol:symbol, size:10, color:INK, line:{color:'#ffffff', width:2}}; }
    return t; };
  return [mk('minimum','circle','Minimum (M)'), mk('saddle', is3d?'diamond-open':'x-thin-open', 'Saddle point (TS)')];
}
function selTrace(is3d){
  const t = {x:[null], y:[null], name:'Selected', mode:'markers', hoverinfo:'skip', showlegend:!is3d};
  if (is3d){ t.type='scatter3d'; t.z=[null]; t.marker={symbol:'circle-open', size:10, color:SEL, line:{width:3, color:SEL}}; }
  else { t.type='scatter'; t.marker={symbol:'circle-open', size:18, color:SEL, line:{width:3, color:SEL}}; }
  return t;
}
function mdTrace(is3d){
  const t = {x:D.md.x, y:D.md.y, name:'MD frames (bound)', mode:'markers', hoverinfo:'skip', showlegend:!is3d};
  if (is3d){ t.type='scatter3d'; t.z=D.md.z.map(v => Math.min(v, D.cap)+D.cap*0.01);
             t.marker={size:1.8, color:INK, opacity:0.45}; }
  else { t.type='scatter'; t.marker={size:3.5, color:INK, opacity:0.3}; }
  return t;
}
function draw(){
  HOVER = D.xl+' %{x:.0f}°  '+D.yl+' %{y:.0f}°<br>ΔE %{customdata:.2f} kJ/mol<extra></extra>';
  const zc = D.z.map(r => r.map(v => v===null ? null : Math.min(v, D.cap)));
  const surf = {type:'surface', x:D.phi, y:D.psi, z:zc, surfacecolor:zc, cmin:0, cmax:D.cap, colorscale:RAMP,
    showscale:false, customdata:D.z, hovertemplate:HOVER,
    contours:{z:{show:true, usecolormap:false, color:'rgba(255,255,255,0.55)', width:1, start:0, end:D.cap, size:D.cstep}},
    lighting:{ambient:0.75, diffuse:0.55, specular:0.08, roughness:0.9}};
  const scene = {xaxis:Object.assign(axis(D.xl+' (°)'), {backgroundcolor:'#ffffff'}),
    yaxis:Object.assign(axis(D.yl+' (°)'), {backgroundcolor:'#ffffff'}),
    zaxis:{title:{text:'ΔE (kJ/mol)', font:{color:INK, size:13}}, range:[0, D.cap*1.08], tickfont:{color:INK2}, gridcolor:GRID},
    camera:{eye:{x:-1.2, y:-1.3, z:0.85}}, aspectmode:'manual', aspectratio:{x:1, y:1, z:0.6}};
  const md3 = D.md ? [mdTrace(true)] : [], md2 = D.md ? [mdTrace(false)] : [];
  Plotly.react('surf', [surf].concat(md3, statTraces(true), [selTrace(true)]),
    {margin:{l:0,r:0,t:36,b:0}, paper_bgcolor:'#ffffff', scene:scene, uirevision:'keep',
     title:{text:D.title3d, font:{color:INK, size:14}, x:0.02, xanchor:'left'}}, cfg);
  const cont = {type:'contour', x:D.phi, y:D.psi, z:zc, zmin:0, zmax:D.cap, colorscale:RAMP, customdata:D.z,
    hovertemplate:HOVER, contours:{start:0, end:D.cap, size:D.cstep, coloring:'fill', showlines:true},
    line:{color:'rgba(255,255,255,0.6)', width:0.8}, connectgaps:false,
    colorbar:{title:{text:'ΔE<br>(kJ/mol)', font:{color:INK, size:12}}, thickness:12, outlinewidth:0, tickfont:{color:INK2}, len:0.9}};
  Plotly.react('cont', [cont].concat(md2, statTraces(false), [selTrace(false)]),
    {margin:{l:62,r:10,t:58,b:48}, paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff', uirevision:'keep',
     xaxis:Object.assign(axis(D.xl+' (°)'), {constrain:'domain'}),
     yaxis:Object.assign(axis(D.yl+' (°)'), {scaleanchor:'x', scaleratio:1, constrain:'domain'}),
     legend:{orientation:'h', x:1, xanchor:'right', y:1.02, yanchor:'bottom', font:{color:INK, size:11}},
     title:{text:D.title2d, font:{color:INK, size:14}, x:0.02, xanchor:'left'}}, cfg);
  if (!bound){ bound = true;
    for (const id of ['surf','cont']){ const el = document.getElementById(id);
      el.on('plotly_hover', e => send('hover', e)); el.on('plotly_click', e => send('click', e)); } }
}
function send(kind, e){
  const p = e && e.points && e.points[0]; if (!p || p.x===undefined || p.y===undefined || !bridge) return;
  const x = +p.x, y = +p.y;
  if (kind==='hover'){ const key = x.toFixed(1)+','+y.toFixed(1); if (key===lastKey) return; lastKey = key;
    if (pending) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => { pending = null; bridge.hovered(x, y); }); }
  else bridge.clicked(x, y);
}
function setData(d){ D = d; draw(); }
function setSelected(phi, psi, rel){
  if (!D) return;
  const k = D.md ? 4 : 3;  // the selection marker is the last trace
  Plotly.restyle('surf', {x:[[phi]], y:[[psi]], z:[[Math.min(rel, D.cap)+D.cap*0.02]]}, [k]);
  Plotly.restyle('cont', {x:[[phi]], y:[[psi]]}, [k]);
}
"""

_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:#fff;font-family:Arial,Helvetica,sans-serif;overflow:hidden}
#wrap{display:flex;height:100%;width:100%} #surf{flex:1.15;min-width:0} #cont{flex:1;min-width:0}
#empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#5c5c5c;font-size:14px}
</style>
<script>%QWEBCHANNEL%</script><script>%PLOTLY%</script></head>
<body><div id="empty">Run a scan or load a PES (.npz / .csv) to see the surface.</div>
<div id="wrap"><div id="surf"></div><div id="cont"></div></div>
<script>%JS%
const _sd = setData; setData = function(d){ document.getElementById('empty').style.display='none'; _sd(d); };
</script></body></html>"""


def _page_path() -> str:
    """Build the HTML page once (Plotly and qwebchannel.js inlined so it works offline)."""
    import hashlib
    import plotly
    key = hashlib.md5((_PAGE + _PAGE_JS + plotly.__version__).encode()).hexdigest()[:10]
    path = os.path.join(tempfile.gettempdir(), f"mdviz_pes_view_{key}.html")
    if os.path.exists(path):
        return path
    from plotly.offline import get_plotlyjs
    f = QFile(":/qtwebchannel/qwebchannel.js")
    f.open(QIODevice.ReadOnly)
    qwc = bytes(f.readAll()).decode("utf-8")
    f.close()
    ramp = json.dumps([[k / (len(PES_RAMP) - 1), c] for k, c in enumerate(PES_RAMP)])
    js = (_PAGE_JS.replace("%INK%", INK).replace("%INK2%", INK_2).replace("%GRID%", GRID)
          .replace("%SEL%", SELECT_COLOR).replace("%RAMP%", ramp))
    html = _PAGE.replace("%QWEBCHANNEL%", qwc).replace("%PLOTLY%", get_plotlyjs()).replace("%JS%", js)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


class _Bridge(QObject):
    readySig = Signal()
    hoveredSig = Signal(float, float)
    clickedSig = Signal(float, float)

    @Slot()
    def ready(self):
        self.readySig.emit()

    @Slot(float, float)
    def hovered(self, x, y):
        self.hoveredSig.emit(x, y)

    @Slot(float, float)
    def clicked(self, x, y):
        self.clickedSig.emit(x, y)


def _clean(a):
    return [[None if (v is None or not math.isfinite(v)) else round(float(v), 4) for v in row] for row in a]


class PESPlotView(QWidget):
    hovered = Signal(float, float)
    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.web = QWebEngineView(self)
        self.web.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.bridge = _Bridge()
        self.channel = QWebChannel(self.web.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web.page().setWebChannel(self.channel)
        self.bridge.readySig.connect(self._on_ready)
        self.bridge.hoveredSig.connect(self.hovered)
        self.bridge.clickedSig.connect(self.clicked)
        self._ready = False
        self._pending = []
        lay.addWidget(self.web)
        self.web.load(QUrl.fromLocalFile(_page_path()))

    def _on_ready(self):
        self._ready = True
        for js in self._pending:
            self.web.page().runJavaScript(js)
        self._pending = []

    def _js(self, code):
        if self._ready:
            self.web.page().runJavaScript(code)
        else:
            self._pending = [c for c in self._pending if not c.startswith(code.split("(")[0])] + [code]

    def set_result(self, res, points, cap=60.0, cstep=None, title=""):
        rel = res.rel
        md = None
        if getattr(res, "md_samples", None) is not None and len(res.md_samples) and np.isfinite(res.energy).all():
            from ..pes import energy_at
            pts = np.asarray(res.md_samples)[:: max(1, len(res.md_samples) // 1500)]
            md = dict(x=[round(float(v), 2) for v in pts[:, 0]], y=[round(float(v), 2) for v in pts[:, 1]],
                      z=[round(float(v), 3) for v in energy_at(res, pts)])
        xl, yl = res.labels if hasattr(res, "labels") else ("φ", "ψ")
        cstep = cstep or (2.0 if cap <= 30 else 4.0 if cap <= 60 else 8.0)
        data = dict(phi=[float(x) for x in res.phi], psi=[float(x) for x in res.psi],
                    z=_clean(rel.T), cap=float(cap), cstep=float(cstep),
                    points=[dict(type=p["type"], label=p["label"], phi=round(p["phi"], 2), psi=round(p["psi"], 2),
                                 rel=round(p["rel"], 3)) for p in points],
                    md=md, xl=xl, yl=yl,
                    title3d=f"Potential energy surface{(' · ' + title) if title else ''}",
                    title2d="Energy map (Ramachandran projection)" if xl == "φ" else "Energy map with MD frames")
        self._js(f"setData({json.dumps(data)})")

    def set_selected(self, phi, psi, rel):
        rel = float(rel) if np.isfinite(rel) else 0.0
        self._js(f"setSelected({float(phi)}, {float(psi)}, {rel})")
