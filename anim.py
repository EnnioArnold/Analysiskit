#!/usr/bin/env python3
"""
Manim-Animationen zur Ableitung (freiwillig).

Drei Szenen:

    secant_to_tangent   Sekante durch P0 und P(x0+h): h wird kleiner, die
                        Sekante dreht sich in die Tangente, die Steigung läuft mit
    derivative_graph    Die Tangente wandert entlang f, daneben entsteht
                        Schritt für Schritt der Graph von f'
    curve_analysis      f, Monotoniebereiche färben, kritische Punkte
                        hervorheben, Krümmung und Wendepunkte

Bewusst ohne LaTeX: Achsen, Beschriftungen und Zahlen sind selbst
gezeichnete Linien und Text-Objekte. Dadurch genügen Cairo und Pango, eine
TeX-Installation ist nicht nötig.

Installation:
    brew install cairo pango pkg-config ffmpeg      # macOS
    pip install manim

Verwendung:
    from anakit.anim import build_job, render
    job = build_job("secant_to_tangent", "x^2", x0=1, quality="medium")
    pfad = render(job, out_dir="filme", progress=lambda p, t: print(p, t))

Von der Kommandozeile (rendert eine Szene in niedriger Qualität):
    python3 -m anakit.anim secant_to_tangent "x^2" 1
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------
# Auftragsbeschreibung — kennt weder Manim noch Qt
# --------------------------------------------------------------------------

#: Grösse, Bildrate und Manim-Kürzel je Qualitätsstufe.
QUALITIES = {
    "low": ("480p15", "-ql"),
    "medium": ("720p30", "-qm"),
    "high": ("1080p60", "-qh"),
    "ultra": ("1440p60", "-qp"),
}
#: Auch die Auflösung selbst ist als Name erlaubt: quality="1080p60".
QUALITY_ALIASES = {v[0]: k for k, v in QUALITIES.items()}

SCENES = ("secant_to_tangent", "derivative_graph", "curve_analysis", "limit_approach")
SCENE_NAMES = {"secant_to_tangent": "SecantToTangentScene",
               "derivative_graph": "DerivativeGraphScene",
               "curve_analysis": "CurveAnalysisScene",
               "limit_approach": "LimitApproachScene"}
#: Ungefähre Zahl der Teilanimationen, für den Fortschrittsbalken.
_EXPECTED = {"secant_to_tangent": 14, "derivative_graph": 12, "curve_analysis": 18,
             "limit_approach": 12}


@dataclass
class Theme:
    """Farben der Animation; dieselbe Farbfolge wie in plots.py."""
    background: str = "#ffffff"
    text: str = "#1d1d1f"
    muted: str = "#86868b"
    grid: str = "#e5e5ea"
    accent: str = "#0071e3"        # Funktion f
    secant: str = "#ff9500"
    tangent: str = "#d70015"
    derivative: str = "#8944ab"    # f'
    good: str = "#1d8a3c"          # f' > 0, steigend
    bad: str = "#d70015"           # f' < 0, fallend
    curved_left: str = "#5ac8fa"   # f'' > 0
    curved_right: str = "#ffcc00"  # f'' < 0
    hi: str = "#ff9500"
    lo: str = "#8944ab"
    sat: str = "#00a2a8"
    wen: str = "#c9a000"

    @staticmethod
    def dark() -> "Theme":
        return Theme(background="#1c1c1e", text="#f5f5f7", muted="#8e8e93",
                     grid="#2f2f32", accent="#0a84ff", secant="#ff9f0a",
                     tangent="#ff453a", derivative="#bf5af2", good="#30d158",
                     bad="#ff453a", curved_left="#40c8e0", curved_right="#ffd60a",
                     hi="#ff9f0a", lo="#bf5af2", sat="#40c8e0", wen="#ffd60a")


@dataclass
class AnimationJob:
    """
    Ein Rendering-Auftrag.

    scene   — 'secant_to_tangent' | 'derivative_graph' | 'curve_analysis' | 'limit_approach'
    f       — die Funktion als Text (SymPy-Schreibweise, z. B. 'x**2' oder
              'x^2 if x<1 else x-1')
    x0      — Stelle der Tangente bzw. des Grenzwerts (secant_to_tangent, limit_approach)
    hs      — die Abstände h, nacheinander (secant_to_tangent, limit_approach)
    xlim/ylim — Ausschnitt; None heisst automatisch
    fmt     — 'mp4' oder 'gif'
    """
    scene: str = "secant_to_tangent"
    f: str = "x^2"
    x0: float | None = 1.0
    hs: list[float] = field(default_factory=lambda: [2.0, 1.0, 0.5, 0.25, 0.1])
    xlim: list[float] | None = None
    ylim: list[float] | None = None
    title: str = ""
    theme: Theme = field(default_factory=Theme)
    quality: str = "low"
    fmt: str = "mp4"
    gif_width: int = 760
    gif_fps: int = 15

    def to_json(self) -> str:
        data = asdict(self)
        data["theme"] = asdict(self.theme)
        return json.dumps(data)

    @staticmethod
    def from_json(text: str) -> "AnimationJob":
        data = json.loads(text)
        theme = Theme(**data.pop("theme"))
        return AnimationJob(theme=theme, **data)


def _f_text(f) -> str:
    """Funktion (Text, SymPy oder Python-Funktion) als Text, der sich wieder einlesen lässt."""
    import sympy as sp
    from .core import parse
    try:
        expr = parse(f)
    except Exception as exc:
        raise ValueError(f"Die Funktion {f!r} lässt sich nicht lesen: {exc}") from None
    if isinstance(f, str):
        return f
    return sp.sstr(expr)


def build_job(scene: str, f, *, quality: str = "low", fmt: str = "mp4", x0=None,
              xlim=None, ylim=None, hs=None, dark: bool = False, title: str = "",
              theme: Theme | None = None) -> AnimationJob:
    """
    Baut einen Auftrag; rechnen und zeichnen tut erst render().

    scene   'secant_to_tangent', 'derivative_graph' oder 'curve_analysis'
    f       Funktion als Text, SymPy-Term oder Python-Funktion
    quality 'low' (Vorschau), 'medium' (720p30), 'high' (1080p60), 'ultra' (1440p60);
            auch '720p30' usw. sind erlaubt
    fmt     'mp4' oder 'gif'
    x0      Stelle der Tangente — nur für secant_to_tangent nötig
    hs      Abstände h der Sekante, in der Reihenfolge der Animation
    """
    quality = QUALITY_ALIASES.get(quality, quality)
    if scene not in SCENE_NAMES:
        raise ValueError(f"Unbekannte Szene {scene!r}. Möglich: " + ", ".join(SCENES))
    if quality not in QUALITIES:
        raise ValueError(f"Unbekannte Qualität {quality!r}. Möglich: "
                         + ", ".join(QUALITIES) + " oder " + ", ".join(QUALITY_ALIASES))
    if fmt not in ("mp4", "gif"):
        raise ValueError("fmt muss 'mp4' oder 'gif' sein")
    from .plots import _xval, _xlim, _Fn
    text = _f_text(f)
    xl = _xlim(xlim, None)
    x0v = None
    if scene in ("secant_to_tangent", "limit_approach"):
        if x0 is None:
            raise ValueError(f"Für {scene} braucht es x0, die Stelle.")
        x0v = _xval(x0, "x0")[0]
        if xl is None:
            xl = (x0v - 3.0, x0v + 3.0)
    hlist = [float(v) for v in (hs if hs is not None else (2, 1, 0.5, 0.25, 0.1))
             if float(v) != 0.0]
    if scene in ("secant_to_tangent", "limit_approach") and not hlist:
        raise ValueError("hs braucht mindestens einen Wert ungleich 0")
    if scene == "secant_to_tangent":
        fn = _Fn(text)
        import numpy as np
        y0 = fn(np.array([x0v]))[0]
        if not np.isfinite(y0):
            raise ValueError("f ist bei x0 nicht definiert")
    yl = None
    if ylim is not None:
        yl = (float(ylim[0]), float(ylim[1]))
        if yl[0] >= yl[1]:
            raise ValueError("ylim muss ein Paar (unten, oben) mit unten < oben sein")
    return AnimationJob(
        scene=scene, f=text, x0=x0v, hs=hlist,
        xlim=list(xl) if xl else None, ylim=list(yl) if yl else None,
        title=title, theme=theme or (Theme.dark() if dark else Theme()),
        quality=quality, fmt=fmt)


# --------------------------------------------------------------------------
# Verfügbarkeit
# --------------------------------------------------------------------------

def manim_available() -> tuple[bool, str]:
    """(verfügbar, Begründung) — prüft Manim und ffmpeg ohne Import-Fehler."""
    try:
        import manim  # noqa: F401
    except ImportError:
        return False, "Manim ist nicht installiert."
    except Exception as exc:                       # defekte Installation
        return False, f"Manim lässt sich nicht laden: {exc}"
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg fehlt — nötig für die Videoausgabe."
    return True, ""


INSTALL_HINT = ("brew install cairo pango pkg-config ffmpeg\n"
                "pip install manim")


# --------------------------------------------------------------------------
# Rechnen für die Szenen (ohne Manim)
# --------------------------------------------------------------------------

def _tick_values(lo: float, hi: float, target: int = 6) -> list[float]:
    import math
    raw = (hi - lo) / max(target, 1)
    e = 10 ** math.floor(math.log10(raw))
    step = next(k * e for k in (1, 2, 5, 10) if k * e >= raw * 0.9)
    first = math.ceil(lo / step - 1e-9)
    out = []
    k = first
    while k * step <= hi + 1e-9 * step:
        out.append(round(k * step, 10))
        k += 1
    return out


def _thin(xs, ys, limit: int = 900):
    """Höchstens `limit` Punkte, Anfang und Ende bleiben."""
    import numpy as np
    n = len(xs)
    if n <= limit:
        return xs, ys
    stride = -(-n // limit)
    idx = np.unique(np.r_[np.arange(0, n, stride), n - 1])
    return xs[idx], ys[idx]


def _pieces(scan, ylim):
    """Kurvenstücke (x, y), die im Bild bleiben; am Rand wird abgeschnitten."""
    import numpy as np
    ylo, yhi = ylim
    out = []
    for sx, sy in scan.segments:
        idx = np.flatnonzero((sy >= ylo) & (sy <= yhi))
        if idx.size < 2:
            continue
        for part in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(part) >= 2:
                out.append(_thin(sx[part], sy[part]))
    return out


def _clip_line(xa, ya, m, xlim, ylim):
    """Gerade durch (xa, ya) mit Steigung m, beschnitten auf das Bild: (x1, x2) oder None."""
    x1, x2 = xlim
    ylo, yhi = ylim
    if m == 0:
        return (x1, x2) if ylo <= ya <= yhi else None
    xs = sorted([xa + (ylo - ya) / m, xa + (yhi - ya) / m])
    a, b = max(x1, xs[0]), min(x2, xs[1])
    return (a, b) if a < b else None


def _pick_font() -> str | None:
    """Eine serifenlose Schrift, die auf dem Rechner da ist (sonst Manims Standard)."""
    try:
        import manimpango
        have = set(manimpango.list_fonts())
    except Exception:
        return None
    for name in ("Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in have:
            return name
    return None


def _setup(job: AnimationJob):
    """Gemeinsame Rechenarbeit: Funktion, Ausschnitt, Abtastung."""
    from .plots import _Fn, _auto_xlim
    fn = _Fn(job.f)
    xlim = tuple(job.xlim) if job.xlim else _auto_xlim(fn)
    return fn, xlim


# --------------------------------------------------------------------------
# Die Szenen
# --------------------------------------------------------------------------

def _build_scenes():
    """
    Erzeugt die Szenenklassen erst beim Aufruf, damit das Modul auch ohne
    installiertes Manim importierbar bleibt.
    """
    import numpy as np
    from manim import (Scene, VGroup, VMobject, Text, Line, DashedLine, Dot, Circle,
                       Polygon, FadeIn, FadeOut, Create, Write, ReplacementTransform,
                       GrowFromCenter, Indicate, ValueTracker, always_redraw, smooth,
                       linear, UP, DOWN, LEFT, RIGHT)
    from .plots import (_Fn, _deriv, _scan, _auto_ylim, _roots, _analyse,
                        _sign_intervals, _slope, _num, _pi_ticks, _pretty)

    class Frame:
        """Ein Koordinatenfeld: rechnet Zahlenpaare in Bildpunkte um und zeichnet die Achsen."""

        def __init__(self, xlim, ylim, width, height, center, trig=False, font=None):
            self.font = font
            self.xlim, self.ylim = tuple(xlim), tuple(ylim)
            self.w, self.h = width, height
            self.cx, self.cy = center
            self.trig = trig

        def p(self, x, y):
            x0, x1 = self.xlim
            y0, y1 = self.ylim
            return np.array([
                self.cx - self.w / 2 + (x - x0) / (x1 - x0) * self.w,
                self.cy - self.h / 2 + (y - y0) / (y1 - y0) * self.h, 0.0])

        def pts(self, xs, ys):
            return np.array([self.p(a, b) for a, b in zip(xs, ys)])

        def inside(self, x, y):
            return self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]

        def axes(self, th) -> VGroup:
            g = VGroup()
            fk = {"font": self.font} if self.font else {}
            x0, x1 = self.xlim
            y0, y1 = self.ylim
            xs, labels = None, None
            if self.trig:
                t = _pi_ticks(x0, x1)
                if t:
                    xs, labels = t
            if xs is None:
                xs = _tick_values(x0, x1, 6)
                labels = [_num(v) for v in xs]
            for v, lab in zip(xs, labels):
                g.add(Line(self.p(v, y0), self.p(v, y1), color=th.grid, stroke_width=1.2))
                t = Text(lab, font_size=16, color=th.muted, **fk)
                t.next_to(self.p(v, y0), DOWN, buff=0.1)
                g.add(t)
            for v in _tick_values(y0, y1, 5):
                g.add(Line(self.p(x0, v), self.p(x1, v), color=th.grid, stroke_width=1.2))
                t = Text(_num(v), font_size=16, color=th.muted, **fk)
                t.next_to(self.p(x0, v), LEFT, buff=0.1)
                g.add(t)
            ya = min(max(0.0, y0), y1)
            xa = min(max(0.0, x0), x1)
            g.add(Line(self.p(x0, ya), self.p(x1, ya), color=th.muted, stroke_width=2.2))
            g.add(Line(self.p(xa, y0), self.p(xa, y1), color=th.muted, stroke_width=2.2))
            return g

        def curve(self, pieces, color, width=5) -> VGroup:
            g = VGroup()
            for sx, sy in pieces:
                m = VMobject(color=color, stroke_width=width)
                m.set_points_as_corners(self.pts(sx, sy))
                g.add(m)
            return g

        def line(self, xa, ya, m, color, width=4, xrange=None):
            """Gerade durch (xa, ya), auf das Feld (und xrange) beschnitten."""
            xl = self.xlim if xrange is None else (max(self.xlim[0], xrange[0]),
                                                   min(self.xlim[1], xrange[1]))
            seg = _clip_line(xa, ya, m, xl, self.ylim)
            if seg is None:
                return VGroup()
            a, b = seg
            return Line(self.p(a, ya + m * (a - xa)), self.p(b, ya + m * (b - xa)),
                        color=color, stroke_width=width)

    class AnaScene(Scene):
        """Gemeinsame Grundlage: Auftrag laden, Funktion, Beschriftung."""

        def setup(self):
            self.job = AnimationJob.from_json(os.environ["ANAKIT_JOB"])
            self.th = self.job.theme
            self.camera.background_color = self.th.background
            self.fn, self.xlim = _setup(self.job)
            self.caption_mob = None
            self.font = _pick_font()

        # -- Hilfen ---------------------------------------------------------
        def T(self, text, size=26, color=None, **kw):
            if self.font:
                kw.setdefault("font", self.font)
            return Text(text, font_size=size, color=color or self.th.text, **kw)

        def headline(self, text):
            t = self.T(text, 36, weight="BOLD")
            t.to_edge(UP, buff=0.3)
            return t

        def say(self, text, size=26, color=None, wait=0.0):
            """Beschriftung am unteren Rand, ersetzt die vorige."""
            new = self.T(text, size, color)
            new.to_edge(DOWN, buff=0.28)
            if self.caption_mob is None:
                self.play(FadeIn(new, shift=UP * 0.15), run_time=0.6)
            else:
                self.play(ReplacementTransform(self.caption_mob, new), run_time=0.6)
            self.caption_mob = new
            if wait:
                self.wait(wait)

        def dot(self, point, color, r=0.1):
            return Dot(point, radius=r, color=color)

        def hollow(self, point, color, r=0.085):
            c = Circle(radius=r, color=color, stroke_width=3.5)
            c.set_fill(self.th.background, opacity=1)
            c.move_to(point)
            return c

        def markers(self, fr, scan, color):
            """Asymptoten, offene und geschlossene Kreise an den Unstetigkeiten."""
            g = VGroup()
            for e in scan.events:
                if e.pole and abs(e.x) > 1e-9 and fr.xlim[0] < e.x < fr.xlim[1]:
                    g.add(DashedLine(fr.p(e.x, fr.ylim[0]), fr.p(e.x, fr.ylim[1]),
                                     dash_length=0.12, color=self.th.muted, stroke_width=2))
                for state, v in ((e.ls, e.lv), (e.rs, e.rv)):
                    if state == "f" and fr.inside(e.x, v):
                        closed = e.val is not None and abs(e.val - v) <= 2e-3 * max(scan.S, 1e-9)
                        g.add(self.dot(fr.p(e.x, v), color) if closed
                              else self.hollow(fr.p(e.x, v), color))
                if e.val is not None and fr.inside(e.x, e.val) and not any(
                        s == "f" and abs(e.val - v) <= 2e-3 * max(scan.S, 1e-9)
                        for s, v in ((e.ls, e.lv), (e.rs, e.rv))):
                    g.add(self.dot(fr.p(e.x, e.val), color))
            return g

        def graph(self, fr, fn, color, scan=None, width=5):
            """Kurve samt Markierungen als Gruppe; gibt (Gruppe, Abtastung) zurück."""
            lo, hi = fr.xlim
            scan = scan or _scan(fn, lo, hi)
            g = VGroup(fr.curve(_pieces(scan, fr.ylim), color, width),
                       self.markers(fr, scan, color))
            return g, scan

        def safe(self, fn, x, default=None):
            y = fn(float(x))
            return float(y) if np.isfinite(y) and abs(y) < 1e9 else default

    # ======================================================================
    class SecantToTangentScene(AnaScene):
        """Sekante -> Tangente: h wird kleiner, die Steigung läuft mit."""

        def construct(self):
            th, job, fn = self.th, self.job, self.fn
            x0 = float(job.x0)
            lo, hi = self.xlim
            y0 = self.safe(fn, x0)
            scan = _scan(fn, lo, hi)
            hs = [h for h in job.hs if self.safe(fn, x0 + h) is not None] or [1.0]
            ylim = tuple(job.ylim) if job.ylim else _auto_ylim(
                scan, include=[y0] + [self.safe(fn, x0 + h) for h in hs])
            fr = Frame((lo, hi), ylim, 8.2, 5.0, (-2.45, -0.25), trig=fn.trig, font=self.font)
            title = self.headline(job.title or "Von der Sekante zur Tangente")

            self.play(Write(title), FadeIn(fr.axes(th)), run_time=1.4)
            graph, _ = self.graph(fr, fn, th.accent, scan)
            flab = self.T("f", 28, th.accent).move_to(fr.p(lo, ylim[1]) + RIGHT * 0.5 + DOWN * 0.3)
            self.play(Create(graph), FadeIn(flab), run_time=2.0)

            p0 = self.dot(fr.p(x0, y0), th.tangent, 0.11)
            p0lab = self.T("P₀", 26, th.tangent).next_to(p0, UP + LEFT, buff=0.08)
            self.play(GrowFromCenter(p0), FadeIn(p0lab), run_time=0.8)
            self.say("Wir suchen die Steigung der Kurve im Punkt P₀.", wait=0.8)

            h = ValueTracker(hs[0])

            def yq(hv):
                return self.safe(fn, x0 + hv, y0)

            def slope(hv):
                return (yq(hv) - y0) / hv

            secant = always_redraw(lambda: fr.line(x0, y0, slope(h.get_value()), th.secant, 4))
            qdot = always_redraw(lambda: self.dot(
                fr.p(x0 + h.get_value(), yq(h.get_value())), th.secant, 0.1))
            qlab = always_redraw(lambda: self.T("Q", 26, th.secant).next_to(
                fr.p(x0 + h.get_value(), yq(h.get_value())), UP + RIGHT, buff=0.08))

            def triangle():
                hv = h.get_value()
                a = fr.p(x0, y0)
                b = fr.p(x0 + hv, y0)
                c = fr.p(x0 + hv, yq(hv))
                g = VGroup(Polygon(a, b, c, fill_color=th.secant, fill_opacity=0.16,
                                   stroke_width=0),
                           DashedLine(a, b, dash_length=0.1, color=th.secant, stroke_width=3),
                           DashedLine(b, c, dash_length=0.1, color=th.secant, stroke_width=3))
                if abs(c[1] - b[1]) > 0.35 and abs(b[0] - a[0]) > 0.3:
                    ht = self.T("h", 22, th.secant)
                    ht.next_to((a + b) / 2, DOWN if c[1] > b[1] else UP, buff=0.08)
                    dt = self.T("Δy", 22, th.secant)
                    dt.next_to((b + c) / 2, RIGHT if b[0] >= a[0] else LEFT, buff=0.1)
                    g.add(ht, dt)
                return g

            tri = always_redraw(triangle)
            panel = RIGHT * 4.85

            def slope_text():
                hv = h.get_value()
                t = self.T(f"m = Δy / h = {_num(slope(hv), 3)}", 28, th.secant)
                return t.move_to(panel + UP * 2.0)

            slope_mob = always_redraw(slope_text)
            head = self.T("Steigung", 24, th.muted).move_to(panel + UP * 2.6)
            self.play(Create(secant), FadeIn(qdot), FadeIn(qlab), FadeIn(tri),
                      FadeIn(head), FadeIn(slope_mob), run_time=1.4)
            self.say("Die Sekante geht durch P₀ und Q = (x₀ + h | f(x₀ + h)).", wait=1.0)

            rows = VGroup()
            table_left = np.array([2.75, 1.3, 0.0])

            def add_row(hv):
                r = self.T(f"h = {_num(hv, 3)}   →   m = {_num(slope(hv), 3)}", 23)
                rows.add(r)
                rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
                rows.align_to(table_left, LEFT).align_to(table_left, UP)
                return r

            add_row_first = add_row(hs[0])
            self.play(FadeIn(add_row_first), run_time=0.5)
            self.say("Jetzt lassen wir h kleiner werden …", wait=0.3)
            for hv in hs[1:]:
                self.play(h.animate.set_value(hv), run_time=1.7, rate_func=smooth)
                r = add_row(hv)
                self.play(FadeIn(r), run_time=0.4)
                self.wait(0.35)

            # Grenzlage: die Tangente
            m_t, bad = _slope(fn, _deriv(fn), x0)
            tiny = hs[-1] / 20.0
            self.play(h.animate.set_value(tiny), run_time=1.2, rate_func=smooth)
            self.wait(0.3)
            sec_now = fr.line(x0, y0, slope(h.get_value()), th.secant, 4)
            for m in (secant, qdot, qlab, tri, slope_mob):
                m.clear_updaters()
            self.remove(secant, qdot, qlab, tri, slope_mob)
            self.add(sec_now)
            if m_t is not None:
                tangent = fr.line(x0, y0, m_t, th.tangent, 6)
                result = self.T(f"h → 0:  m → f′(x₀) = {_num(m_t, 3)}", 26, th.tangent)
                result.move_to(panel + UP * 2.0)
                self.play(ReplacementTransform(sec_now, tangent), FadeOut(qlab),
                          FadeOut(qdot), FadeOut(tri), run_time=1.4)
                self.play(Write(result), run_time=1.0)
                self.say("Die Tangente ist die Grenzlage der Sekanten.", wait=0.8)
                self.play(Indicate(tangent, color=th.tangent, scale_factor=1.0), run_time=1.2)
            else:
                result = self.T("f ist bei x₀ nicht differenzierbar.", 28, th.tangent)
                result.move_to(panel + UP * 2.0)
                self.play(FadeOut(qlab), FadeOut(qdot), FadeOut(tri), FadeOut(sec_now),
                          Write(result), run_time=1.2)
                self.say("Die Sekantensteigungen laufen nicht gegen einen Wert: keine Tangente.",
                         wait=0.8)
            self.wait(1.5)

    # ======================================================================
    class DerivativeGraphScene(AnaScene):
        """Tangente wandert entlang f; daneben entsteht der Graph von f'."""

        def construct(self):
            th, job, fn = self.th, self.job, self.fn
            lo, hi = self.xlim
            d1 = _deriv(fn)
            d2 = _deriv(d1)
            sc0, sc1 = _scan(fn, lo, hi), _scan(d1, lo, hi)
            sc2 = _scan(d2, lo, hi, n=1200)
            crit, _infl = _analyse(fn, d1, d2, sc0, sc1, sc2)
            yl0 = tuple(job.ylim) if job.ylim else _auto_ylim(sc0, include=[p.y for p in crit])
            yl1 = _auto_ylim(sc1, include=[0])
            fl = Frame((lo, hi), yl0, 6.1, 4.3, (-3.55, -0.15), trig=fn.trig, font=self.font)
            fr = Frame((lo, hi), yl1, 6.1, 4.3, (3.55, -0.15), trig=fn.trig, font=self.font)

            title = self.headline(job.title or "Die Ableitung als Steigungsfunktion")
            lab_f = self.T("f(x)", 28, th.accent).next_to(fl.p(lo, yl0[1]), UP + RIGHT, buff=0.12)
            lab_d = self.T("f′(x) = Steigung von f", 28, th.derivative).next_to(
                fr.p(lo, yl1[1]), UP + RIGHT, buff=0.12)
            self.play(Write(title), FadeIn(fl.axes(th)), FadeIn(fr.axes(th)), run_time=1.4)
            self.play(FadeIn(lab_f), FadeIn(lab_d), run_time=0.6)
            gf, _ = self.graph(fl, fn, th.accent, sc0)
            self.play(Create(gf), run_time=2.0)

            span = hi - lo
            t = ValueTracker(lo + 0.03 * span)

            # f' wächst mit: der Teil links von t
            d1_pieces = _pieces(sc1, yl1)

            def derivative_so_far():
                tv = t.get_value()
                g = VGroup()
                for sx, sy in d1_pieces:
                    n = int(np.searchsorted(sx, tv, side="right"))
                    if n >= 2:
                        m = VMobject(color=th.derivative, stroke_width=5)
                        m.set_points_as_corners(fr.pts(sx[:n], sy[:n]))
                        g.add(m)
                return g

            def slope_now():
                v = d1(float(t.get_value()))
                return float(v) if np.isfinite(v) and abs(v) < 1e6 else None

            def tangent_now():
                tv = t.get_value()
                y = self.safe(fn, tv)
                m = slope_now()
                if y is None or m is None:
                    return VGroup()
                color = th.good if m > 1e-6 else (th.bad if m < -1e-6 else th.muted)
                w = 0.17 * span
                return fl.line(tv, y, m, color, 6, xrange=(tv - w, tv + w))

            def left_dot():
                tv = t.get_value()
                y = self.safe(fn, tv)
                return VGroup() if y is None or not fl.inside(tv, y) else \
                    self.dot(fl.p(tv, y), th.tangent, 0.1)

            def right_dot():
                tv = t.get_value()
                m = slope_now()
                if m is None or not fr.inside(tv, m):
                    return VGroup()
                return VGroup(
                    DashedLine(fr.p(tv, 0), fr.p(tv, m), dash_length=0.09,
                               color=th.derivative, stroke_width=3),
                    self.dot(fr.p(tv, m), th.derivative, 0.1))

            def readout():
                m = slope_now()
                tv = t.get_value()
                txt = f"x = {_num(tv)}      f′(x) = {_num(m, 2) if m is not None else 'nicht definiert'}"
                col = th.text if m is None else (
                    th.good if m > 1e-6 else (th.bad if m < -1e-6 else th.muted))
                return self.T(txt, 26, col).move_to(RIGHT * 3.55 + DOWN * 2.85)

            tangent_m = always_redraw(tangent_now)
            ldot_m = always_redraw(left_dot)
            rdot_m = always_redraw(right_dot)
            deriv_m = always_redraw(derivative_so_far)
            read_m = always_redraw(readout)
            self.add(tangent_m, ldot_m, rdot_m, deriv_m, read_m)
            self.wait(0.3)
            self.say("Grün: die Tangente steigt, f′ > 0.   Rot: sie fällt, f′ < 0.", 24, wait=0.4)

            stops = [p for p in sorted(crit, key=lambda q: q.x)
                     if lo + 0.03 * span < p.x < hi - 0.03 * span][:5]
            targets = [p.x for p in stops] + [hi - 0.03 * span]
            total = max(targets[-1] - t.get_value(), 1e-9)
            for k, target in enumerate(targets):
                dist = abs(target - t.get_value())
                self.play(t.animate.set_value(target),
                          run_time=max(0.9, 12.0 * dist / total), rate_func=linear)
                if k < len(stops):
                    note = self.T("waagrechte Tangente:  f′(x) = 0", 26, th.muted)
                    note.next_to(fl.p(stops[k].x, stops[k].y), UP, buff=0.35)
                    self.play(FadeIn(note, shift=UP * 0.1), run_time=0.4)
                    self.wait(0.7)
                    self.play(FadeOut(note), run_time=0.3)
            self.say("f′ sagt an jeder Stelle, wie steil f dort ist.", wait=2.0)

    # ======================================================================
    class CurveAnalysisScene(AnaScene):
        """Kurvendiskussion: Monotonie, kritische Punkte, Krümmung."""

        def construct(self):
            th, job, fn = self.th, self.job, self.fn
            lo, hi = self.xlim
            d1 = _deriv(fn)
            d2 = _deriv(d1)
            sc0, sc1, sc2 = _scan(fn, lo, hi), _scan(d1, lo, hi), _scan(d2, lo, hi)
            crit, infl = _analyse(fn, d1, d2, sc0, sc1, sc2)
            ylim = tuple(job.ylim) if job.ylim else _auto_ylim(
                sc0, include=[p.y for p in crit + infl], pad=0.14)
            fr = Frame((lo, hi), ylim, 8.3, 5.0, (-2.0, -0.1), trig=fn.trig, font=self.font)
            title = self.headline(job.title or "Kurvenanalyse")

            self.play(Write(title), FadeIn(fr.axes(th)), run_time=1.4)
            graph, _ = self.graph(fr, fn, th.accent, sc0)
            self.play(Create(graph), run_time=2.2)
            flab = self.T("f(x)", 28, th.accent).next_to(fr.p(lo, ylim[1]), DOWN + RIGHT, buff=0.15)
            self.play(FadeIn(flab), run_time=0.4)

            def band(a, b, color, opacity=0.2):
                pa, pb = fr.p(max(a, lo), ylim[0]), fr.p(min(b, hi), ylim[1])
                r = Polygon(pa, [pb[0], pa[1], 0], pb, [pa[0], pb[1], 0],
                            fill_color=color, fill_opacity=opacity, stroke_width=0)
                r.set_z_index(-1)
                return r

            # 1. Monotonie
            roots1 = _roots(d1, sc1)
            iv1 = _sign_intervals(d1, sc1, roots1)
            bands = VGroup(*[band(a, b, th.good if s > 0 else th.bad)
                             for a, b, s in iv1 if s])
            self.say("Monotonie:  f′ > 0 → f steigt (grün),  f′ < 0 → f fällt (rot).", 24)
            self.play(FadeIn(bands), run_time=1.6)
            self.wait(1.0)

            # 2. Kritische Punkte
            name = {"max": "Hochpunkt", "min": "Tiefpunkt", "sat": "Sattelpunkt",
                    "wen": "Wendepunkt"}
            colour = {"max": th.hi, "min": th.lo, "sat": th.sat, "wen": th.wen}
            letter = {"max": "H", "min": "T", "sat": "S", "wen": "W"}
            crit_marks = []
            if crit:
                self.say("Kritische Punkte:  f′(x) = 0 → waagrechte Tangente.", 24)
            half = 0.05 * (hi - lo)
            for p in crit:
                if not fr.inside(p.x, p.y):
                    continue
                tang = Line(fr.p(p.x - half, p.y), fr.p(p.x + half, p.y),
                            color=th.tangent, stroke_width=5)
                dot = self.dot(fr.p(p.x, p.y), colour[p.kind], 0.13)
                lab = self.T(letter[p.kind], 24, colour[p.kind], weight="BOLD")
                up = p.kind in ("max", "sat")
                lab.next_to(fr.p(p.x, p.y), UP if up else DOWN, buff=0.16)
                self.play(Create(tang), GrowFromCenter(dot), FadeIn(lab, shift=UP * 0.1),
                          run_time=0.9)
                crit_marks.append(VGroup(tang, dot, lab))
                self.wait(0.3)
            self.wait(0.6)

            # 3. Krümmung
            roots2 = _roots(d2, sc2)
            iv2 = _sign_intervals(d2, sc2, roots2)
            bands2 = VGroup(*[band(a, b, th.curved_left if s > 0 else th.curved_right, 0.26)
                              for a, b, s in iv2 if s])
            self.say("Krümmung:  f″ > 0 → linksgekrümmt (blau),  f″ < 0 → rechtsgekrümmt (gelb).", 22)
            self.play(FadeOut(bands), FadeIn(bands2), run_time=1.6)
            self.wait(0.8)
            if infl:
                self.say("Wendepunkte:  f″ wechselt das Vorzeichen.", 24)
            for p in infl:
                if not fr.inside(p.x, p.y):
                    continue
                dot = self.dot(fr.p(p.x, p.y), th.wen, 0.13)
                lab = self.T(letter["wen"], 24, th.wen, weight="BOLD")
                lab.next_to(fr.p(p.x, p.y), DOWN, buff=0.16)
                self.play(GrowFromCenter(dot), FadeIn(lab, shift=UP * 0.1), run_time=0.9)
                crit_marks.append(VGroup(dot, lab))
                self.wait(0.3)

            # 4. Zusammenfassung rechts
            lines = [self.T("Zusammenfassung", 26, th.muted, weight="BOLD")]
            for p in crit + infl:
                lines.append(self.T(f"{letter[p.kind]}  {name[p.kind]}: ({_num(p.x)} | {_num(p.y)})",
                                    20, colour[p.kind]))
            if len(lines) == 1:
                lines.append(self.T("keine kritischen Punkte", 20))
            lines = lines[:10]
            lines.append(self.T("■ f″ > 0: linksgekrümmt", 18, th.curved_left))
            lines.append(self.T("■ f″ < 0: rechtsgekrümmt", 18, th.curved_right))
            summary = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            if summary.width > 4.3:
                summary.scale(4.3 / summary.width)
            summary.move_to(RIGHT * 4.9 + UP * 0.2)
            summary.align_to(RIGHT * 2.8, LEFT)
            self.say("Hoch-, Tief-, Sattel- und Wendepunkte im Überblick.", 24)
            self.play(FadeIn(summary, shift=LEFT * 0.2), run_time=1.0)
            self.wait(2.2)

    # ======================================================================
    class LimitApproachScene(AnaScene):
        """Grenzwert: x nähert sich x0 von links und von rechts, f(x) läuft mit."""

        def construct(self):
            th, job, fn = self.th, self.job, self.fn
            x0 = float(job.x0)
            lo, hi = self.xlim
            scan = _scan(fn, lo, hi)
            hs = sorted(set(abs(h) for h in job.hs), reverse=True)
            ys = [self.safe(fn, x0 + sg * h) for h in hs for sg in (-1, 1)]
            ylim = tuple(job.ylim) if job.ylim else _auto_ylim(
                scan, include=[v for v in ys if v is not None])
            fr = Frame((lo, hi), ylim, 8.2, 5.0, (-2.45, -0.25), trig=fn.trig, font=self.font)
            title = self.headline(job.title or "Der Grenzwert: x nähert sich x₀")

            self.play(Write(title), FadeIn(fr.axes(th)), run_time=1.4)
            graph, _ = self.graph(fr, fn, th.accent, scan)
            flab = self.T("f", 28, th.accent).move_to(fr.p(lo, ylim[1]) + RIGHT * 0.5 + DOWN * 0.3)
            self.play(Create(graph), FadeIn(flab), run_time=2.0)

            vline = DashedLine(fr.p(x0, ylim[0]), fr.p(x0, ylim[1]), dash_length=0.12,
                               color=th.muted, stroke_width=2.5)
            x0lab = self.T("x₀", 24, th.muted).next_to(fr.p(x0, ylim[0]), DOWN, buff=0.45)
            self.play(Create(vline), FadeIn(x0lab), run_time=0.8)
            self.say("Wir nähern uns der Stelle x₀ — von links und von rechts.", wait=0.6)

            h = ValueTracker(hs[0])
            side_color = {-1: th.secant, 1: th.tangent}

            def probe(sg):
                def build():
                    hv = h.get_value()
                    y = self.safe(fn, x0 + sg * hv)
                    if y is None or not fr.inside(x0 + sg * hv, y):
                        return VGroup()
                    p = fr.p(x0 + sg * hv, y)
                    foot = fr.p(x0 + sg * hv, ylim[0])
                    return VGroup(DashedLine(foot, p, dash_length=0.1, color=side_color[sg],
                                             stroke_width=2.5),
                                  self.dot(p, side_color[sg], 0.1))
                return always_redraw(build)

            left, right = probe(-1), probe(1)
            self.play(FadeIn(left), FadeIn(right), run_time=0.8)

            panel = RIGHT * 4.85
            head = self.T("x → x₀", 24, th.muted).move_to(panel + UP * 2.6)
            self.play(FadeIn(head), run_time=0.4)
            rows = VGroup()
            table_left = np.array([2.75, 2.0, 0.0])

            def add_row(hv):
                fl, fr_ = self.safe(fn, x0 - hv), self.safe(fn, x0 + hv)
                a = "—" if fl is None else _num(fl, 4)
                b = "—" if fr_ is None else _num(fr_, 4)
                r = self.T(f"h = {_num(hv, 3)}:  {a}  |  {b}", 21)
                rows.add(r)
                rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
                rows.align_to(table_left, LEFT).align_to(table_left, UP)
                return r

            sub = self.T("links  |  rechts", 18, th.muted).move_to(panel + UP * 2.25)
            self.play(FadeIn(sub), FadeIn(add_row(hs[0])), run_time=0.5)
            self.say("Jetzt wird der Abstand h zu x₀ immer kleiner …", wait=0.3)
            for hv in hs[1:]:
                self.play(h.animate.set_value(hv), run_time=1.5, rate_func=smooth)
                r = add_row(hv)
                self.play(FadeIn(r), run_time=0.4)
                self.wait(0.3)

            tiny = hs[-1] / 20.0
            lv, rv = self.safe(fn, x0 - tiny), self.safe(fn, x0 + tiny)
            big = lambda v: v is None or abs(v) > 1e4
            if big(lv) or big(rv):
                text, col = "Die Funktionswerte wachsen über alle Grenzen: kein endlicher Grenzwert.", th.bad
            elif abs(lv - rv) > 1e-2 * max(1.0, abs(lv), abs(rv)):
                text, col = (f"Links {_num(lv, 3)}, rechts {_num(rv, 3)}: "
                             "die Seiten stimmen nicht überein, kein Grenzwert."), th.bad
            else:
                L = (lv + rv) / 2
                text, col = f"Beide Seiten laufen gegen L = {_num(L, 3)}.", th.good
                if fr.inside(x0, L):
                    hline = DashedLine(fr.p(lo, L), fr.p(hi, L), dash_length=0.12,
                                       color=th.good, stroke_width=3)
                    self.play(h.animate.set_value(tiny), run_time=1.2, rate_func=smooth)
                    self.play(Create(hline), run_time=0.8)
                    self.play(Indicate(self.hollow(fr.p(x0, L), th.good), color=th.good),
                              run_time=0.8)
            if col is th.bad:
                self.play(h.animate.set_value(tiny), run_time=1.2, rate_func=smooth)
            self.say(text, 24, col, wait=0.8)
            self.wait(1.5)

    return {"secant_to_tangent": SecantToTangentScene,
            "derivative_graph": DerivativeGraphScene,
            "curve_analysis": CurveAnalysisScene,
            "limit_approach": LimitApproachScene}


# --------------------------------------------------------------------------
# Rendern
# --------------------------------------------------------------------------

#: Wird vom Kindprozess ausgeführt. Lädt den Auftrag aus der Umgebung.
#: Manim sucht Szenenklassen nur in diesem Modul, darum wird __module__
#: der hier erzeugten Klassen umgehängt.
_RUNNER = '''
import importlib, importlib.util, os, sys
# Der Ordner ist das Paket, auch wenn er nicht so heisst (z. B. »Analysiskit«).
if {pkg!r} not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        {pkg!r}, os.path.join({here!r}, "__init__.py"),
        submodule_search_locations=[{here!r}])
    _p = importlib.util.module_from_spec(_spec)
    sys.modules[{pkg!r}] = _p
    _spec.loader.exec_module(_p)
_mod = importlib.import_module({pkg!r} + ".anim")

for _key, _cls in _mod._build_scenes().items():
    _cls.__module__ = __name__
    globals()[_mod.SCENE_NAMES[_key]] = _cls
'''

_PROGRESS = re.compile(r"(\d+)\s*%")


def _package() -> tuple[str, str]:
    """(Name des Pakets, Ordner, der es enthält) — damit der Kindprozess es importieren kann."""
    here = Path(__file__).resolve().parent
    return (__package__ or here.name).split(".")[0], str(here.parent)


def render(job: AnimationJob, *, progress: Callable[[int, str], None] | None = None,
           out_dir: str | Path | None = None, timeout: float = 900) -> Path:
    """
    Rendert den Auftrag und gibt den Pfad der fertigen Datei zurück (MP4,
    bei job.fmt == 'gif' ein GIF).

    Manim läuft in einem eigenen Prozess: Es schreibt viel in globale
    Konfiguration und lässt sich im selben Prozess nicht sauber mehrfach
    starten. progress bekommt (Prozent, Text). Fehlt Manim, gibt es einen
    RuntimeError mit Installationshinweis.
    """
    ok, why = manim_available()
    if not ok:
        raise RuntimeError(f"{why}\nInstallation:\n{INSTALL_HINT}")
    if job.scene not in SCENE_NAMES:
        raise ValueError(f"Unbekannte Szene {job.scene!r}")

    pkg, root = _package()
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="anakit-anim-"))
    work.mkdir(parents=True, exist_ok=True)
    script = work / "scene_runner.py"
    script.write_text(_RUNNER.format(here=str(Path(__file__).resolve().parent), pkg=pkg), encoding="utf-8")

    folder, flag = QUALITIES.get(QUALITY_ALIASES.get(job.quality, job.quality),
                                 QUALITIES["low"])
    name = SCENE_NAMES[job.scene]
    stem = f"{job.scene}"
    # Immer MP4 rendern. Manims eigener GIF-Export erzeugt eine fehlerhafte
    # Palette (schwarze Schrift wird gelb); das GIF entsteht darum unten in
    # einem zweiten Schritt mit palettegen.
    cmd = [sys.executable, "-m", "manim", flag, "--disable_caching",
           "--media_dir", str(work / "media"), "-o", stem, str(script), name]

    env = dict(os.environ)
    env["ANAKIT_JOB"] = job.to_json()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    if progress:
        progress(2, "Manim startet …")

    process = subprocess.Popen(cmd, cwd=str(work), env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=1)
    # Manim zählt je Animation von 0 bis 100. Für einen Balken, der nicht
    # zurückspringt, zählen wir die abgeschlossenen Animationen mit.
    expected = _EXPECTED.get(job.scene, 12) + (
        2 * len(job.hs) if job.scene in ("secant_to_tangent", "limit_approach") else 0)
    log: list[str] = []
    done = last = 0
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, stop)
    timer.start()
    try:
        for line in process.stdout:
            log.append(line)
            for match in _PROGRESS.finditer(line):
                within = int(match.group(1))
                if within >= 100:
                    done += 1
                overall = min(97, int((done + within / 100) / expected * 95) + 3)
                if overall > last and progress:
                    last = overall
                    progress(overall, "Rendere …")
        process.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise RuntimeError("Das Rendern hat zu lange gedauert.")

    output = "".join(log)
    # Manim meldet manche Fehler nur im Text und beendet trotzdem mit 0.
    if process.returncode != 0 or "Traceback" in output:
        lines = [l.rstrip() for l in log if l.strip()]
        tail = "\n".join(lines[-12:]) or "Keine Ausgabe."
        raise RuntimeError(f"Manim ist fehlgeschlagen:\n{tail}")

    candidates = sorted((work / "media").rglob(f"{stem}*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Manim hat keine Datei erzeugt.")
    video = candidates[0]

    if job.fmt == "gif":
        if progress:
            progress(98, "Erzeuge GIF …")
        video = to_gif(video, work / f"{stem}.gif", width=job.gif_width, fps=job.gif_fps)
    if progress:
        progress(100, "Fertig.")
    return video


def to_gif(video: Path, target: Path, *, width: int = 760, fps: int = 15) -> Path:
    """
    Wandelt das Video in ein GIF mit eigener Palette um.

    ffmpeg erzeugt die Palette aus dem Material selbst (palettegen), sonst
    landet die Schrift auf einer Standardpalette und verfärbt sich.
    Hochskaliert wird nie: ist das Video schmaler als `width`, bleibt es so,
    sonst wäre das Ergebnis nur unscharf aufgeblasen.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg fehlt — nötig für die GIF-Ausgabe.")
    chain = (f"fps={fps},scale='min({width},iw)':-1:flags=lanczos,"
             f"split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
             f"[s1][p]paletteuse=dither=bayer:bayer_scale=3")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", chain, str(target)],
        capture_output=True, text=True)
    if result.returncode != 0 or not Path(target).exists():
        raise RuntimeError(f"GIF-Umwandlung fehlgeschlagen:\n"
                           f"{result.stderr.strip()[-400:]}")
    return Path(target)


# --------------------------------------------------------------------------
# Beispiel von der Kommandozeile
# --------------------------------------------------------------------------

_DEMO_F = {"secant_to_tangent": "x^2", "derivative_graph": "sin(x)",
           "curve_analysis": "x^3 - 3x", "limit_approach": "(x^2-1)/(x-1)"}


def _demo(argv: list[str]) -> int:
    scene = argv[0] if argv else "secant_to_tangent"
    ok, why = manim_available()
    if not ok:
        print(why)
        print(INSTALL_HINT)
        return 1
    f = argv[1] if len(argv) > 1 else _DEMO_F.get(scene, "x^2")
    x0 = float(argv[2]) if len(argv) > 2 else 1.0
    job = build_job(scene, f, x0=x0 if scene in ("secant_to_tangent", "limit_approach") else None,
                    quality=os.environ.get("ANAKIT_QUALITY", "low"))
    path = render(job, progress=lambda p, t: print(f"{p:3d}%  {t}"))
    print("Fertig:", path)
    return 0


if __name__ == "__main__":
    if not __package__:
        # Direkt gestartet: das Paket über seinen Ordner importierbar machen.
        _here = Path(__file__).resolve().parent
        sys.path.insert(0, str(_here.parent))
        import importlib
        sys.exit(importlib.import_module(_here.name + ".anim")._demo(sys.argv[1:]))
    sys.exit(_demo(sys.argv[1:]))
