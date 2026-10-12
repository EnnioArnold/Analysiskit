#!/usr/bin/env python3
"""
Zeichnen: Funktionen, Sekanten, Tangenten, Ableitungen und Grenzwerte.

Alles mit Matplotlib (Agg-tauglich, nie plt.show) — das passt zu euren
Notebooks und lässt sich drucken. Jede Funktion gibt eine
matplotlib.figure.Figure zurück. Plotly gibt es zusätzlich für die
interaktive Variante (`plotly_function`); fehlt es, bleibt alles andere
benutzbar und nur diese eine Funktion meldet es klar.

Die Zeichnungen kommen mit Unstetigkeiten zurecht: Polstellen werden nicht
überbrückt (gestrichelte Asymptote), Sprünge und Lücken bekommen offene und
geschlossene Kreise.

    plot_function("tan(x)")
    plot_secant("x^2", -1, 3)
    plot_tangent("1/x", 1)
    plot_curve("x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27", (-1.5, 4.5))
    plot_limit("x^2 if x<1 else x-1", 1)
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import sympy as sp

from .core import fmt, parse, to_callable, x as X_SYMBOL

__all__ = [
    "plot_function", "plot_secant", "plot_secants_to_tangent", "plot_tangent",
    "plot_derivative", "plot_curve", "plot_limit",
    "has_plotly", "plotly_function", "save_html", "COLORS",
]

# Dieselbe Farbfolge wie in lialkit, damit die Hefte zusammengehören.
COLORS = ["#0071e3", "#ff9500", "#248a3d", "#d70015", "#8944ab", "#00a2a8"]

_LIGHT = dict(
    dark=False, bg="#ffffff", fg="#1d1d1f", muted="#86868b", grid="#e5e5ea",
    spine="#d2d2d7", axis="#6e6e73",
    f="#0071e3", d1="#8944ab", d2="#00a2a8", sec="#ff9500", sec2="#b34700",
    tan="#d70015", pt="#248a3d", pos="#248a3d", neg="#d70015",
    hi="#ff9500", lo="#8944ab", sat="#00a2a8", wen="#c9a000")
_DARK = dict(
    dark=True, bg="#1c1c1e", fg="#f5f5f7", muted="#8e8e93", grid="#2f2f32",
    spine="#48484a", axis="#a1a1a6",
    f="#0a84ff", d1="#bf5af2", d2="#40c8e0", sec="#ff9f0a", sec2="#ff6a1a",
    tan="#ff453a", pt="#30d158", pos="#30d158", neg="#ff453a",
    hi="#ff9f0a", lo="#bf5af2", sat="#40c8e0", wen="#ffd60a")


def _theme(dark: bool) -> dict:
    th = dict(_DARK if dark else _LIGHT)
    th["rc"] = {
        "figure.facecolor": th["bg"], "axes.facecolor": th["bg"],
        "savefig.facecolor": th["bg"], "text.color": th["fg"],
        "axes.labelcolor": th["fg"], "xtick.color": th["axis"],
        "ytick.color": th["axis"], "axes.edgecolor": th["spine"],
        "font.size": 10.5, "axes.titlesize": 12, "axes.unicode_minus": True,
        "mathtext.fontset": "dejavusans", "legend.facecolor": th["bg"],
        "legend.edgecolor": th["spine"], "figure.dpi": 100,
        "axes.formatter.use_mathtext": False,
    }
    return th


def _finish(fig):
    """Figur zurückgeben, ohne dass Jupyter sie doppelt anzeigt."""
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


# ==========================================================================
# Zahlen und Texte
# ==========================================================================

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
INF = float("inf")


def _num(v, nd: int = 2) -> str:
    """2.0 -> '2', 1.2345 -> '1.23', -0.5 -> '−0.5', 1e9 -> '∞'."""
    v = float(v)
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    if math.isnan(v):
        return "?"
    if abs(v - round(v)) < 10 ** (-nd - 1):
        s = str(int(round(v)))
    else:
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.replace("-", "−")


def _pretty(expr) -> str:
    """Term als lesbarer Text mit hochgestellten Exponenten."""
    if isinstance(expr, sp.Piecewise):
        parts = []
        for e, c in expr.args:
            if c is True or c == sp.true:
                parts.append(f"{_pretty(e)} sonst")
            else:
                cond = sp.sstr(c).replace("**", "^").replace("<=", "≤").replace(">=", "≥")
                cond = re.sub(r"\^\(?(-?\d+)\)?", lambda m: m.group(1).translate(_SUP), cond)
                parts.append(f"{_pretty(e)} für {cond.replace('-', '−')}")
        return "; ".join(parts)
    s = fmt(expr)
    s = re.sub(r"\^\(?(-?\d+)\)?", lambda m: m.group(1).translate(_SUP), s)
    s = re.sub(r"\bpi\b", "π", s).replace("sqrt(", "√(")
    s = s.replace("-", "−")
    return s


def _xval(v, what: str = "x0"):
    """Zahl, Bruch oder Term wie 'pi/2' -> (float, Beschriftung)."""
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v), _num(float(v))
    try:
        e = parse(v)
        val = float(e)
    except Exception:
        raise ValueError(f"{what} muss eine Zahl sein (z. B. 1, 0.5 oder 'pi/2'), "
                         f"nicht {v!r}") from None
    if not math.isfinite(val):
        raise ValueError(f"{what} muss eine endliche Zahl sein")
    return val, (str(int(e)) if e.is_Integer else _pretty(e))


def _xlim(xlim, default=None):
    if xlim is None:
        return default
    try:
        lo, hi = float(xlim[0]), float(xlim[1])
    except Exception:
        raise ValueError("xlim muss ein Paar (links, rechts) sein") from None
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ValueError("xlim muss ein Paar (links, rechts) mit links < rechts sein")
    return lo, hi


# ==========================================================================
# Funktionen auswerten
# ==========================================================================

class _Fn:
    """Eine Funktion von x: ruft sicher mit Arrays auf, liefert nan statt Fehler."""

    def __init__(self, f=None, *, call: Callable | None = None, expr=None):
        if isinstance(f, _Fn):
            self.expr, self._call = f.expr, f._call
            return
        if call is not None:
            self.expr, self._call = expr, call
            return
        try:
            self.expr = parse(f)
            self._call = to_callable(self.expr)
        except Exception as exc:
            raise ValueError(f"Die Funktion {f!r} lässt sich nicht lesen: {exc}") from None

    @property
    def trig(self) -> bool:
        return self.expr is not None and self.expr.has(
            sp.sin, sp.cos, sp.tan, sp.cot)

    def __call__(self, xs):
        xs = np.asarray(xs, dtype=float)
        scalar = xs.ndim == 0
        xs = xs.reshape(1) if scalar else xs
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y = np.asarray(self._call(xs))
            except Exception:
                y = np.array([self._one(v) for v in xs.ravel()]).reshape(xs.shape)
            if np.iscomplexobj(y):
                y = np.where(np.abs(y.imag) <= 1e-12 * (1 + np.abs(y.real)),
                             y.real, np.nan)
            try:
                y = np.broadcast_to(y, xs.shape).astype(float)
            except Exception:
                y = np.array([self._one(v) for v in xs.ravel()]).reshape(xs.shape)
        return float(y[0]) if scalar else y

    def _one(self, v):
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = complex(self._call(float(v)))
            return r.real if abs(r.imag) <= 1e-12 * (1 + abs(r.real)) else np.nan
        except Exception:
            return np.nan


def _deriv(fn: _Fn) -> _Fn:
    """Ableitung; symbolisch wenn möglich, sonst mit dem Differenzenquotienten."""
    if fn.expr is not None:
        try:
            e = sp.diff(fn.expr, X_SYMBOL)
            e = e.replace(sp.DiracDelta, lambda *a: sp.Integer(0))
            if not e.has(sp.Derivative, sp.Subs, sp.DiracDelta):
                return _Fn(expr=e, call=to_callable(e))
        except Exception:
            pass

    def numeric(xs):
        xs = np.asarray(xs, dtype=float)
        hh = 1e-5 * (1 + np.abs(xs))
        return (fn(xs + hh) - fn(xs - hh)) / (2 * hh)
    return _Fn(call=numeric)


def _fin(y):
    y = np.asarray(y, dtype=float)
    return np.isfinite(y) & (np.abs(y) < 1e12)


# ==========================================================================
# Abtasten: Pole, Sprünge, Lücken finden
# ==========================================================================

@dataclass
class _Event:
    """
    Eine Stelle, an der die Kurve nicht einfach durchläuft.

    ls/rs: Verhalten links/rechts — 'f' endlicher Grenzwert, 'p' Pol,
    'n' nicht definiert, 'o' schwingt (kein Grenzwert).
    lv/rv: der Grenzwert (bei 'p' ±inf). val: Funktionswert oder None.
    """
    x: float
    ls: str
    lv: float | None
    rs: str
    rv: float | None
    val: float | None

    @property
    def pole(self) -> bool:
        return self.ls == "p" or self.rs == "p"


class _Scan:
    """Abtastung einer Funktion auf [lo, hi] samt Unstetigkeiten."""

    def __init__(self, lo, hi, xs, ys, events, segments, S, M):
        self.lo, self.hi = lo, hi
        self.xs, self.ys = xs, ys
        self.events = events
        self.segments = segments
        self.S, self.M = S, M

    def curve(self):
        xs, ys = [], []
        for sx, sy in self.segments:
            xs += [sx, [np.nan]]
            ys += [sy, [np.nan]]
        if not xs:
            return np.array([]), np.array([])
        return np.concatenate(xs), np.concatenate(ys)

    def values(self):
        """Alle endlichen Funktionswerte der Abtastung (für die Achsengrenzen)."""
        parts = [sy for _sx, sy in self.segments if len(sy)]
        return np.concatenate(parts) if parts else np.array([])


def _snap(c: float) -> float:
    r = round(c, 8) + 0.0
    return float(r if abs(r - c) < 1e-9 * max(1.0, abs(c)) else c)


def _side(fn, c, side, S, M):
    """
    Verhalten von fn links (side=-1) oder rechts (+1) von c.
    Rückgabe (Zustand, Wert); Zustände siehe _Event.
    """
    sc = max(1.0, abs(c))
    d = 10.0 ** -np.arange(4, 10) * sc               # 1e-4 ... 1e-9
    v = fn(c + side * d)
    if not (np.isfinite(v[4]) and np.isfinite(v[5])):
        return "n", None
    a = np.abs(v)
    a2 = a[2] if np.isfinite(a[2]) else 0.0
    if a[5] > 30 * M and a[5] > a[4] and a[5] > 3 * a2:
        return "p", math.copysign(INF, v[5])
    if (np.isfinite(v[3]) and a[5] > a[4] > a[3] and a[5] - a[4] > 0.2 * S
            and np.sign(v[3]) == np.sign(v[4]) == np.sign(v[5])):
        return "p", math.copysign(INF, v[5])
    if np.all(np.isfinite(v)):
        diffs = np.abs(np.diff(v))
        floor = 1e-9 * (1 + abs(v[5]))
        shrinking = all(diffs[k + 1] <= 0.8 * diffs[k] + floor for k in range(len(diffs) - 1))
        if shrinking or diffs[-1] <= 1e-3 * S:
            val = float(v[5])
            r = round(val, 6)
            return "f", (r if abs(val - r) < 2e-7 * (1 + abs(r)) else val)
    return "o", None


def _classify(fn, c, S, M):
    """Event an der Stelle c, oder None wenn die Funktion dort einfach stetig ist."""
    ls, lv = _side(fn, c, -1, S, M)
    rs, rv = _side(fn, c, +1, S, M)
    v = fn(np.array([c]))[0]
    val = float(v) if _fin(v) else None
    tol = max(1e-5 * (1 + abs(lv or 0)), 1e-3 * S)
    if ls == "f" and rs == "f" and val is not None \
            and abs(lv - rv) <= tol and abs(val - lv) <= tol:
        return None
    if ls == "n" and rs == "n":
        return None
    if val is not None and (ls == "p" or rs == "p") and abs(val) > 30 * M:
        val = None          # Rundungsrest direkt am Pol, kein echter Wert
    return _Event(c, ls, lv, rs, rv, val)


def _boundary(fn, a, b):
    """a: Funktionswert endlich, b: nicht -> Stelle des Übergangs (a', b')."""
    sc = max(1.0, abs(a), abs(b))
    for _ in range(100):
        if abs(b - a) < 1e-13 * sc:
            break
        m = 0.5 * (a + b)
        if _fin(fn(np.array([m])))[0]:
            a = m
        else:
            b = m
    return a, b


def _jump_point(fn, a, b):
    """Zwischen zwei endlichen Stellen: dort, wo der Sprung am grössten ist."""
    sc = max(1.0, abs(a), abs(b))
    ya, yb = fn(np.array([a, b]))
    for _ in range(100):
        if abs(b - a) < 1e-13 * sc:
            break
        m = 0.5 * (a + b)
        ym = fn(np.array([m]))[0]
        if not np.isfinite(ym):
            return 0.5 * (a + b)
        if abs(ym - ya) > abs(yb - ym):
            b, yb = m, ym
        else:
            a, ya = m, ym
    return 0.5 * (a + b)


def _peak_point(fn, a, b):
    """Stelle des grössten |f| auf [a,b] (für Pole zwischen Rasterpunkten)."""
    for _ in range(80):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        y1, y2 = np.abs(fn(np.array([m1, m2])))
        if not np.isfinite(y1):
            y1 = INF
        if not np.isfinite(y2):
            y2 = INF
        if y1 < y2:
            a = m1
        else:
            b = m2
    return 0.5 * (a + b)


def _nice_points(lo, hi):
    pts = [np.arange(math.ceil(lo * 2) / 2, hi + 1e-9, 0.5)]
    k0, k1 = math.ceil(lo / (math.pi / 2)), math.floor(hi / (math.pi / 2))
    if k1 - k0 < 400:
        pts.append(np.arange(k0, k1 + 1) * (math.pi / 2))
    return np.concatenate(pts)


def _scan(fn: _Fn, lo: float, hi: float, n: int = 2400) -> _Scan:
    w = hi - lo
    xs = np.unique(np.concatenate([np.linspace(lo, hi, n), _nice_points(lo, hi)]))
    xs = xs[(xs >= lo) & (xs <= hi)]
    ys = fn(xs)
    ok = _fin(ys)
    yv = ys[ok]
    if yv.size >= 3:
        p5, p95 = np.percentile(yv, [5, 95])
        S = max(float(p95 - p5), 1e-6)
        M = max(S, float(np.percentile(np.abs(yv), 90)))
    else:
        S = M = 1.0
    N = len(xs)
    events: list[_Event] = []

    def add(ev):
        if ev is None:
            return
        for e in events:
            if abs(e.x - ev.x) < 1e-7 * max(1.0, abs(ev.x)):
                return
        events.append(ev)

    # 1. Nicht definierte Stellen: Lücken, Pole, Definitionsränder
    i = 0
    while i < N:
        if ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < N and not ok[j + 1]:
            j += 1
        left = right = None
        if i > 0:
            left = _boundary(fn, xs[i - 1], xs[i])          # (a, b)
        if j < N - 1:
            r = _boundary(fn, xs[j + 1], xs[j])             # (a, b) von rechts
            right = (r[1], r[0])
        sc = max(1.0, abs(xs[i]))
        if left and right and abs(left[1] - right[0]) < 1e-9 * sc:
            add(_classify(fn, _snap(xs[i]), S, M))
        else:
            cl = _snap(0.5 * (left[0] + left[1])) if left else None
            cr = _snap(0.5 * (right[0] + right[1])) if right else None
            merged = None
            if left and right and cr - cl < 1e-3 * w:
                # Schmale Lücke: meist ein Pol, dessen Werte über der Grenze liegen
                merged = _classify(fn, _snap(0.5 * (cl + cr)), S, M)
                if merged is not None and not merged.pole:
                    merged = None
            if merged is not None:
                add(merged)
            else:
                if left:
                    add(_classify(fn, cl, S, M))
                if right:
                    add(_classify(fn, cr, S, M))
        i = j + 1

    # 2. Sprünge und Pole zwischen zwei endlichen Rasterpunkten
    both = ok[:-1] & ok[1:]
    dy = np.where(both, np.abs(np.diff(np.where(ok, ys, 0.0))), 0.0)
    med = float(np.median(dy[both])) if both.any() else 0.0
    thr = max(8 * med, 0.01 * S, 1e-9)
    cand = np.argsort(dy)[::-1][:80]
    for k in cand:
        if dy[k] <= thr:
            break
        step = xs[k + 1] - xs[k]
        if any(e.pole and abs(e.x - xs[k]) <= 3 * step for e in events):
            continue
        c = _jump_point(fn, xs[k], xs[k + 1])
        cp = _peak_point(fn, xs[k], xs[k + 1])        # Pol zwischen den Rasterpunkten?
        ends = float(np.max(np.abs(ys[k:k + 2])))
        fp = fn(np.array([cp]))[0]
        if not np.isfinite(fp) or abs(fp) > 10 * max(ends, M):
            c = cp
        add(_classify(fn, _snap(c), S, M))

    # 3. Pole, die genau zwischen zwei gleich grossen Werten liegen (1/x²)
    a = np.where(ok, np.abs(ys), 0.0)
    for k in range(1, N - 1):
        if a[k] > 20 * M and a[k] >= a[k - 1] and a[k] >= a[k + 1]:
            step = xs[k + 1] - xs[k - 1]
            if any(e.pole and abs(e.x - xs[k]) <= 2 * step for e in events):
                continue
            c = _snap(_peak_point(fn, xs[k - 1], xs[k + 1]))
            add(_classify(fn, c, S, M))

    events = [e for e in events if lo - 1e-12 <= e.x <= hi + 1e-12]
    events.sort(key=lambda e: e.x)
    ev_x = [e.x for e in events]

    # Verdichten dort, wo die Kurve nach ±∞ läuft
    extra = []
    for e in events:
        for side, state in ((-1, e.ls), (1, e.rs)):
            if state == "p":
                extra += [e.x + side * w * d for d in (3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-8)]
    if extra:
        xs = np.unique(np.concatenate([xs, np.array(extra)]))
        xs = xs[(xs >= lo) & (xs <= hi)]
    for c in ev_x:
        xs = xs[np.abs(xs - c) > 1e-11 * max(1.0, abs(c))]
    ys = fn(xs)
    ok = _fin(ys)
    ys = np.where(ok, ys, np.nan)

    segments = []
    for k in range(len(events) + 1):
        a_ = ev_x[k - 1] if k > 0 else -INF
        b_ = ev_x[k] if k < len(events) else INF
        sel = (xs > a_) & (xs < b_)
        sx, sy = list(xs[sel]), list(ys[sel])
        if k > 0 and events[k - 1].rs == "f":
            sx.insert(0, a_)
            sy.insert(0, events[k - 1].rv)
        if k < len(events) and events[k].ls == "f":
            sx.append(b_)
            sy.append(events[k].lv)
        sx, sy = np.array(sx), np.array(sy)
        good = np.isfinite(sy)
        if not good.any():
            continue
        # an NaN-Stellen in Stücke teilen
        idx = np.flatnonzero(good)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        for part in np.split(idx, breaks + 1):
            if len(part) >= 2:
                segments.append((sx[part], sy[part]))

    return _Scan(lo, hi, xs, ys, events, segments, S, M)


# ==========================================================================
# Achsen, Grenzen, Zeichenhilfen
# ==========================================================================

def _auto_ylim(scan: _Scan, include=(), pad: float = 0.08):
    y = scan.values()
    inc = [float(v) for v in include if v is not None and np.isfinite(v)]
    pole = any(e.pole for e in scan.events)
    if y.size == 0:
        y = np.array(inc if inc else [-1.0, 1.0])
    ymin, ymax = float(y.min()), float(y.max())
    if pole or (ymax - ymin) > 60 * scan.S:
        p5, p95 = np.percentile(y, [7, 93])
        span = max(float(p95 - p5), 1e-9)
        ymin = max(ymin, float(p5) - 0.5 * span)
        ymax = min(ymax, float(p95) + 0.5 * span)
    if inc:
        ymin, ymax = min(ymin, min(inc)), max(ymax, max(inc))
    if ymax - ymin < 1e-9:
        base = max(abs(ymax), 1.0)
        ymin, ymax = ymin - 0.5 * base, ymax + 0.5 * base
    rng = ymax - ymin
    if ymin > 0 and ymin < 0.75 * rng:
        ymin = 0.0
    elif ymax < 0 and -ymax < 0.75 * rng:
        ymax = 0.0
    rng = ymax - ymin
    return ymin - pad * rng, ymax + pad * rng


def _check_ylim(ylim):
    try:
        lo, hi = float(ylim[0]), float(ylim[1])
    except Exception:
        raise ValueError("ylim muss ein Paar (unten, oben) sein") from None
    if lo >= hi:
        raise ValueError("ylim muss ein Paar (unten, oben) mit unten < oben sein")
    return lo, hi


def _pi_ticks(lo, hi):
    """Beschriftete Stellen in Vielfachen von π/2 (oder π) oder None."""
    w = hi - lo
    if w > 26 or w < 2.5:
        return None
    step = math.pi / 2 if w <= 12 else math.pi
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    ks = list(range(k0, k1 + 1))
    if not ks or len(ks) > 20:
        return None

    def name(k):
        half = k if step < 3 else 2 * k           # in Halben von π
        if half == 0:
            return "0"
        sign = "−" if half < 0 else ""
        a = abs(half)
        if a % 2 == 0:
            n = a // 2
            return f"{sign}{'' if n == 1 else n}π"
        return f"{sign}{'' if a == 1 else a}π/2"
    return [k * step for k in ks], [name(k) for k in ks]


def _trig_ticks(ax, lo, hi):
    t = _pi_ticks(lo, hi)
    if t:
        ax.set_xticks(t[0])
        ax.set_xticklabels(t[1])


def _style(ax, th, lo, hi, trig=False, xlabel="x", ylabel="y"):
    ax.grid(True, color=th["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=th["axis"], linewidth=1.0, zorder=1.5)
    ax.axvline(0, color=th["axis"], linewidth=1.0, zorder=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["spine"])
    ax.set_xlim(lo, hi)
    ax.tick_params(length=3, labelsize=9.5)
    if trig:
        _trig_ticks(ax, lo, hi)
    if xlabel:
        ax.set_xlabel(xlabel, loc="right", labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, loc="top", rotation=0, labelpad=14)


def _open_dot(ax, th, color, px, py, size=7.5, z=6):
    ax.plot([px], [py], "o", ms=size, mfc=th["bg"], mec=color, mew=1.9,
            zorder=z, clip_on=True, ls="none")


def _closed_dot(ax, th, color, px, py, size=7.5, z=6):
    ax.plot([px], [py], "o", ms=size, mfc=color, mec=th["bg"], mew=1.0,
            zorder=z, clip_on=True, ls="none")


def _draw(ax, th, scan: _Scan, color, lw=2.3, label=None, z=3, skip_x=None,
          asymptotes=True):
    """Kurve zeichnen, Pole als Asymptoten, Sprünge/Lücken mit Kreisen."""
    cx, cy = scan.curve()
    if cx.size:
        ax.plot(cx, cy, color=color, lw=lw, label=label, zorder=z,
                solid_capstyle="round", solid_joinstyle="round")
    for e in scan.events:
        if skip_x is not None and abs(e.x - skip_x) < 1e-7:
            continue
        if asymptotes and e.pole and abs(e.x) > 1e-9:
            ax.axvline(e.x, color=th["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1.8)
        tol = 1e-5
        closed_any = False
        for state, v in ((e.ls, e.lv), (e.rs, e.rv)):
            if state != "f":
                continue
            if e.val is not None and abs(e.val - v) <= max(tol * (1 + abs(v)), 2e-3 * scan.S):
                _closed_dot(ax, th, color, e.x, v, z=z + 2)
                closed_any = True
            else:
                _open_dot(ax, th, color, e.x, v, z=z + 2)
        if e.val is not None and not closed_any:
            _closed_dot(ax, th, color, e.x, e.val, z=z + 3)


def _annot(ax, th, px, py, text, dx=8, dy=8, color=None, ha="left", va="bottom",
           size=9.5, z=12, bold=False):
    return ax.annotate(
        text, (px, py), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
        fontsize=size, color=color or th["fg"], zorder=z,
        fontweight="bold" if bold else "normal",
        bbox=dict(boxstyle="round,pad=0.22", fc=th["bg"], ec="none", alpha=0.82))


def _line(ax, m, px, py, lo, hi, **kw):
    ax.plot([lo, hi], [py + m * (lo - px), py + m * (hi - px)], **kw)


def _lighten(color, amount):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _title(ax, th, text, sub=None):
    ax.set_title(text if not sub else f"{text}\n{sub}", fontsize=11.5, loc="left",
                 color=th["fg"], pad=10)


def _at(fn, v, what="x"):
    y = fn(np.array([v]))[0]
    if not _fin(y):
        raise ValueError(f"f ist bei {what} = {_num(v)} nicht definiert "
                         f"(Lücke, Pol oder ausserhalb des Definitionsbereichs)")
    return float(y)


def _legend_out(ax, handles, th, size=9):
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.015, 1.0),
                  frameon=False, fontsize=size, borderaxespad=0, labelcolor=th["fg"])


# ==========================================================================
# plot_function
# ==========================================================================

def _norm_points(points, fn):
    out = []
    for p in points or ():
        label = None
        if isinstance(p, (tuple, list, np.ndarray)):
            if len(p) == 2:
                px, py = _xval(p[0], "Punkt")[0], float(parse(p[1]))
            elif len(p) == 3:
                px, py, label = _xval(p[0], "Punkt")[0], float(parse(p[1])), str(p[2])
            else:
                raise ValueError("Ein Punkt ist x, (x, y) oder (x, y, 'Name')")
        else:
            px = _xval(p, "Punkt")[0]
            py = _at(fn, px)
        out.append((px, py, label))
    return out


def plot_function(f, xlim=(-5, 5), ylim=None, points=(), title="", dark=False):
    """
    Zeichnet den Graphen von f.

    points: x-Werte (der Punkt liegt dann auf dem Graphen), Paare (x, y) oder
    Tripel (x, y, 'Name'). Pole bekommen Asymptoten, Sprünge und Lücken
    offene/geschlossene Kreise. ylim=None wählt die Achse selbst, so dass
    Pole die Zeichnung nicht ins Unlesbare strecken.
    """
    _m, plt = _plt()
    th = _theme(dark)
    lo, hi = _xlim(xlim, (-5, 5))
    fn = _Fn(f)
    pts = _norm_points(points, fn)
    with plt.rc_context(th["rc"]):
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        sc = _scan(fn, lo, hi)
        yl = _check_ylim(ylim) if ylim is not None else \
            _auto_ylim(sc, include=[p[1] for p in pts])
        _draw(ax, th, sc, th["f"])
        for k, (px, py, label) in enumerate(pts):
            _closed_dot(ax, th, th["pt"], px, py, size=8, z=8)
            _annot(ax, th, px, py, label or f"({_num(px)} | {_num(py)})", color=th["pt"],
                   bold=True)
        _style(ax, th, lo, hi, trig=fn.trig, ylabel="y")
        ax.set_ylim(*yl)
        if title:
            _title(ax, th, title)
        elif fn.expr is not None:
            s = _pretty(fn.expr)
            _title(ax, th, f"f(x) = {s}" if len(s) <= 46 else "Graph von f")
        return _finish(fig)


# ==========================================================================
# Sekante
# ==========================================================================

def _secant_xlim(x1, x2):
    a, b = min(x1, x2), max(x1, x2)
    pad = max(0.6 * (b - a), 1.0)
    return a - pad, b + pad


def plot_secant(f, x1, x2, xlim=None, dark=False):
    """
    Zeichnet f, die Punkte P1 und P2, die Sekante und das Steigungsdreieck
    mit Δx und Δy. Im Titel steht die Sekantensteigung m = Δy/Δx.
    """
    _m, plt = _plt()
    th = _theme(dark)
    fn = _Fn(f)
    x1, _l1 = _xval(x1, "x1")
    x2, _l2 = _xval(x2, "x2")
    if x1 == x2:
        raise ValueError("x1 und x2 müssen verschieden sein — sonst gibt es keine "
                         "Sekante (der Differenzenquotient wäre 0/0).")
    y1, y2 = _at(fn, x1, "x1"), _at(fn, x2, "x2")
    lo, hi = _xlim(xlim, None) or _secant_xlim(x1, x2)
    dx, dy = x2 - x1, y2 - y1
    m = dy / dx
    with plt.rc_context(th["rc"]):
        fig, ax = plt.subplots(figsize=(7.4, 5.0), layout="constrained")
        sc = _scan(fn, lo, hi)
        yl = _auto_ylim(sc, include=[y1, y2, y1 + 0.0])
        _draw(ax, th, sc, th["f"])
        _line(ax, m, x1, y1, lo, hi, color=th["sec"], lw=2.0, zorder=4)
        # Steigungsdreieck
        ax.fill([x1, x2, x2], [y1, y1, y2], color=th["sec"], alpha=0.13, lw=0, zorder=2)
        ax.plot([x1, x2], [y1, y1], color=th["sec2"], lw=1.6, ls=(0, (5, 2)), zorder=4.5)
        ax.plot([x2, x2], [y1, y2], color=th["sec2"], lw=1.6, ls=(0, (5, 2)), zorder=4.5)
        below = y2 > y1
        _annot(ax, th, (x1 + x2) / 2, y1, f"Δx = {_num(dx)}", 0, -5 if below else 5,
               color=th["sec2"], ha="center", va="top" if below else "bottom", bold=True)
        right = x2 > x1
        _annot(ax, th, x2, (y1 + y2) / 2, f"Δy = {_num(dy)}", 7 if right else -7, 0,
               color=th["sec2"], ha="left" if right else "right", va="center", bold=True)
        for px, py, name, off in ((x1, y1, "P₁", (-8, 8) if m >= 0 else (-8, -10)),
                                  (x2, y2, "P₂", (8, 8) if m >= 0 else (8, -10))):
            _closed_dot(ax, th, th["sec2"], px, py, size=8, z=8)
            _annot(ax, th, px, py, f"{name}({_num(px)} | {_num(py)})", off[0], off[1],
                   color=th["fg"], ha="right" if off[0] < 0 else "left",
                   va="bottom" if off[1] > 0 else "top", size=9.5)
        _style(ax, th, lo, hi, trig=fn.trig)
        ax.set_ylim(*yl)
        _title(ax, th, "Sekante durch P₁ und P₂")
        ax.set_title(f"Sekante durch P₁ und P₂:  "
                     rf"$m=\frac{{\Delta y}}{{\Delta x}}=\frac{{{_num(y2)}-({_num(y1)})}}"
                     rf"{{{_num(x2)}-({_num(x1)})}}={_num(m)}$",
                     fontsize=11.5, loc="left", color=th["fg"], pad=12)
        return _finish(fig)


# ==========================================================================
# Steigung an einer Stelle
# ==========================================================================

def _slope(fn, dfn, x0):
    """
    (m, None) wenn f bei x0 differenzierbar ist, sonst (None, (links, rechts)).
    Senkrechte Tangenten ergeben (None, (±inf, ±inf)).
    """
    e = 1e-6 * max(1.0, abs(x0))
    y0, yl, yr = fn(np.array([x0, x0 - e, x0 + e]))
    sl, sr = (y0 - yl) / e, (yr - y0) / e
    if not (np.isfinite(sl) and np.isfinite(sr)):
        return None, (sl, sr)
    if abs(sl) > 1e5 and abs(sr) > 1e5:
        return None, (math.copysign(INF, sl), math.copysign(INF, sr))
    if abs(sl - sr) > 1e-3 * (1 + abs(sl) + abs(sr)):
        return None, (float(sl), float(sr))
    mid = 0.5 * (sl + sr)
    an = dfn(np.array([x0]))[0]
    if np.isfinite(an) and abs(an - mid) <= 1e-3 * (1 + abs(an)):
        return float(an), None
    return float(mid), None


def _ramp(c1, c2, n):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    if n == 1:
        return [tuple(b)]
    return [tuple(a + (b - a) * k / (n - 1)) for k in range(n)]


def plot_secants_to_tangent(f, x0, hs=(2, 1, 0.5, 0.25), xlim=None, dark=False):
    """
    Sekanten durch P0 = (x0 | f(x0)) und Q = (x0+h | f(x0+h)) für kleiner
    werdendes h. Sie drehen sich in die Tangente — die Legende zeigt die
    Steigungen, die gegen f′(x0) laufen.
    """
    _m, plt = _plt()
    from matplotlib.lines import Line2D
    th = _theme(dark)
    fn = _Fn(f)
    dfn = _deriv(fn)
    x0, l0 = _xval(x0, "x0")
    y0 = _at(fn, x0, "x0")
    hs = [float(hh) for hh in hs if float(hh) != 0.0]
    if not hs:
        raise ValueError("hs braucht mindestens einen Wert ungleich 0")
    rows = []
    for hh in hs:
        yq = fn(np.array([x0 + hh]))[0]
        if _fin(yq):
            rows.append((hh, float(yq), (yq - y0) / hh))
    if not rows:
        raise ValueError("f ist an keiner der Stellen x0 + h definiert")
    if xlim is None:
        a = min([x0] + [x0 + r[0] for r in rows])
        b = max([x0] + [x0 + r[0] for r in rows])
        pad = max(1.2, 0.5 * (b - a))
        lo, hi = a - pad, b + pad
    else:
        lo, hi = _xlim(xlim)
    m, bad = _slope(fn, dfn, x0)
    with plt.rc_context(th["rc"]):
        fig, ax = plt.subplots(figsize=(8.6, 5.0), layout="constrained")
        sc = _scan(fn, lo, hi)
        yl = _auto_ylim(sc, include=[y0] + [r[1] for r in rows])
        _draw(ax, th, sc, th["f"])
        cols = _ramp(_lighten(th["sec"], 0.15), th["sec2"], len(rows))
        handles = []
        for (hh, yq, mh), col in zip(rows, cols):
            _line(ax, mh, x0, y0, lo, hi, color=col, lw=1.7, zorder=4, alpha=0.95)
            _closed_dot(ax, th, col, x0 + hh, yq, size=6.5, z=7)
            handles.append(Line2D([], [], color=col, lw=2,
                                  label=f"h = {_num(hh)}:  m = {_num(mh, 3)}"))
        if m is not None:
            _line(ax, m, x0, y0, lo, hi, color=th["tan"], lw=2.6, zorder=5)
            handles.append(Line2D([], [], color=th["tan"], lw=2.6,
                                  label=f"Tangente (h → 0):  f′({l0}) = {_num(m, 3)}"))
        elif bad is not None:
            handles.append(Line2D([], [], color="none",
                                  label="f ist bei x₀ nicht differenzierbar"))
        _closed_dot(ax, th, th["tan"], x0, y0, size=9, z=9)
        _annot(ax, th, x0, y0, f"P₀({l0} | {_num(y0)})", -8, 10, ha="right")
        _style(ax, th, lo, hi, trig=fn.trig)
        ax.set_ylim(*yl)
        _title(ax, th, "Von der Sekante zur Tangente",
               f"Differenzenquotient  (f(x₀+h) − f(x₀)) / h  für h → 0")
        _legend_out(ax, handles, th)
        return _finish(fig)


def _nice_step(target):
    if target <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(target))
    for k in (1, 2, 2.5, 5, 10):
        if k * e >= target * 0.8:
            return k * e
    return 10 * e


def plot_tangent(f, x0, xlim=None, secants_h=(), dark=False):
    """
    Zeichnet die Tangente in P0 = (x0 | f(x0)) samt Steigungsdreieck.
    Beschriftet ist f′(x0). Mit secants_h erscheinen zusätzlich Sekanten.
    Ist f dort nicht differenzierbar, werden stattdessen die beiden
    einseitigen Steigungen gezeigt.
    """
    _m, plt = _plt()
    th = _theme(dark)
    fn = _Fn(f)
    dfn = _deriv(fn)
    x0, l0 = _xval(x0, "x0")
    y0 = _at(fn, x0, "x0")
    lo, hi = _xlim(xlim, None) or (x0 - 3, x0 + 3)
    m, bad = _slope(fn, dfn, x0)
    secs = []
    for hh in secants_h or ():
        hh = float(hh)
        yq = fn(np.array([x0 + hh]))[0] if hh != 0 else np.nan
        if _fin(yq):
            secs.append((hh, float(yq), (yq - y0) / hh))
    with plt.rc_context(th["rc"]):
        fig, ax = plt.subplots(figsize=(7.4, 5.0), layout="constrained")
        sc = _scan(fn, lo, hi)
        yl = _auto_ylim(sc, include=[y0] + [s[1] for s in secs])
        _draw(ax, th, sc, th["f"])
        for hh, yq, mh in secs:
            _line(ax, mh, x0, y0, lo, hi, color=th["sec"], lw=1.4, alpha=0.75, zorder=3.5)
            _closed_dot(ax, th, th["sec"], x0 + hh, yq, size=6, z=7)
        note = None
        if m is not None:
            _line(ax, m, x0, y0, lo, hi, color=th["tan"], lw=2.4, zorder=5)
            run = _nice_step((hi - lo) / 7)
            ax.fill([x0, x0 + run, x0 + run], [y0, y0, y0 + m * run], color=th["tan"],
                    alpha=0.10, lw=0, zorder=2)
            ax.plot([x0, x0 + run, x0 + run], [y0, y0, y0 + m * run], color=th["tan"],
                    lw=1.3, ls=(0, (4, 2)), zorder=4.5)
            _annot(ax, th, x0 + run / 2, y0, f"Δx = {_num(run)}", 0,
                   5 if m < 0 else -5, ha="center", va="bottom" if m < 0 else "top",
                   color=th["tan"], size=9)
            _annot(ax, th, x0 + run, y0 + m * run / 2, f"Δy = {_num(m * run, 3)}",
                   6 if m < 0 else -6, 0, ha="left" if m < 0 else "right", va="center",
                   color=th["tan"], size=9)
            b = y0 - m * x0
            sign = "+" if b >= 0 else "−"
            _title(ax, th, rf"Tangente in P₀({l0} | {_num(y0)}):   $f'({_num(x0)})={_num(m, 3)}$",
                   f"t(x) = {_num(m, 3)}·x {sign} {_num(abs(b), 3)}")
            xl = x0 - 0.22 * (hi - lo)
            _annot(ax, th, xl, y0 + m * (xl - x0), f"f′({l0}) = {_num(m, 3)}", 0, 9,
                   color=th["tan"], bold=True, ha="center")
        else:
            sl, sr = bad
            if abs(sl) == INF and abs(sr) == INF:
                ax.axvline(x0, color=th["tan"], lw=2.4, zorder=5)
                note = "senkrechte Tangente — f′ existiert dort nicht"
            elif np.isfinite(sl) and np.isfinite(sr):
                span = (hi - lo) * 0.35
                for s_, side in ((sl, -1), (sr, +1)):
                    xa, xb = (x0 - span, x0) if side < 0 else (x0, x0 + span)
                    ax.plot([xa, xb], [y0 + s_ * (xa - x0), y0 + s_ * (xb - x0)],
                            color=th["tan"], lw=2.2, ls=(0, (5, 2)), zorder=5)
                note = f"Knick: Steigung links {_num(sl, 3)}, rechts {_num(sr, 3)}"
            else:
                note = "keine Tangente"
            _title(ax, th, f"f ist bei x₀ = {l0} nicht differenzierbar", note)
        _closed_dot(ax, th, th["tan"], x0, y0, size=9, z=9)
        _style(ax, th, lo, hi, trig=fn.trig)
        ax.set_ylim(*yl)
        return _finish(fig)


# ==========================================================================
# Nullstellen, kritische Punkte, Wendepunkte
# ==========================================================================

def _sgn(v, tiny=1e-13):
    return 0 if (not np.isfinite(v) or abs(v) < tiny) else (1 if v > 0 else -1)


def _bisect_root(fn, a, b):
    fa = fn(np.array([a]))[0]
    for _ in range(100):
        m = 0.5 * (a + b)
        fm = fn(np.array([m]))[0]
        if fm == 0 or abs(b - a) < 1e-14 * max(1.0, abs(m)):
            return m
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def _min_abs(fn, a, b):
    for _ in range(100):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        y1, y2 = np.abs(fn(np.array([m1, m2])))
        if y1 < y2:
            b = m2
        else:
            a = m1
    return 0.5 * (a + b)


def _dedupe(vals, tol=1e-7):
    out = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > tol * max(1.0, abs(v)):
            out.append(v)
    return out


def _roots(fn: _Fn, scan: _Scan):
    """Nullstellen von fn auf dem Abtastbereich: Vorzeichenwechsel und Berührpunkte."""
    lo, hi = scan.lo, scan.hi
    if fn.expr is not None:
        try:
            if fn.expr.is_polynomial(X_SYMBOL):
                if fn.expr == 0:
                    return []
                q = sp.Poly(sp.sqf_part(fn.expr), X_SYMBOL)
                if q.degree() < 1:
                    return []
                rts = np.roots([float(c) for c in q.all_coeffs()])
                real = [r.real for r in rts if abs(r.imag) <= 1e-7 * (1 + abs(r.real))]
                return _dedupe([r for r in real if lo <= r <= hi])
        except Exception:
            pass
    out = []
    for sx, sy in scan.segments:
        if len(sx) < 3:
            continue
        top = float(np.max(np.abs(sy)))
        if top < 1e-9:                       # überall 0
            continue
        s = np.sign(sy)
        for k in np.flatnonzero(s[:-1] * s[1:] < 0):
            out.append(_bisect_root(fn, sx[k], sx[k + 1]))
        for k in np.flatnonzero(s == 0):
            out.append(float(sx[k]))
        a = np.abs(sy)
        touch = [k for k in range(1, len(sx) - 1)
                 if a[k] <= a[k - 1] and a[k] <= a[k + 1] and a[k] < 1e-3 * top
                 and s[k - 1] * s[k + 1] >= 0 and (a[k] < a[k - 1] or a[k] < a[k + 1])]
        if len(touch) > 20:                  # Rauschen um 0, keine echten Berührpunkte
            continue
        for k in touch:
            c = _min_abs(fn, sx[k - 1], sx[k + 1])
            if abs(fn(np.array([c]))[0]) < 1e-9 * max(1.0, top):
                out.append(c)
    return _dedupe(out)


@dataclass
class _Pt:
    x: float
    y: float
    kind: str          # 'max' | 'min' | 'sat' | 'wen'
    y1: float | None = None
    y2: float | None = None


def _analyse(fn, d1, d2, sc0, sc1, sc2):
    """Kritische Punkte (max/min/sat) und Wendepunkte (wen) von f."""
    lo, hi = sc0.lo, sc0.hi
    delta = 1e-5 * max(hi - lo, 1.0)
    ev0 = [e.x for e in sc0.events]
    ev1 = [e.x for e in sc1.events]

    def near(c, lst):
        return any(abs(c - e) < 1e-6 * max(1.0, abs(c)) for e in lst)

    def flips(g, c):
        l, r = g(np.array([c - delta, c + delta]))
        return _sgn(l, 1e-13 * (1 + sc1.S)), _sgn(r, 1e-13 * (1 + sc1.S))

    crit: list[_Pt] = []
    cands = list(_roots(d1, sc1))
    cands += [e.x for e in sc1.events if not near(e.x, cands)]
    for c in _dedupe(cands):
        if near(c, ev0):
            continue
        y = fn(np.array([c]))[0]
        if not _fin(y):
            continue
        l, r = flips(d1, c)
        is_root = _fin(d1(np.array([c]))[0]) and abs(d1(np.array([c]))[0]) < 1e-7 * (1 + sc1.S)
        if l > 0 and r < 0:
            kind = "max"
        elif l < 0 and r > 0:
            kind = "min"
        elif is_root:
            kind = "sat"
        else:
            continue
        crit.append(_Pt(c, float(y), kind,
                        y1=float(d1(np.array([c]))[0]) if is_root else 0.0,
                        y2=_nan_to(d2(np.array([c]))[0])))
    infl: list[_Pt] = []
    cands = list(_roots(d2, sc2))
    cands += [e.x for e in sc2.events if not near(e.x, cands)]
    for c in _dedupe(cands):
        if near(c, ev0) or near(c, ev1):
            continue
        y = fn(np.array([c]))[0]
        if not _fin(y):
            continue
        l, r = d2(np.array([c - delta, c + delta]))
        tiny = 1e-13 * (1 + sc2.S)
        if _sgn(l, tiny) * _sgn(r, tiny) < 0:
            infl.append(_Pt(c, float(y), "wen",
                            y1=_nan_to(d1(np.array([c]))[0]), y2=0.0))
    # Sattelpunkte (f' = 0 ohne Vorzeichenwechsel, f'' wechselt) nicht doppelt zeichnen
    sat_x = [p.x for p in crit if p.kind == "sat"]
    infl = [p for p in infl if not near(p.x, sat_x)]
    return crit, infl


def _nan_to(v, default=None):
    return float(v) if np.isfinite(v) else default


def _sign_intervals(fn, scan, roots):
    out = []
    for sx, _sy in scan.segments:
        cuts = sorted({float(sx[0]), float(sx[-1])}
                      | {r for r in roots if sx[0] < r < sx[-1]})
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 1e-12:
                continue
            m = fn(np.array([0.5 * (a + b)]))[0]
            s = 0 if (not np.isfinite(m) or abs(m) < 1e-12) else (1 if m > 0 else -1)
            out.append((a, b, s))
    return out


def _shade_spans(ax, th, intervals, alpha=0.11):
    for a, b, s in intervals:
        if s:
            ax.axvspan(a, b, color=th["pos"] if s > 0 else th["neg"], alpha=alpha,
                       lw=0, zorder=0.6)


def _fill_sign(ax, th, scan, yl, alpha=0.24):
    for sx, sy in scan.segments:
        yc = np.clip(sy, yl[0] - (yl[1] - yl[0]), yl[1] + (yl[1] - yl[0]))
        ax.fill_between(sx, 0, yc, where=yc > 0, interpolate=True, color=th["pos"],
                        alpha=alpha, lw=0, zorder=1.2)
        ax.fill_between(sx, 0, yc, where=yc < 0, interpolate=True, color=th["neg"],
                        alpha=alpha, lw=0, zorder=1.2)


_MARK = {"max": ("^", "hi", "Hochpunkt"), "min": ("v", "lo", "Tiefpunkt"),
         "sat": ("s", "sat", "Sattelpunkt"), "wen": ("D", "wen", "Wendepunkt")}


def _marker(ax, th, kind, px, py, size=9, z=9):
    m, key, _name = _MARK[kind]
    ax.plot([px], [py], m, ms=size, mfc=th[key], mec=th["bg"], mew=1.2, zorder=z,
            clip_on=True, ls="none")


def _marker_handle(th, kind, label=None):
    from matplotlib.lines import Line2D
    m, key, name = _MARK[kind]
    return Line2D([], [], marker=m, ls="none", ms=8, mfc=th[key], mec=th["bg"],
                  label=label or name)


def _label_points(ax, th, pts, lo, hi, label_ok=True):
    if not label_ok:
        return
    w = hi - lo
    letter = {"max": "H", "min": "T", "sat": "S", "wen": "W"}
    for p in pts:
        ha = "left" if p.x < lo + 0.12 * w else ("right" if p.x > hi - 0.12 * w else "center")
        dy = {"max": 11, "min": -11, "sat": 11, "wen": -12}[p.kind]
        _annot(ax, th, p.x, p.y, f"{letter[p.kind]}({_num(p.x)} | {_num(p.y)})", 0, dy,
               ha=ha, va="bottom" if dy > 0 else "top", size=8.5,
               color=th[_MARK[p.kind][1]] if not th["dark"] else th["fg"], bold=True)


def _guides(axes, th, xs):
    for ax in axes:
        for gx in xs:
            ax.axvline(gx, color=th["muted"], lw=0.9, ls=(0, (1.5, 3)), alpha=0.9,
                       zorder=1.4)


def _auto_xlim(fn: _Fn):
    """Sinnvoller Ausschnitt um Nullstellen, Extrema, Wendestellen, Pole."""
    a, b = -6.5, 6.5
    sc0 = _scan(fn, a, b, n=1600)
    d1 = _deriv(fn)
    d2 = _deriv(d1)
    sc1, sc2 = _scan(d1, a, b, n=1600), _scan(d2, a, b, n=1600)
    pts = [e.x for e in sc0.events if e.pole or e.ls == "f" or e.rs == "f"]
    pts += _roots(fn, sc0) + _roots(d1, sc1) + _roots(d2, sc2)
    pts += [e.x for e in sc1.events] + [e.x for e in sc2.events]
    pts = [p for p in pts if a <= p <= b]
    if not pts:
        return -5.0, 5.0
    left, right = min(pts + [0.0]), max(pts + [0.0])
    margin = max(0.75, 0.12 * (right - left))
    left, right = left - margin, right + margin
    if right - left < 6:
        mid = 0.5 * (left + right)
        left, right = mid - 3, mid + 3
    return float(left), float(right)


# ==========================================================================
# plot_derivative und plot_curve
# ==========================================================================

def plot_derivative(f, xlim=None, dark=False):
    """
    Zwei Felder untereinander wie auf den Folien: oben f, darunter f′.
    Wo f eine waagrechte Tangente hat, ist f′ = 0 (gepunktete Linien).
    """
    _m, plt = _plt()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    th = _theme(dark)
    fn = _Fn(f)
    lo, hi = _xlim(xlim, None) or _auto_xlim(fn)
    d1 = _deriv(fn)
    d2 = _deriv(d1)
    with plt.rc_context(th["rc"]):
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True,
                                     layout="constrained")
        sc0, sc1 = _scan(fn, lo, hi), _scan(d1, lo, hi)
        sc2 = _scan(d2, lo, hi, n=1200)
        crit, _infl = _analyse(fn, d1, d2, sc0, sc1, sc2)
        yl0 = _auto_ylim(sc0, include=[p.y for p in crit])
        yl1 = _auto_ylim(sc1, include=[0])
        r1 = [p.x for p in crit]
        _draw(a0, th, sc0, th["f"], label="f")
        _draw(a1, th, sc1, th["d1"], label="f′")
        _fill_sign(a1, th, sc1, yl1)
        # waagrechte Tangenten
        half = 0.045 * (hi - lo)
        for p in crit:
            a0.plot([p.x - half, p.x + half], [p.y, p.y], color=th["tan"], lw=2.0,
                    zorder=5, solid_capstyle="round")
            _marker(a0, th, p.kind, p.x, p.y, size=7.5)
            _marker(a1, th, p.kind, p.x, 0.0, size=7.5)
        _guides((a0, a1), th, r1)
        _style(a0, th, lo, hi, trig=fn.trig, xlabel="", ylabel="f(x)")
        _style(a1, th, lo, hi, trig=fn.trig, xlabel="x", ylabel="f′(x)")
        a0.set_ylim(*yl0)
        a1.set_ylim(*yl1)
        a0.set_title("Funktion  f(x)", fontsize=11.5, loc="left", pad=8)
        a1.set_title("Ableitung  f′(x) = Steigung von f", fontsize=11.5, loc="left", pad=8)
        h0 = [Line2D([], [], color=th["tan"], lw=2.2, label="waagrechte Tangente\n(f′ = 0)")]
        h0 += [_marker_handle(th, k) for k in ("max", "min", "sat") if any(p.kind == k for p in crit)]
        _legend_out(a0, h0, th)
        _legend_out(a1, [Patch(fc=th["pos"], alpha=0.35, label="f′ > 0: f steigt"),
                         Patch(fc=th["neg"], alpha=0.35, label="f′ < 0: f fällt")], th)
        return _finish(fig)


def plot_curve(f, xlim=None, dark=False):
    """
    Kurvendiskussion in drei Feldern wie auf Folie sw4:
    f, f′ (Steigung) und f″ (Krümmung). Wo f′ > 0 ist, steigt f (grün
    hinterlegt), wo f′ < 0, fällt f (rot). Im dritten Feld zeigt das
    Vorzeichen von f″ die Krümmung. Hoch-, Tief-, Sattel- und
    Wendepunkte sind markiert und in allen Feldern durch gepunktete Linien
    verbunden.
    """
    _m, plt = _plt()
    from matplotlib.patches import Patch
    th = _theme(dark)
    fn = _Fn(f)
    lo, hi = _xlim(xlim, None) or _auto_xlim(fn)
    d1 = _deriv(fn)
    d2 = _deriv(d1)
    with plt.rc_context(th["rc"]):
        fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.4), sharex=True,
                                 layout="constrained")
        a0, a1, a2 = axes
        sc0, sc1, sc2 = _scan(fn, lo, hi), _scan(d1, lo, hi), _scan(d2, lo, hi)
        crit, infl = _analyse(fn, d1, d2, sc0, sc1, sc2)
        r1 = _roots(d1, sc1)
        r2 = _roots(d2, sc2)
        yl0 = _auto_ylim(sc0, include=[p.y for p in crit + infl],
                         pad=0.2 if len(crit) + len(infl) <= 8 else 0.08)
        yl1 = _auto_ylim(sc1, include=[0] + [p.y1 for p in infl if p.y1 is not None])
        yl2 = _auto_ylim(sc2, include=[0] + [p.y2 for p in crit if p.y2 is not None])

        _shade_spans(a0, th, _sign_intervals(d1, sc1, r1))
        _draw(a0, th, sc0, th["f"])
        _draw(a1, th, sc1, th["d1"])
        _draw(a2, th, sc2, th["d2"])
        _fill_sign(a1, th, sc1, yl1)
        _fill_sign(a2, th, sc2, yl2)

        few = len(crit) + len(infl) <= 8
        for p in crit + infl:
            _marker(a0, th, p.kind, p.x, p.y)
        _label_points(a0, th, crit + infl, lo, hi, label_ok=few)
        for p in crit:
            _marker(a1, th, p.kind, p.x, 0.0, size=8)
            if p.y2 is not None:
                _marker(a2, th, p.kind, p.x, p.y2, size=7)
        for p in infl:
            if p.y1 is not None:
                _marker(a1, th, "wen", p.x, p.y1, size=7)
            _marker(a2, th, "wen", p.x, 0.0, size=8)
        for p in crit:
            if p.kind == "sat":
                _marker(a2, th, "sat", p.x, p.y2 or 0.0, size=7)
        _guides(axes, th, [p.x for p in crit + infl])

        _style(a0, th, lo, hi, trig=fn.trig, xlabel="", ylabel="f(x)")
        _style(a1, th, lo, hi, trig=fn.trig, xlabel="", ylabel="f′(x)")
        _style(a2, th, lo, hi, trig=fn.trig, xlabel="x", ylabel="f″(x)")
        a0.set_ylim(*yl0)
        a1.set_ylim(*yl1)
        a2.set_ylim(*yl2)
        a0.set_title("Funktion  f(x)", fontsize=11.5, loc="left", pad=8)
        a1.set_title("1. Ableitung  f′(x) — Steigung", fontsize=11.5, loc="left", pad=8)
        a2.set_title("2. Ableitung  f″(x) — Krümmung", fontsize=11.5, loc="left", pad=8)

        kinds = [k for k in ("max", "min", "sat", "wen")
                 if any(p.kind == k for p in crit + infl)]
        _legend_out(a0, [Patch(fc=th["pos"], alpha=0.35, label="f′ > 0: f steigt"),
                         Patch(fc=th["neg"], alpha=0.35, label="f′ < 0: f fällt")]
                    + [_marker_handle(th, k) for k in kinds], th)
        h1 = [Patch(fc=th["pos"], alpha=0.35, label="f′ > 0"),
              Patch(fc=th["neg"], alpha=0.35, label="f′ < 0")]
        if crit:
            h1.append(_marker_handle(th, "max", "Nullstelle von f′\n(Art siehe oben)")
                      if False else _marker_handle(th, crit[0].kind, "f′ = 0 (kritische Stelle)"))
        if infl:
            h1.append(_marker_handle(th, "wen", "Extremum von f′\n(Wendestelle von f)"))
        _legend_out(a1, h1, th)
        h2 = [Patch(fc=th["pos"], alpha=0.35, label="f″ > 0: linksgekrümmt"),
              Patch(fc=th["neg"], alpha=0.35, label="f″ < 0: rechtsgekrümmt")]
        if infl:
            h2.append(_marker_handle(th, "wen", "f″ = 0 (Wendestelle)"))
        _legend_out(a2, h2, th)
        return _finish(fig)


# ==========================================================================
# Grenzwert
# ==========================================================================

def _limit_text(state, v, which):
    if state == "f":
        return _num(v)
    if state == "p":
        return "+∞" if v > 0 else "−∞"
    if state == "o":
        return "existiert nicht (schwingt)"
    return "nicht definiert"


def plot_limit(f, x0, xlim=None, dark=False):
    """
    Annäherung an x0 von links und von rechts. Der Grenzwert erscheint als
    waagrechte Linie, der Funktionswert f(x0) als Punkt; ist f dort nicht
    definiert (Lücke), steht an dieser Stelle ein offener Kreis. Bei Polen
    laufen die Punkte aus dem Bild — unter der Zeichnung steht, was
    links und rechts herauskommt.
    """
    _m, plt = _plt()
    th = _theme(dark)
    fn = _Fn(f)
    x0, l0 = _xval(x0, "x0")
    lo, hi = _xlim(xlim, None) or (x0 - 3, x0 + 3)
    if not lo < x0 < hi:
        raise ValueError("x0 muss innerhalb von xlim liegen")
    with plt.rc_context(th["rc"]):
        fig, ax = plt.subplots(figsize=(7.6, 5.4), layout="constrained")
        sc = _scan(fn, lo, hi)
        v0 = fn(np.array([x0]))[0]
        val = float(v0) if _fin(v0) else None
        ls, lv = _side(fn, x0, -1, sc.S, sc.M)
        rs, rv = _side(fn, x0, +1, sc.S, sc.M)
        inc = [y for y in (lv, rv, val) if y is not None and np.isfinite(y)]
        yl = _auto_ylim(sc, include=inc)
        _draw(ax, th, sc, th["f"], skip_x=x0)
        ax.axvline(x0, color=th["muted"], lw=1.1, ls=(0, (4, 3)), zorder=1.8)

        w = 0.2 * min(x0 - lo, hi - x0, hi - lo)
        for side, state, col, text in ((-1, ls, th["sec"], "von links"),
                                       (+1, rs, th["pt"], "von rechts")):
            ds = w * 0.5 ** np.arange(4)
            px = x0 + side * ds
            py = fn(px)
            for k, (a_, b_) in enumerate(zip(px, py)):
                if np.isfinite(b_):
                    ax.plot([a_], [b_], "o", ms=6.5, mfc=col, mec=th["bg"],
                            mew=0.8, zorder=7, alpha=0.4 + 0.6 * k / 3, clip_on=True)
            if np.isfinite(py[0]) and np.isfinite(py[2]):
                ax.annotate("", xy=(px[2], py[2]), xytext=(px[0], py[0]),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, mutation_scale=15,
                                            shrinkA=4, shrinkB=4), zorder=6.5)
            if np.isfinite(py[0]):
                _annot(ax, th, px[0], py[0], text, -9 * side, 11, color=col,
                       ha="right" if side < 0 else "left", size=9.5, z=2.6, bold=True)

        # Grenzwert-Linien
        same = ls == "f" and rs == "f" and abs(lv - rv) <= 1e-5 * (1 + abs(lv))
        if same:
            ax.axhline(lv, color=th["tan"], lw=1.4, ls=(0, (6, 3)), zorder=2.5)
            _annot(ax, th, lo, lv, f"Grenzwert {_num(lv)}", 6, 5, color=th["tan"],
                   bold=True)
        else:
            for state, v, col, name, ha, px in ((ls, lv, th["sec"], "links", "left", lo),
                                                (rs, rv, th["pt"], "rechts", "right", hi)):
                if state == "f":
                    ax.axhline(v, color=col, lw=1.3, ls=(0, (6, 3)), zorder=2.5)
                    _annot(ax, th, px, v, f"{name}: {_num(v)}", 6 if ha == "left" else -6, 5,
                           color=col, ha=ha, bold=True)
        # Punkte an der Stelle selbst
        if val is not None:
            _closed_dot(ax, th, th["tan"], x0, val, size=9, z=9)
        for state, v in ((ls, lv), (rs, rv)):
            if state == "f" and (val is None or abs(val - v) > 1e-5 * (1 + abs(v))):
                _open_dot(ax, th, th["tan"], x0, v, size=9, z=9)
        _style(ax, th, lo, hi, trig=fn.trig)
        ax.set_ylim(*yl)

        # Beschriftung
        if ls == "f" and rs == "f" and same:
            lim = f"Grenzwert = {_num(lv)}"
        elif ls == "p" and rs == "p" and lv == rv:
            lim = f"Grenzwert = {_limit_text('p', lv, '')} (uneigentlich)"
        else:
            lim = "Grenzwert existiert nicht"
        if val is None:
            fx = "f(x₀) ist nicht definiert"
        else:
            fx = f"f(x₀) = {_num(val)}"
            if same and abs(val - lv) <= 1e-5 * (1 + abs(lv)):
                fx += "  → f ist bei x₀ stetig"
            elif same:
                fx += "  → hebbare Unstetigkeit"
        lines = [f"x → {l0}⁻:  {_limit_text(ls, lv, 'l')}      "
                 f"x → {l0}⁺:  {_limit_text(rs, rv, 'r')}", f"{lim}      {fx}"]
        _title(ax, th, f"Grenzwert für x → {l0}")
        ax.text(0.0, -0.13, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color=th["fg"], linespacing=1.5)
        return _finish(fig)


# ==========================================================================
# Plotly (freiwillig)
# ==========================================================================

def has_plotly() -> bool:
    try:
        import plotly.graph_objects  # noqa: F401
        return True
    except Exception:
        return False


_PLOTLY_HINT = ("Plotly ist nicht installiert — die interaktive Zeichnung braucht es.\n"
                "Installation:  pip install plotly\n"
                "(Die Matplotlib-Zeichnungen wie plot_function funktionieren auch ohne.)")


def plotly_function(f, xlim=(-5, 5), with_derivatives=True):
    """
    Interaktive Zeichnung (zoomen, Werte ablesen): f und, wenn
    with_derivatives=True, darunter f′ und f″. Pole werden nicht überbrückt.
    Wirft RuntimeError, wenn Plotly fehlt.
    """
    if not has_plotly():
        raise RuntimeError(_PLOTLY_HINT)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lo, hi = _xlim(xlim, (-5, 5))
    fn = _Fn(f)
    fns = [("f(x)", fn, COLORS[0])]
    if with_derivatives:
        d1 = _deriv(fn)
        fns += [("f′(x)", d1, COLORS[4]), ("f″(x)", _deriv(d1), COLORS[5])]
    rows = len(fns)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.4] + [0.3] * (rows - 1) if rows > 1 else [1.0],
                        subplot_titles=[n for n, _f, _c in fns])
    for r, (name, g, col) in enumerate(fns, start=1):
        sc = _scan(g, lo, hi)
        cx, cy = sc.curve()
        yl = _auto_ylim(sc, include=[0] if r > 1 else ())
        cy = np.where(np.abs(cy) > 4 * max(abs(yl[0]), abs(yl[1])) + 1, np.nan, cy)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=name, connectgaps=False,
                                 line=dict(color=col, width=2.6),
                                 hovertemplate="x = %{x:.4g}<br>" + name + " = %{y:.4g}<extra></extra>"),
                      row=r, col=1)
        ox = [e.x for e in sc.events if e.pole and abs(e.x) > 1e-9]
        for px in ox:
            fig.add_vline(x=px, line=dict(color="#86868b", width=1, dash="dash"),
                          row=r, col=1)
        for e in sc.events:
            for state, v in ((e.ls, e.lv), (e.rs, e.rv)):
                if state != "f":
                    continue
                closed = e.val is not None and abs(e.val - v) <= 1e-5 * (1 + abs(v))
                fig.add_trace(go.Scatter(
                    x=[e.x], y=[v], mode="markers", showlegend=False,
                    marker=dict(size=9, color=col if closed else "white",
                                line=dict(color=col, width=2)),
                    hovertemplate=f"x = {e.x:.4g}<br>Grenzwert = {v:.4g}<extra></extra>"),
                    row=r, col=1)
        fig.update_yaxes(range=list(yl), row=r, col=1, zeroline=True,
                         zerolinecolor="#86868b", gridcolor="#e5e5ea")
    fig.update_xaxes(range=[lo, hi], zeroline=True, zerolinecolor="#86868b",
                     gridcolor="#e5e5ea", showspikes=True, spikemode="across",
                     spikethickness=1)
    fig.update_xaxes(title_text="x", row=rows, col=1)
    fig.update_layout(template="plotly_white", height=320 + 230 * rows - 230 * (rows == 1) * 0,
                      margin=dict(l=50, r=20, t=50, b=40), showlegend=False,
                      hovermode="x")
    return fig


def save_html(fig, path):
    """
    Speichert eine Plotly-Figur als eigenständige HTML-Datei (funktioniert
    offline) oder eine Matplotlib-Figur als HTML-Seite mit eingebettetem SVG.
    Gibt den Pfad zurück.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "write_html"):
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    elif hasattr(fig, "savefig"):
        import io
        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        path.write_text(
            "<!doctype html><html lang='de'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Zeichnung</title><body style='margin:0;padding:16px;"
            "font-family:system-ui;background:#fff'>" + buf.getvalue() + "</body></html>",
            encoding="utf-8")
    else:
        raise TypeError("save_html erwartet eine Plotly- oder Matplotlib-Figur")
    return path
