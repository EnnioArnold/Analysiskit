#!/usr/bin/env python3
"""
Semesterwoche 4: Monotonie, kritische Punkte, Krümmung und Extrema.

Alle Verfahren rechnen exakt (SymPy) und zeigen den Weg, wie er in der
Prüfung auf dem Papier stünde: Ableitung bilden, Nullstellen bestimmen,
Vorzeichentabelle mit Testwerten, Schluss aus den Kriterien der Vorlesung.

    from anakit.analysis import monotony, critical_points, curvature
    from anakit.analysis import extrema, inflection_points, curve_discussion

    curve_discussion("x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27")

Kriterien (Folien SW4):

    f'(x) > 0   =>  f streng monoton wachsend      f'(x) >= 0  =>  monoton wachsend
    f'(x) < 0   =>  f streng monoton fallend       f'(x) <= 0  =>  monoton fallend
    f''(x) > 0  =>  Linkskrümmung                  f''(x) < 0  =>  Rechtskrümmung
    f'(x0) = 0 und f''(x0) > 0  =>  relatives Minimum
    f'(x0) = 0 und f''(x0) < 0  =>  relatives Maximum
    f'(x0) = 0 und f''(x0) = 0  =>  keine Aussage möglich
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import sympy as sp

from .core import fmt, parse, tex, x
from .render import Result, Step

__all__ = ["monotony", "critical_points", "curvature", "extrema",
           "inflection_points", "curve_discussion", "MonotonyResult",
           "CriticalResult", "CurvatureResult", "ExtremaResult",
           "InflectionResult", "DiscussionResult", "SignRow", "PointInfo"]

MIN = "relatives Minimum"
MAX = "relatives Maximum"
SAD = "Sattelpunkt"
UNB = "unbestimmt"

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# ==========================================================================
# Kleine Hilfen
# ==========================================================================

def _fl(e) -> float:
    """Gleitkommawert eines (auch unendlichen) Ausdrucks."""
    if e == sp.oo:
        return math.inf
    if e == -sp.oo:
        return -math.inf
    return float(sp.N(e, 30))


def _inexact(e) -> bool:
    return isinstance(e, sp.Basic) and (e.has(sp.CRootOf) or e.has(sp.Float))


def _dec(e, digits: int = 4) -> str:
    v = float(sp.re(sp.N(e, 30)))
    s = f"{v:.{digits}f}"
    return "0." + "0" * digits if s.startswith("-0.") and float(s) == 0 else s


def _item(e):
    """Für Items: exakt, ausser bei CRootOf/Float -> Näherung als Text."""
    if _inexact(e):
        return "≈ " + _dec(e)
    return e


def _valstr(e) -> str:
    """Kurzer Text für einen Wert: exakt wenn kurz, sonst Näherung."""
    if _inexact(e):
        return "≈ " + _dec(e)
    s = fmt(e)
    if len(s) > 28:
        return "≈ " + _dec(e)
    return s


def _simp(e):
    if e.is_polynomial(x):
        return sp.expand(e)
    try:
        return sp.simplify(e)
    except Exception:
        return e


def _sign_num(val) -> int:
    """Vorzeichen eines (nicht exakt null erwarteten) Werts, numerisch robust."""
    try:
        v = sp.re(sp.N(val, 40))
        if v.is_number is False or v == sp.nan:
            return 0
        fv = float(v)
    except Exception:
        return 0
    if abs(fv) < 1e-30:
        return 0
    return 1 if fv > 0 else -1


def _sign_at(expr, pt) -> int:
    """Vorzeichen von expr an einer Stelle pt (exakte Null wird erkannt)."""
    val = expr.subs(x, pt)
    inexact = _inexact(pt)
    if not inexact:
        try:
            s = sp.simplify(val)
            if s == 0:
                return 0
            val = s
        except Exception:
            pass
    try:
        v = sp.re(sp.N(val, 80))
        fv = float(v)
        if abs(fv) < (1e-40 if inexact else 1e-30):
            return 0
        return 1 if fv > 0 else -1
    except Exception:
        return 0


def _value_at(f, pt):
    """f(pt); für CRootOf bei Polynomen mit dem Minimalpolynom reduziert."""
    if isinstance(pt, sp.CRootOf) and f.is_polynomial(x):
        try:
            p = sp.Poly(pt.poly.as_expr().subs(pt.poly.gen, x), x)
            r = sp.rem(sp.Poly(f, x), p)
            return r.as_expr().subs(x, pt)
        except Exception:
            pass
    val = f.subs(x, pt)
    if _inexact(pt):
        return val
    try:
        return sp.simplify(val)
    except Exception:
        return val


def _bounds(domain):
    if domain is None:
        return -sp.oo, sp.oo
    a, b = domain
    a = -sp.oo if a is None else parse(a)
    b = sp.oo if b is None else parse(b)
    if _fl(a) >= _fl(b):
        raise ValueError("domain=(a, b) verlangt a < b.")
    return a, b


def _test_point(a: float, b: float):
    """Ein möglichst schöner Testwert im offenen Intervall (a, b)."""
    if math.isinf(a) and math.isinf(b):
        return sp.Integer(0)
    if math.isinf(a):
        return sp.Integer(math.floor(b) - 1)
    if math.isinf(b):
        return sp.Integer(math.ceil(a) + 1)
    mid = (a + b) / 2
    lo_i, hi_i = math.floor(a) + 1, math.ceil(b) - 1
    if lo_i <= hi_i:
        return sp.Integer(min(max(round(mid), lo_i), hi_i))
    for d in (2, 10, 100, 1000, 10 ** 4, 10 ** 6, 10 ** 9):
        q = sp.Rational(round(mid * d), d)
        if a < float(q) < b:
            return q
    return sp.nsimplify(mid)


# --------------------------------------------------------------------------
# Nullstellen
# --------------------------------------------------------------------------

def _numeric_roots(g, lo, hi):
    """Nullstellen durch Vorzeichenwechsel auf einem dichten Raster + Bisektion."""
    import numpy as np
    flo, fhi = _fl(lo), _fl(hi)
    if math.isinf(flo) and math.isinf(fhi):
        wl, wh = -10.0, 10.0
    elif math.isinf(flo):
        wl, wh = fhi - 20.0, fhi
    elif math.isinf(fhi):
        wl, wh = flo, flo + 20.0
    else:
        wl, wh = flo, fhi
    fn = sp.lambdify(x, g, ["numpy", {"DiracDelta": lambda t: 0.0 * np.asarray(t, dtype=float), "Heaviside": lambda t, h0=0.5: np.heaviside(t, h0)}])

    def ev(t):
        with np.errstate(all="ignore"):
            return np.asarray(fn(t), dtype=float) * np.ones_like(t)

    xs = np.linspace(wl, wh, 40001)
    ys = ev(xs)
    s = np.sign(ys)
    out = [sp.Float(float(xs[i]), 15) for i in np.where(ys == 0)[0] if 0 < i < len(xs) - 1]
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        a, b = float(xs[i]), float(xs[i + 1])
        fa = float(ev(np.array([a]))[0])
        for _ in range(100):
            m = (a + b) / 2
            fm = float(ev(np.array([m]))[0])
            if fm == 0:
                a = b = m
                break
            if (fm < 0) == (fa < 0):
                a, fa = m, fm
            else:
                b = m
        out.append(sp.Float((a + b) / 2, 15))
    return out, (wl, wh)


def _find_roots(g, lo, hi):
    """
    Reelle Nullstellen von g im offenen Intervall (lo, hi).
    Gibt (Liste sortiert, exakt?, Hinweis) zurück.
    """
    flo, fhi = _fl(lo), _fl(hi)
    if g == 0:
        return [], True, "identisch null"
    cands, exact, info = None, True, ""
    try:
        numer = sp.fraction(sp.together(g))[0]
    except Exception:
        numer = g
    # 1. Polynom -> real_roots (exakt, ggf. CRootOf)
    try:
        poly = sp.Poly(numer, x)
        if not poly.is_zero:
            cands = list(set(sp.real_roots(poly)))
    except Exception:
        cands = None
    # 2. sonst solveset
    if cands is None:
        try:
            sol = sp.solveset(numer, x, sp.Interval(lo, hi))
            if isinstance(sol, sp.FiniteSet):
                cands = [r for r in sol if r.is_real is not False]
            elif sol == sp.S.EmptySet:
                cands = []
        except Exception:
            cands = None
    # 3. sonst numerisch
    if cands is None:
        cands, (wl, wh) = _numeric_roots(g, lo, hi)
        exact = False
        info = (f"Keine exakte Lösung gefunden; numerisch gesucht im Fenster "
                f"[{wl:.4g}, {wh:.4g}] (Vorzeichenwechsel, 40000 Rasterpunkte). Ausserhalb des Fensters "
                f"sind die Aussagen nicht gesichert (z. B. periodische Funktionen).")
    out = []
    for r in sorted(cands, key=_fl):
        fr = _fl(r)
        if not (flo < fr < fhi):
            continue
        if out and abs(fr - _fl(out[-1])) < 1e-12 * (1 + abs(fr)):
            continue
        out.append(r)
    return out, exact, info


# ==========================================================================
# Zeilen der Vorzeichentabelle
# ==========================================================================

@dataclass
class SignRow:
    """Ein Intervall der Vorzeichentabelle: Testwert, Wert und Vorzeichen."""
    lo: object
    hi: object
    test: object
    value: object
    sign: int
    lo_kind: str = "rand"     # 'rand' | 'nst' | 'pol'
    hi_kind: str = "rand"

    @property
    def interval(self):
        return sp.Interval.open(self.lo, self.hi)

    def __iter__(self):
        yield from (self.interval, self.test, self.value, self.sign)


def _build_table(g, roots, poles, lo, hi) -> list[SignRow]:
    pts = [(r, "nst") for r in roots] + [(p, "pol") for p in poles]
    pts.sort(key=lambda t: _fl(t[0]))
    bounds = [(lo, "rand")] + pts + [(hi, "rand")]
    rows = []
    for (a, ka), (b, kb) in zip(bounds, bounds[1:]):
        t = _test_point(_fl(a), _fl(b))
        val = g.subs(x, t)
        rows.append(SignRow(a, b, t, val, _sign_num(val), ka, kb))
    return rows


def _merge(rows, sign):
    """Zusammenhängende Intervalle mit gleichem Vorzeichen (über Nullstellen hinweg)."""
    out, cur, prev_ok = [], None, False
    for r in rows:
        if r.sign == sign:
            if cur is not None and prev_ok and r.lo_kind == "nst":
                cur[1] = r.hi
            else:
                cur = [r.lo, r.hi]
                out.append(cur)
            prev_ok = True
        else:
            prev_ok = False
    return [sp.Interval.open(a, b) for a, b in out]


_REL = {1: ">", -1: "<", 0: "="}


# ==========================================================================
# Gemeinsame Rechnung
# ==========================================================================

@dataclass
class PointInfo:
    """Ein kritischer Punkt samt Einordnung."""
    x0: object
    y0: object
    kind: str
    f2_sign: int
    f2: object = None
    left: int = 0           # Vorzeichen von f' links von x0
    right: int = 0
    supplemented: bool = False   # f''(x0)=0 -> Einordnung nur über Zusatzprüfung

    @property
    def point(self):
        return (self.x0, self.y0)

    def __iter__(self):
        yield from (self.x0, self.y0)

    def __getitem__(self, i):
        return (self.x0, self.y0)[i]


class _Analysis:
    """Rechnet f, f', f'', f''' und alle Tabellen einmal aus (gemeinsam genutzt)."""

    def __init__(self, f, domain=None):
        self.f = parse(f)
        if self.f.has(sp.Abs, sp.sign, sp.Piecewise, sp.Heaviside):
            raise ValueError(
                "Die Funktion ist an einer Stelle nicht differenzierbar "
                "(Betrag, Signum, abschnittsweise). Die Kurvendiskussion "
                "braucht f' und f'' überall; teile die Funktion in Abschnitte "
                "und diskutiere jeden einzeln.")
        self.domain = domain
        self.lo, self.hi = _bounds(domain)
        self.warnings: list[str] = []
        self.names: dict = {}
        self.d1 = _simp(sp.diff(self.f, x))
        self.d2 = _simp(sp.diff(self.d1, x))
        self.d3 = _simp(sp.diff(self.d2, x))
        self._cls: dict = {}

    # -- Namen ---------------------------------------------------------
    def _register(self, roots, prefix):
        for k, r in enumerate(roots, 1):
            if _inexact(r) and r not in self.names:
                self.names[r] = f"{prefix}{str(k).translate(_SUB)}"

    def s(self, r) -> str:
        if r in self.names:
            return self.names[r]
        return fmt(r)

    def t(self, r) -> str:
        if r in self.names:
            n = self.names[r]
            return n[0] + "_{" + n[1:].translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")) + "}"
        return tex(r)

    def ivl(self, a, b) -> str:
        return f"({self.s(a)}, {self.s(b)})"

    def ivls(self, lst) -> str:
        return " ∪ ".join(self.ivl(i.start, i.end) for i in lst) if lst else "—"

    # -- Definitionsbereich, Pole ---------------------------------------
    @property
    def domain_str(self) -> str:
        if self.domain is None:
            return "ℝ"
        return self.ivl(self.lo, self.hi)

    @cached_property
    def poles(self):
        try:
            s = sp.singularities(self.f, x)
        except Exception:
            return []
        out = []
        if isinstance(s, sp.FiniteSet):
            for p in s:
                if p.is_real is not False and _fl(self.lo) < _fl(p) < _fl(self.hi):
                    out.append(p)
        return sorted(out, key=_fl)

    def _roots_of(self, g, prefix):
        roots, exact, info = _find_roots(g, self.lo, self.hi)
        pf = [p for p in self.poles]
        roots = [r for r in roots if all(abs(_fl(r) - _fl(p)) > 1e-12 for p in pf)]
        self._register(roots, prefix)
        if info and info != "identisch null":
            self.warnings.append(info)
        return roots, exact, info

    # -- f' ------------------------------------------------------------
    @cached_property
    def _r1(self):
        return self._roots_of(self.d1, "x")

    @property
    def roots1(self):
        return self._r1[0]

    @cached_property
    def table1(self):
        return _build_table(self.d1, self.roots1, self.poles, self.lo, self.hi)

    # -- f'' -----------------------------------------------------------
    @cached_property
    def _r2(self):
        return self._roots_of(self.d2, "w")

    @property
    def roots2(self):
        return self._r2[0]

    @cached_property
    def table2(self):
        return _build_table(self.d2, self.roots2, self.poles, self.lo, self.hi)

    # -- Nachbarzeilen --------------------------------------------------
    @staticmethod
    def _neighbors(rows, x0):
        fx = _fl(x0)
        for a, b in zip(rows, rows[1:]):
            if abs(_fl(a.hi) - fx) < 1e-12 * (1 + abs(fx)):
                return a, b
        return None, None

    # -- Einordnung -----------------------------------------------------
    def classify(self, x0) -> PointInfo:
        key = x0
        if key in self._cls:
            return self._cls[key]
        f2s = _sign_at(self.d2, x0)
        f2v = self.d2.subs(x, x0) if _inexact(x0) else _simp(self.d2.subs(x, x0))
        ra, rb = self._neighbors(self.table1, x0)
        sl = ra.sign if ra else 0
        sr = rb.sign if rb else 0
        sup = False
        if f2s > 0:
            kind = MIN
        elif f2s < 0:
            kind = MAX
        else:
            sup = True
            if sl < 0 and sr > 0:
                kind = MIN
            elif sl > 0 and sr < 0:
                kind = MAX
            elif sl != 0 and sl == sr:
                kind = SAD
            else:
                kind = UNB
        info = PointInfo(x0, _value_at(self.f, x0), kind, f2s, f2v, sl, sr, sup)
        self._cls[key] = info
        return info

    @cached_property
    def points(self) -> list[PointInfo]:
        return [self.classify(r) for r in self.roots1]

    # -- Standardbereich für Zeichnungen --------------------------------
    def default_xlim(self):
        pts = [_fl(r) for r in list(self.roots1) + list(self.roots2) + list(self.poles)]
        pts = [p for p in pts if math.isfinite(p)]
        if pts:
            a, b = min(pts), max(pts)
            pad = max(1.0, 0.35 * (b - a))
            a, b = a - pad, b + pad
        else:
            a, b = -4.0, 4.0
        if math.isfinite(_fl(self.lo)):
            a = _fl(self.lo)
        if math.isfinite(_fl(self.hi)):
            b = _fl(self.hi)
        return (a, b)

    # ==================================================================
    # Schritte
    # ==================================================================

    def step_derivative(self, order=2) -> Step:
        items = [("f(x)", self.f), ("f'(x)", self.d1)]
        if order >= 2:
            items.append(("f''(x)", self.d2))
        return Step("Ableitungen bilden",
                    "Die Ableitungen von $f(x)$ mit den Ableitungsregeln.",
                    items, rule="Ableitungsregeln")

    def _roots_step(self, deriv_name, g, roots, exact, info, what, prefix) -> Step:
        items = []
        if g.is_polynomial(x):
            fa = sp.factor(g)
            if fa != g:
                items.append((f"{deriv_name}(x) faktorisiert", fa))
        if roots:
            for k, r in enumerate(roots, 1):
                items.append((f"{prefix}{str(k).translate(_SUB)}", _item(r)))
        notes = []
        for r in roots:
            if _inexact(r):
                if isinstance(r, sp.CRootOf):
                    notes.append(f"{self.s(r)} = {sp.sstr(r)} (exakt, SymPy-Wurzel) ≈ {_dec(r, 6)}")
                else:
                    notes.append(f"{self.s(r)} ≈ {_dec(r, 6)}")
        if notes:
            notes.insert(0, "Nicht alle Nullstellen lassen sich als einfacher Bruch/Wurzel schreiben; "
                            "sie bleiben exakt als Wurzel des Polynoms stehen.")
        if info and info != "identisch null":
            notes.append(info)
        if self.poles:
            notes.append("Nicht definiert bei: " + ", ".join(self.s(p) for p in self.poles) + ".")
        if g == 0:
            detail = f"${deriv_name}(x)=0$ gilt für alle $x$ ($f$ ist konstant)."
        elif roots:
            detail = f"{what} Lösen von $ {deriv_name}(x)=0 $ liefert:"
        else:
            detail = f"${deriv_name}(x)=0$ hat im betrachteten Bereich keine reelle Lösung."
        return Step(f"Nullstellen von {deriv_name} bestimmen", detail, items,
                    note=" ".join(notes), rule=f"{deriv_name}(x) = 0")

    def _table_step(self, deriv_name, rows, verdict, title) -> Step:
        items = []
        for r in rows:
            val = _valstr(r.value)
            items.append((self.ivl(r.lo, r.hi),
                          f"{deriv_name}({self.s(r.test)}) = {val}  {_REL[r.sign]} 0   ⇒   {verdict(r.sign)}"))
        return Step(title,
                    f"Zwischen zwei Nullstellen ändert {deriv_name} sein Vorzeichen nicht "
                    f"(Zwischenwertsatz). Ein Testwert pro Intervall genügt.",
                    items, rule="Vorzeichentabelle")

    # -- Monotonie ------------------------------------------------------
    @staticmethod
    def mono_verdict(s):
        return {1: "streng monoton wachsend", -1: "streng monoton fallend", 0: "konstant"}[s]

    @staticmethod
    def curv_verdict(s):
        return {1: "Linkskrümmung", -1: "Rechtskrümmung", 0: "gerade (f'' = 0)"}[s]

    def steps_monotony(self, derivative=True) -> list[Step]:
        steps = []
        if derivative:
            steps.append(Step("Erste Ableitung bilden", "Für die Monotonie zählt das Vorzeichen von $f'$.",
                              [("f(x)", self.f), ("f'(x)", self.d1)], rule="Ableitungsregeln"))
        r, exact, info = self._r1
        steps.append(self._roots_step("f'", self.d1, r, exact, info,
                                      "Nur an Nullstellen von $f'$ kann die Monotonie wechseln.", "x"))
        steps.append(self._table_step("f'", self.table1, self.mono_verdict, "Vorzeichentabelle von f'"))
        inc, dec = _merge(self.table1, 1), _merge(self.table1, -1)
        const = _merge(self.table1, 0)
        items = [("streng monoton wachsend auf", self.ivls(inc)),
                 ("streng monoton fallend auf", self.ivls(dec))]
        if const:
            items.append(("konstant auf", self.ivls(const)))
        note = ""
        merged = [rt for rt in self.roots1
                  if any(i.start < rt < i.end for i in inc + dec)]
        if merged:
            note = ("Bei " + ", ".join(self.s(m) for m in merged) +
                    " ist $f'=0$, aber f' wechselt dort das Vorzeichen nicht; "
                    "f bleibt über die Stelle hinweg streng monoton (nur die Steigung ist dort 0).")
            note = note.replace("$f'=0$", "f'=0")
        steps.append(Step("Monotonieverhalten ablesen",
                          "f'(x) > 0 ⇒ streng monoton wachsend (insbesondere monoton wachsend); "
                          "f'(x) < 0 ⇒ streng monoton fallend (insbesondere monoton fallend).",
                          items, note=note, rule="Monotoniekriterium"))
        return steps

    # -- Kritische Punkte -------------------------------------------------
    def steps_critical(self, derivative=True) -> list[Step]:
        steps = []
        if derivative:
            steps.append(Step("Erste Ableitung bilden",
                              "Kritische Stellen sind die Lösungen von $f'(x_0)=0$.",
                              [("f(x)", self.f), ("f'(x)", self.d1)], rule="Ableitungsregeln"))
        r, exact, info = self._r1
        steps.append(self._roots_step("f'", self.d1, r, exact, info,
                                      "Kritische Stellen $x_0$ erfüllen $f'(x_0)=0$.", "x"))
        if not self.points:
            steps.append(Step("Kritische Punkte", "Es gibt keine kritischen Stellen, also auch keine "
                              "kritischen Punkte (keine Extrema, keine Sattelpunkte).", []))
            return steps
        items = []
        for p in self.points:
            items.append((f"x₀ = {self.s(p.x0)}", f"({_valstr(p.x0) if _inexact(p.x0) else self.s(p.x0)} / "
                                                   f"{_valstr(p.y0)})"))
        steps.append(Step("Kritische Punkte (x₀ / f(x₀))",
                          "Funktionswert an jeder kritischen Stelle: $f(x_0)$ einsetzen.",
                          items, rule="kritischer Punkt (x₀/f(x₀))"))
        for p in self.points:
            steps.append(self.step_classify(p, extrema_mode=False))
        return steps

    def _f1_item(self, row):
        return f"f'({self.s(row.test)}) = {_valstr(row.value)} {_REL[row.sign]} 0"

    def step_classify(self, p: PointInfo, extrema_mode: bool) -> Step:
        s0 = self.s(p.x0)
        t0 = self.t(p.x0)
        ra, rb = self._neighbors(self.table1, p.x0)
        items = [("f''(x₀)", _item(p.f2))]
        rel = _REL[p.f2_sign]
        if not p.supplemented:
            detail = (f"Bei $x_0={t0}$ gilt $f'(x_0)=0$ und $f''(x_0) {rel} 0$ ⇒ {p.kind} "
                      f"bei ({s0} / {_valstr(p.y0)}).")
            rule = ("f'(x₀)=0 ∧ f''(x₀)>0 ⇒ relatives Minimum" if p.f2_sign > 0
                    else "f'(x₀)=0 ∧ f''(x₀)<0 ⇒ relatives Maximum")
            return Step(f"Einordnung bei x₀ = {s0}", detail, items, rule=rule)
        # f''(x0) = 0
        if extrema_mode:
            detail = (f"Bei $x_0={t0}$ gilt $f'(x_0)=0$ und $f''(x_0)=0$ ⇒ keine Aussage möglich "
                      f"(Kriterium Folie 13). Zusatzprüfung: Vorzeichen von $f'$ links und rechts von $x_0$.")
        else:
            detail = (f"Bei $x_0={t0}$ gilt $f'(x_0)=0$ und $f''(x_0)=0$; die 2. Ableitung entscheidet "
                      f"nicht. Darum: Vorzeichen von $f'$ links und rechts von $x_0$.")
        if ra and rb:
            items.append(("f' links von x₀", self._f1_item(ra)))
            items.append(("f' rechts von x₀", self._f1_item(rb)))
        if p.kind == MIN:
            concl = "f' wechselt von − nach + ⇒ " + MIN
        elif p.kind == MAX:
            concl = "f' wechselt von + nach − ⇒ " + MAX
        elif p.kind == SAD:
            concl = "f' wechselt das Vorzeichen nicht ⇒ " + SAD
        else:
            concl = "keine Entscheidung möglich"
        items.append(("Ergebnis", f"{p.kind} bei ({s0} / {_valstr(p.y0)})"))
        return Step(f"Einordnung bei x₀ = {s0} (Zusatzprüfung über f')", detail, items,
                    note=f"Zusatz (nicht Teil des Kriteriums auf Folie 13): {concl}.",
                    rule="Vorzeichenwechsel von f'")

    # -- Krümmung -------------------------------------------------------
    def steps_curvature(self, derivative=True) -> list[Step]:
        steps = []
        if derivative:
            steps.append(Step("Zweite Ableitung bilden", "Für die Krümmung zählt das Vorzeichen von $f''$.",
                              [("f'(x)", self.d1), ("f''(x)", self.d2)], rule="Ableitungsregeln"))
        r, exact, info = self._r2
        steps.append(self._roots_step("f''", self.d2, r, exact, info,
                                      "Die Krümmung kann nur an Nullstellen von $f''$ wechseln.", "w"))
        steps.append(self._table_step("f''", self.table2, self.curv_verdict, "Vorzeichentabelle von f''"))
        left, right = _merge(self.table2, 1), _merge(self.table2, -1)
        wendes = self.wendestellen
        items = [("Linkskrümmung (f'' > 0) auf", self.ivls(left)),
                 ("Rechtskrümmung (f'' < 0) auf", self.ivls(right))]
        for w in wendes:
            items.append((f"Wendestelle {self.s(w)}", f"Vorzeichenwechsel von f'' bei x = {_valstr(w) if _inexact(w) else self.s(w)}"))
        note = ""
        noflip = [w for w in self.roots2 if w not in wendes]
        if noflip:
            note = ("f''=0 bei " + ", ".join(self.s(w) for w in noflip) +
                    ", aber ohne Vorzeichenwechsel: keine Wendestelle, die Krümmung bleibt gleich.")
        steps.append(Step("Krümmungsverhalten ablesen",
                          "f''(x) > 0 ⇒ Linkskrümmung; f''(x) < 0 ⇒ Rechtskrümmung. "
                          "Wendestellen sind Nullstellen von f'' mit Vorzeichenwechsel.",
                          items, note=note, rule="Krümmungskriterium"))
        return steps

    @cached_property
    def wendestellen(self):
        out = []
        for w in self.roots2:
            ra, rb = self._neighbors(self.table2, w)
            if ra and rb and ra.sign * rb.sign < 0:
                out.append(w)
        return out

    @cached_property
    def inflection(self) -> tuple[list, list]:
        """(Wendepunkte, f''=0 ohne Vorzeichenwechsel) als Listen von dicts."""
        pts, non = [], []
        for w in self.roots2:
            ra, rb = self._neighbors(self.table2, w)
            f3 = _sign_at(self.d3, w)
            sl = ra.sign if ra else 0
            sr = rb.sign if rb else 0
            slope = _simp(self.d1.subs(x, w)) if not _inexact(w) else self.d1.subs(x, w)
            slope_zero = _sign_at(self.d1, w) == 0
            d = dict(x0=w, y0=_value_at(self.f, w), f3=f3, left=sl, right=sr, slope=slope,
                     horizontal=slope_zero, via="f'''" if f3 != 0 else "Vorzeichenwechsel")
            is_infl = (f3 != 0) or (sl * sr < 0)
            d["inflection"] = is_infl
            d["change"] = ("Rechts → Links" if (f3 > 0 or (f3 == 0 and sl < 0 < sr)) else "Links → Rechts") \
                if is_infl else ""
            (pts if is_infl else non).append(d)
        return pts, non

    def steps_inflection(self, derivative=True) -> list[Step]:
        steps = []
        if derivative:
            steps.append(Step("Ableitungen bilden", "Wendestellen: $f''(x_0)=0$ und $f'''(x_0)\\neq 0$.",
                              [("f''(x)", self.d2), ("f'''(x)", self.d3)], rule="Ableitungsregeln"))
        r, exact, info = self._r2
        steps.append(self._roots_step("f''", self.d2, r, exact, info,
                                      "Kandidaten für Wendestellen sind die Lösungen von $f''(x_0)=0$.", "w"))
        pts, non = self.inflection
        if not r:
            steps.append(Step("Wendepunkte", "Keine Nullstelle von f'' ⇒ keine Wendepunkte."))
        for d in pts + non:
            w = d["x0"]
            s0 = self.s(w)
            items = [("f'''(x₀)", _item(self.d3.subs(x, w) if _inexact(w) else _simp(self.d3.subs(x, w))))]
            if d["f3"] != 0:
                detail = (f"Bei $x_0={self.t(w)}$ gilt $f''(x_0)=0$ und $f'''(x_0)\\neq 0$ ⇒ Wendestelle "
                          f"(Krümmungswechsel {d['change']}).")
                rule = "f''(x₀)=0 ∧ f'''(x₀)≠0 ⇒ Wendepunkt"
            else:
                detail = (f"Bei $x_0={self.t(w)}$ gilt $f''(x_0)=0$ und $f'''(x_0)=0$ ⇒ das Kriterium "
                          f"entscheidet nicht. Darum: Vorzeichen von $f''$ links und rechts.")
                ra, rb = self._neighbors(self.table2, w)
                if ra and rb:
                    items.append(("f'' links von x₀", f"f''({self.s(ra.test)}) = {_valstr(ra.value)} {_REL[ra.sign]} 0"))
                    items.append(("f'' rechts von x₀", f"f''({self.s(rb.test)}) = {_valstr(rb.value)} {_REL[rb.sign]} 0"))
                rule = "Vorzeichenwechsel von f''"
            if d["inflection"]:
                items.append(("Wendepunkt", f"({s0 if not _inexact(w) else _valstr(w)} / {_valstr(d['y0'])})"))
                note = ""
                if d["horizontal"]:
                    note = "Hier ist zugleich f'(x₀)=0: horizontale Tangente, also ein Sattelpunkt."
                else:
                    note = f"Steigung der Wendetangente: f'(x₀) = {_valstr(d['slope'])}."
            else:
                note = ("Kein Vorzeichenwechsel von f'' ⇒ keine Wendestelle "
                        "(Beispiel x⁴ bei 0).")
                if d["f3"] == 0 and not d["inflection"]:
                    items.append(("Ergebnis", "kein Wendepunkt"))
            steps.append(Step(f"Prüfung bei x₀ = {s0}", detail, items, note=note, rule=rule))
        return steps

    # -- Extrema --------------------------------------------------------
    def boundary_points(self):
        out = []
        for end, side in ((self.lo, "+"), (self.hi, "-")):
            if math.isinf(_fl(end)):
                continue
            try:
                val = self.f.subs(x, end)
                if val.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
                    raise ValueError
                val = sp.simplify(val)
            except Exception:
                try:
                    val = sp.limit(self.f, x, end, side)
                except Exception:
                    continue
            out.append((end, val))
        return out

    def steps_extrema(self, derivative=True) -> list[Step]:
        steps = []
        if derivative:
            steps.append(Step("Ableitungen bilden",
                              "Kriterium: $f'(x_0)=0$ und Vorzeichen von $f''(x_0)$.",
                              [("f'(x)", self.d1), ("f''(x)", self.d2)], rule="Ableitungsregeln"))
        r, exact, info = self._r1
        steps.append(self._roots_step("f'", self.d1, r, exact, info,
                                      "Kandidaten für Extrema: $f'(x_0)=0$.", "x"))
        if not self.points:
            steps.append(Step("Lokale Extrema", "Keine kritische Stelle ⇒ keine relativen Extrema."))
        for p in self.points:
            steps.append(self.step_classify(p, extrema_mode=True))
        bd = self.boundary_points()
        if bd:
            items = [(f"f({self.s(a)})", _item(v)) for a, v in bd]
            steps.append(Step("Randpunkte als Kandidaten für globale Extrema",
                              "Globale Extrema liegen bei relativen Extrema oder am Rand des Bereichs.",
                              items, note="Randpunkte sind keine relativen Extrema im Sinne des Kriteriums; "
                                          "bei offenem Intervall werden sie nicht angenommen (nur Grenzwert)."))
        return steps

    def extrema_lists(self):
        mn = [(p.x0, p.y0) for p in self.points if p.kind == MIN]
        mx = [(p.x0, p.y0) for p in self.points if p.kind == MAX]
        sd = [(p.x0, p.y0) for p in self.points if p.kind == SAD]
        un = [(p.x0, p.y0) for p in self.points if p.kind == UNB]
        return mn, mx, sd, un

    def global_candidates(self):
        if math.isinf(_fl(self.lo)) or math.isinf(_fl(self.hi)):
            return None, None
        cands = list(self.boundary_points()) + [(p.x0, p.y0) for p in self.points
                                                if p.kind in (MIN, MAX, SAD)]
        cands = [c for c in cands if not c[1].has(sp.zoo, sp.nan)]
        if not cands:
            return None, None
        gmax = max(cands, key=lambda c: float(sp.N(c[1], 30)))
        gmin = min(cands, key=lambda c: float(sp.N(c[1], 30)))
        return gmax, gmin


def _as_analysis(f, domain=None) -> _Analysis:
    return f if isinstance(f, _Analysis) else _Analysis(f, domain)


# ==========================================================================
# Ergebnisse
# ==========================================================================

class _Figure:
    def figure(self, xlim=None, dark: bool = False, **kwargs):
        """f, f' und f'' untereinander (Monotonie-/Krümmungsfärbung, kritische Punkte)."""
        from . import plots
        a = self._a
        return plots.plot_curve(a.f, xlim if xlim is not None else a.default_xlim(),
                                dark=dark, **kwargs)


def _setup(res: Result, a: _Analysis):
    res._a = a
    res.f = a.f
    res.d1, res.d2, res.d3 = a.d1, a.d2, a.d3


def _finish(res: Result, a: _Analysis):
    """Warnungen am Ende übernehmen (Nullstellen werden erst beim Rechnen gesucht)."""
    res.warnings = list(dict.fromkeys(a.warnings))


class MonotonyResult(_Figure, Result):
    """Monotonieverhalten: `.increasing`, `.decreasing`, `.critical_x`, `.table`."""

    def __init__(self, a: _Analysis, derivative=True):
        super().__init__()
        _setup(self, a)
        self.title = f"Monotonie von f(x) = {fmt(a.f)}"
        self.table = a.table1
        self.critical_x = list(a.roots1)
        self.increasing = _merge(a.table1, 1)
        self.decreasing = _merge(a.table1, -1)
        self.constant = _merge(a.table1, 0)
        self.steps = a.steps_monotony(derivative)
        self.summary = (f"streng monoton wachsend auf {a.ivls(self.increasing)}; "
                        f"streng monoton fallend auf {a.ivls(self.decreasing)}")
        self.highlights = [("streng monoton wachsend auf", a.ivls(self.increasing)),
                           ("streng monoton fallend auf", a.ivls(self.decreasing))]


class CriticalResult(_Figure, Result):
    """Kritische Stellen/Punkte: `.points` (PointInfo), `.critical_x`, `.minima`, `.maxima`, `.saddles`."""

    def __init__(self, a: _Analysis, derivative=True):
        super().__init__()
        _setup(self, a)
        self.title = f"Kritische Punkte von f(x) = {fmt(a.f)}"
        self.points = list(a.points)
        self.critical_x = list(a.roots1)
        self.minima, self.maxima, self.saddles, self.undetermined = a.extrema_lists()
        self.steps = a.steps_critical(derivative)
        self.summary = (f"{len(self.points)} kritische(r) Punkt(e): {len(self.minima)} Minimum/Minima, "
                        f"{len(self.maxima)} Maximum/Maxima, {len(self.saddles)} Sattelpunkt(e)")
        self.highlights = [(f"kritischer Punkt bei x₀ = {a.s(p.x0)}",
                            f"({a.s(p.x0) if not _inexact(p.x0) else _valstr(p.x0)} / {_valstr(p.y0)}): {p.kind}")
                           for p in self.points]


class CurvatureResult(_Figure, Result):
    """Krümmung: `.left`, `.right`, `.inflection_x`, `.inflection_points`, `.table`."""

    def __init__(self, a: _Analysis, derivative=True):
        super().__init__()
        _setup(self, a)
        self.title = f"Krümmungsverhalten von f(x) = {fmt(a.f)}"
        self.table = a.table2
        self.left = _merge(a.table2, 1)
        self.right = _merge(a.table2, -1)
        self.zeros = list(a.roots2)
        self.inflection_x = list(a.wendestellen)
        pts, _ = a.inflection
        self.inflection_points = [(d["x0"], d["y0"]) for d in pts]
        self.steps = a.steps_curvature(derivative)
        self.summary = (f"Linkskrümmung auf {a.ivls(self.left)}; Rechtskrümmung auf {a.ivls(self.right)}")
        self.highlights = [("Linkskrümmung auf", a.ivls(self.left)),
                           ("Rechtskrümmung auf", a.ivls(self.right))]
        if self.inflection_x:
            self.highlights.append(("Wendestellen", ", ".join(
                (_valstr(w) if _inexact(w) else a.s(w)) for w in self.inflection_x)))


class ExtremaResult(_Figure, Result):
    """Relative Extrema: `.minima`, `.maxima`, `.saddles`, `.undetermined` (Listen von (x0, f(x0)))."""

    def __init__(self, a: _Analysis, derivative=True):
        super().__init__()
        _setup(self, a)
        self.title = f"Lokale Extrema von f(x) = {fmt(a.f)}"
        self.points = list(a.points)
        self.critical_x = list(a.roots1)
        self.minima, self.maxima, self.saddles, self.undetermined = a.extrema_lists()
        #: Stellen mit f''(x0)=0, an denen das Kriterium allein nichts aussagt
        self.no_statement = [p.point for p in a.points if p.supplemented]
        #: davon über die Zusatzprüfung (Vorzeichenwechsel von f') entschieden
        self.supplemented = [p for p in a.points if p.supplemented and p.kind != UNB]
        self.boundary = a.boundary_points()
        self.global_max, self.global_min = a.global_candidates()
        self.steps = a.steps_extrema(derivative)
        self.summary = (f"{len(self.minima)} relative(s) Minimum/Minima, {len(self.maxima)} relative(s) "
                        f"Maximum/Maxima, {len(self.saddles)} Sattelpunkt(e)")
        self.highlights = []
        for p in self.points:
            pos = f"({a.s(p.x0) if not _inexact(p.x0) else _valstr(p.x0)} / {_valstr(p.y0)})"
            self.highlights.append((p.kind, pos + (" [Zusatzprüfung über f']" if p.supplemented else "")))
        if self.global_max is not None:
            self.highlights.append(("globales Maximum (Kandidat)",
                                    f"({a.s(self.global_max[0])} / {_valstr(self.global_max[1])})"))
            self.highlights.append(("globales Minimum (Kandidat)",
                                    f"({a.s(self.global_min[0])} / {_valstr(self.global_min[1])})"))


class InflectionResult(_Figure, Result):
    """Wendepunkte: `.points` (Liste (x0, f(x0))), `.details`, `.no_inflection`."""

    def __init__(self, a: _Analysis, derivative=True):
        super().__init__()
        _setup(self, a)
        self.title = f"Wendepunkte von f(x) = {fmt(a.f)}"
        pts, non = a.inflection
        self.details = pts
        self.points = [(d["x0"], d["y0"]) for d in pts]
        self.inflection_x = [d["x0"] for d in pts]
        self.no_inflection = [(d["x0"], d["y0"]) for d in non]
        self.steps = a.steps_inflection(derivative)
        self.summary = f"{len(self.points)} Wendepunkt(e)"
        self.highlights = [("Wendepunkt", f"({a.s(p[0]) if not _inexact(p[0]) else _valstr(p[0])} / {_valstr(p[1])})")
                           for p in self.points]


class DiscussionResult(_Figure, Result):
    """Gesamtübersicht: Monotonie, kritische Punkte, Krümmung, lokale Extrema."""

    def __init__(self, a: _Analysis):
        super().__init__()
        _setup(self, a)
        self.title = f"Kurvendiskussion von f(x) = {fmt(a.f)}"
        self.monotony = MonotonyResult(a, derivative=False)
        self.critical = CriticalResult(a, derivative=False)
        self.curvature = CurvatureResult(a, derivative=False)
        self.extrema = ExtremaResult(a, derivative=False)
        self.inflection = InflectionResult(a, derivative=False)
        self.minima, self.maxima = self.extrema.minima, self.extrema.maxima
        self.saddles, self.undetermined = self.extrema.saddles, self.extrema.undetermined
        self.increasing, self.decreasing = self.monotony.increasing, self.monotony.decreasing
        self.left, self.right = self.curvature.left, self.curvature.right
        self.inflection_points = self.curvature.inflection_points
        self.critical_x = list(a.roots1)

        S = self.steps
        S.append(Step("Beispielfunktion",
                      f"Gegeben ist $f(x)$ auf dem Bereich {a.domain_str}; zuerst die Ableitungen.",
                      [("f(x)", a.f), ("f'(x)", a.d1), ("f''(x)", a.d2), ("Bereich", a.domain_str)],
                      rule="Ableitungsregeln"))
        S.append(Step("Erste Ableitung und Monotonie",
                      "f'(x) > 0 ⇒ streng monoton wachsend, f'(x) < 0 ⇒ streng monoton fallend.",
                      [("f'(x)", a.d1)]))
        S.extend(self.monotony.steps)
        S.append(Step("Kritische Punkte",
                      "Kritische Stelle x₀: f'(x₀) = 0; kritischer Punkt: (x₀ / f(x₀)).",
                      [("f'(x₀)", 0)]))
        S.extend(self.critical.steps[1:])
        S.append(Step("Zweite Ableitung und Krümmungsverhalten",
                      "f''(x) > 0 ⇒ Linkskrümmung, f''(x) < 0 ⇒ Rechtskrümmung.",
                      [("f''(x)", a.d2)]))
        S.extend(self.curvature.steps)
        S.append(Step("Lokale Extrema",
                      "f'(x₀)=0 ∧ f''(x₀)>0 ⇒ relatives Minimum; f'(x₀)=0 ∧ f''(x₀)<0 ⇒ relatives Maximum; "
                      "f'(x₀)=0 ∧ f''(x₀)=0 ⇒ keine Aussage möglich."))
        S.extend([st for st in self.extrema.steps if st.title.startswith("Einordnung")
                  or st.title.startswith("Randpunkte")] or
                 [Step("Lokale Extrema", "Keine kritische Stelle ⇒ keine relativen Extrema.")])

        #: Zusammenfassung als Tabelle (Liste von Zeilen)
        rows = [("Monotonie ↗", a.ivls(self.increasing)), ("Monotonie ↘", a.ivls(self.decreasing)),
                ("Linkskrümmung", a.ivls(self.left)), ("Rechtskrümmung", a.ivls(self.right))]
        for p in a.points:
            rows.append((f"x₀ = {a.s(p.x0)}", f"({_valstr(p.x0) if _inexact(p.x0) else a.s(p.x0)} / "
                                                f"{_valstr(p.y0)}): {p.kind}"
                         + ("  [über Zusatzprüfung]" if p.supplemented else "")))
        for w in a.wendestellen:
            rows.append((f"Wendestelle {a.s(w)}", f"({_valstr(w) if _inexact(w) else a.s(w)} / "
                                                   f"{_valstr(_value_at(a.f, w))})"))
        self.summary_table = rows
        S.append(Step("Zusammenfassung", "Alle Ergebnisse auf einen Blick.", rows))
        self.summary = (f"{len(self.minima)} Minimum/Minima, {len(self.maxima)} Maximum/Maxima, "
                        f"{len(self.saddles)} Sattelpunkt(e), {len(self.inflection_points)} Wendepunkt(e)")
        self.highlights = [(lbl, val) for lbl, val in rows]


# ==========================================================================
# Öffentliche Funktionen
# ==========================================================================

def monotony(f, domain=None) -> MonotonyResult:
    """Monotonieverhalten über das Vorzeichen von f' (Vorzeichentabelle)."""
    a = _as_analysis(f, domain)
    r = MonotonyResult(a)
    _finish(r, a)
    return r


def critical_points(f, domain=None) -> CriticalResult:
    """Kritische Stellen/Punkte (f'(x0)=0), eingeordnet als Minimum, Maximum oder Sattelpunkt."""
    a = _as_analysis(f, domain)
    r = CriticalResult(a)
    _finish(r, a)
    return r


def curvature(f, domain=None) -> CurvatureResult:
    """Krümmungsverhalten über das Vorzeichen von f''; benennt Wendestellen."""
    a = _as_analysis(f, domain)
    r = CurvatureResult(a)
    _finish(r, a)
    return r


def extrema(f, domain=None) -> ExtremaResult:
    """Relative Extrema mit dem Kriterium der 2. Ableitung (Folie 13) plus Zusatzprüfung über f'."""
    a = _as_analysis(f, domain)
    r = ExtremaResult(a)
    _finish(r, a)
    return r


def inflection_points(f, domain=None) -> InflectionResult:
    """Wendepunkte: f''(x0)=0 und f'''(x0)≠0 (sonst Vorzeichenwechsel von f'')."""
    a = _as_analysis(f, domain)
    r = InflectionResult(a)
    _finish(r, a)
    return r


def curve_discussion(f, domain=None) -> DiscussionResult:
    """Gesamte Kurvendiskussion der Folien SW4 (Monotonie, kritische Punkte, Krümmung, Extrema)."""
    a = _as_analysis(f, domain)
    r = DiscussionResult(a)
    _finish(r, a)
    return r
