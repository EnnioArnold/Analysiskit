#!/usr/bin/env python3
"""
Semesterwoche 5: Numerische Mathematik I.

Nullstellen mit Bisektion und Newton-Verfahren, numerisches Differenzieren
mit Vorwärts-, Rückwärts- und zentraler Differenz samt Fehlerordnung und
optimaler Schrittweite. Gerechnet wird hier bewusst mit Gleitkommazahlen
(float, doppelte Genauigkeit), denn genau darum geht es in der Woche; nur
die Ableitung für das Newton-Verfahren und der Vergleichswert f'(x0) sind
exakt (SymPy).

    from anakit.numerik import bisection, newton, numeric_diff

    bisection("x^2 - 1.5", 1, 4, tol=1e-6)     # ≈ 1.224745
    newton("x^2 - 1.5", 3)                     # ≈ 1.2247448713915890
    numeric_diff("x^3 - 5", 1, kind="zentral") # ≈ 3, Fehler ~ h²

Folien SW5:

    Bisektion:     m_k = (a_k + b_k)/2; Vorzeichenwechsel links => [a_k, m_k],
                   sonst [m_k, b_k]; Ende, wenn b_k − a_k klein genug.
    Newton:        x_{k+1} = x_k − f(x_k)/f'(x_k); Ende, wenn |x_{k+1} − x_k| klein.
    Differenzen:   vorwärts (f(x0+h) − f(x0))/h,  rückwärts (f(x0) − f(x0−h))/h,
                   zentral (f(x0+h) − f(x0−h))/(2h).
    Fehlerordnung: vorwärts/rückwärts O(h), optimale Schrittweite √ε_Mach;
                   zentral O(h²), optimale Schrittweite ∛ε_Mach.
"""

from __future__ import annotations

import math
import sys

import sympy as sp

from .core import fmt, parse, to_callable, x
from .render import Result, Step

__all__ = ["bisection", "newton", "numeric_diff", "optimal_step", "diff_error_table",
           "BisectionResult", "NewtonResult", "NumDiffResult", "DiffErrorTableResult",
           "EPS_MACH", "DIFF_KINDS"]

#: Maschinengenauigkeit von float (doppelte Genauigkeit), ca. 2.22e-16.
EPS_MACH = sys.float_info.epsilon

#: Die drei Differenzen mit Fehlerordnung p (Fehler ~ h^p).
DIFF_KINDS = {"vorwaerts": 1, "rueckwaerts": 1, "zentral": 2}
_ALIASES = {"vorwärts": "vorwaerts", "rückwärts": "rueckwaerts", "forward": "vorwaerts",
            "backward": "rueckwaerts", "central": "zentral"}
_KIND_NAME = {"vorwaerts": "Vorwärtsdifferenz", "rueckwaerts": "Rückwärtsdifferenz",
              "zentral": "Zentrale Differenz"}


def _L(s: str) -> str:
    return s if "\\" in s else r"\!" + s


def _n(v: float, digits: int = 12) -> str:
    """Gleitkommazahl als kurzer Text (Dezimalpunkt, echtes Minuszeichen)."""
    return f"{float(v):.{digits}g}".replace("-", "−")


def _func(f):
    """(Term, float-Funktion) aus Text, Zahl, Term oder Python-Funktion."""
    expr = parse(f)
    return expr, to_callable(expr)


def _kind(kind: str) -> str:
    k = _ALIASES.get(str(kind).lower(), str(kind).lower())
    if k not in DIFF_KINDS:
        raise ValueError("kind muss 'vorwaerts', 'rueckwaerts' oder 'zentral' sein")
    return k


def _finite(v: float, what: str) -> float:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{what} ist nicht endlich (Funktion dort nicht definiert?)")
    return v


# ==========================================================================
# Bisektion
# ==========================================================================

class BisectionResult(Result):
    """Nullstellensuche durch Intervallhalbierung (Folie 6)."""

    title = "Bisektion"

    def __init__(self, expr, a, b, tol, rows, root):
        super().__init__()
        self.f, self.a, self.b, self.tol = expr, a, b, tol
        #: Zeilen (k, a_k, b_k, m_k, f(m_k)); die letzte gehört zum letzten Intervall.
        self.rows = rows
        self.root = root
        self.iterations = len(rows) - 1  # Halbierungen; die letzte Zeile ist nur das Ergebnis
        self.last_interval = (rows[-1][1], rows[-1][2]) if rows else (a, b)
        self.summary = ("Auf [a, b] stetig mit Vorzeichenwechsel: die Intervallmitte des "
                        "letzten Intervalls ist der Näherungswert.")
        self.highlights = [(_L(r"x_N \approx"), _n(root)), ("Schritte", self.iterations),
                           (_L(r"f(x_N)"), _n(rows[-1][4] if rows else 0))]

    @property
    def value(self) -> float:
        return self.root

    def figure(self, dark: bool = False, **kw):
        from ..grafik.plots import plot_bisection
        return plot_bisection(self.f, self.a, self.b, self.rows, dark=dark, **kw)


def bisection(f, a, b, tol: float = 1e-6, max_iter: int = 200) -> BisectionResult:
    """
    Bisektion auf [a, b]: f muss stetig sein und an den Enden verschiedene
    Vorzeichen haben. Abbruch, wenn die Intervalllänge ≤ tol.

        bisection("x^2 - 1.5", 1, 4, 1e-6).root      # ≈ 1.2247448
        bisection("x^3 - 5", 1, 2).iterations         # 20
    """
    expr, fn = _func(f)
    a, b, tol = float(a), float(b), float(tol)
    if not a < b:
        raise ValueError("Für die Bisektion muss a < b gelten.")
    if tol <= 0:
        raise ValueError("Die Toleranz muss positiv sein.")
    fa, fb = _finite(fn(a), "f(a)"), _finite(fn(b), "f(b)")
    if fa * fb > 0:
        raise ValueError("f(a) und f(b) haben dasselbe Vorzeichen: ohne Vorzeichenwechsel "
                         "garantiert die Bisektion keine Nullstelle in [a, b].")
    steps = [
        Step("Voraussetzungen prüfen",
             "f stetig auf $[a, b]$ und $f(a)\\cdot f(b) < 0$ (Vorzeichenwechsel).",
             [(_L("f(a)"), _n(fa)), (_L("f(b)"), _n(fb))], rule="Zwischenwertsatz"),
        Step("Idee", "Mitte $m_k = \\frac{a_k + b_k}{2}$; hat $f$ auf $[a_k, m_k]$ einen "
             "Vorzeichenwechsel, ist das neue Intervall $[a_k, m_k]$, sonst $[m_k, b_k]$.",
             [(_L("[a_0, b_0]"), f"[{_n(a)}, {_n(b)}]")], rule="Bisektion"),
    ]
    rows = []
    lo, hi, flo = a, b, fa
    k = 0
    while hi - lo > tol and k < max_iter:
        m = lo + (hi - lo) / 2
        fm = _finite(fn(m), f"f({_n(m)})")
        rows.append((k, lo, hi, m, fm))
        left = flo * fm <= 0
        steps.append(Step(
            f"Schritt {k}", "Vorzeichenwechsel links, neues Intervall $[a_k, m_k]$." if left
            else "Kein Vorzeichenwechsel links, neues Intervall $[m_k, b_k]$.",
            [(_L(f"a_{{{k}}}"), _n(lo)), (_L(f"b_{{{k}}}"), _n(hi)),
             (_L(f"m_{{{k}}}"), _n(m)), (_L(f"f(m_{{{k}}})"), _n(fm))]))
        if fm == 0:
            lo = hi = m
            break
        if left:
            hi = m
        else:
            lo, flo = m, fm
        k += 1
    root = lo + (hi - lo) / 2
    rows.append((k, lo, hi, root, _finite(fn(root), "f(x_N)")))
    steps.append(Step("Ergebnis", f"Intervalllänge {_n(hi - lo, 3)} ≤ {_n(tol, 3)}; die Mitte "
                      "des letzten Intervalls ist der Näherungswert.",
                      [(_L(f"[a_{{{k}}}, b_{{{k}}}]"), f"[{_n(lo)}, {_n(hi)}]"),
                       (_L(r"x_N"), _n(root))], rule="Bisektion"))
    res = BisectionResult(expr, a, b, tol, rows, root)
    res.steps = steps
    if k >= max_iter:
        res.warnings.append("Höchstzahl der Schritte erreicht, Toleranz evtl. unter der Maschinengenauigkeit.")
    return res


# ==========================================================================
# Newton-Verfahren
# ==========================================================================

class NewtonResult(Result):
    """Nullstellensuche durch Linearisieren (Folie 9)."""

    title = "Newton-Verfahren"

    def __init__(self, expr, dexpr, x0, tol, rows, root, converged):
        super().__init__()
        self.f, self.df, self.x0, self.tol = expr, dexpr, x0, tol
        #: Zeilen (k, x_k, f(x_k), f'(x_k), x_{k+1})
        self.rows = rows
        self.root = root
        self.iterations = len(rows)
        self.converged = converged
        self.summary = ("Die Tangente in x_k wird durch ihre Nullstelle ersetzt: "
                        "x_{k+1} = x_k − f(x_k)/f'(x_k).")
        self.highlights = [(_L(r"x_N \approx"), _n(root)), ("Schritte", self.iterations),
                           (_L(r"f(x_N)"), _n(rows[-1][2] if rows else 0))]

    @property
    def value(self) -> float:
        return self.root

    def figure(self, dark: bool = False, **kw):
        from ..grafik.plots import plot_newton
        return plot_newton(self.f, self.rows, dark=dark, **kw)


def newton(f, x0, tol: float = 1e-10, max_iter: int = 50, numerisch: bool = False) -> NewtonResult:
    """
    Newton-Verfahren mit Startwert x0. Abbruch, wenn |x_{k+1} − x_k| ≤ tol.
    f'(x) wird exakt abgeleitet; mit numerisch=True stattdessen zentral mit
    der optimalen Schrittweite ∛ε·max(1, |x|) angenähert (wie im Notebook).

        newton("x^2 - 1.5", 3).root      # 1.2247448713915890
        newton("x^3 - 5", 3).root        # 1.7099759466766968
    """
    expr, fn = _func(f)
    dexpr = sp.diff(expr, x)
    dfn = to_callable(dexpr)
    xk, tol = float(x0), float(tol)
    if tol <= 0:
        raise ValueError("Die Toleranz muss positiv sein.")

    def d(v):
        if numerisch:
            hh = optimal_step("zentral") * max(1.0, abs(v))
            return (fn(v + hh) - fn(v - hh)) / (2 * hh)
        return dfn(v)

    steps = [
        Step("Voraussetzungen", "f in der Nähe der Nullstelle stetig differenzierbar, "
             "$f'$ dort nicht 0; Startwert $x_0$ nahe der Nullstelle.",
             [("f(x)", expr), ("f'(x)", dexpr), (_L("x_0"), _n(xk))]),
        Step("Idee", "Linearisieren: $t(x) = f(x_k) + f'(x_k)(x - x_k)$; die Nullstelle der "
             "Tangente ist der neue Näherungswert.",
             [(_L("x_{k+1}"), _L(r"x_k - \frac{f(x_k)}{f'(x_k)}"))], rule="Newton-Verfahren"),
    ]
    rows, converged = [], False
    for k in range(max_iter):
        fx, dfx = _finite(fn(xk), f"f({_n(xk)})"), _finite(d(xk), f"f'({_n(xk)})")
        if dfx == 0:
            raise ValueError(f"f'({_n(xk)}) = 0: die Tangente ist waagrecht, das Verfahren "
                             "bricht ab. Anderen Startwert wählen.")
        nxt = xk - fx / dfx
        rows.append((k, xk, fx, dfx, nxt))
        steps.append(Step(
            f"Schritt {k}", "",
            [(_L(f"x_{{{k}}}"), _n(xk)), (_L(f"f(x_{{{k}}})"), _n(fx)),
             (_L(f"f'(x_{{{k}}})"), _n(dfx)), (_L(f"x_{{{k + 1}}}"), _n(nxt))]))
        done = abs(nxt - xk) <= tol
        xk = nxt
        if done:
            converged = True
            break
    steps.append(Step("Ergebnis", f"|x_{{k+1}} − x_k| ≤ {_n(tol, 3)}." if converged
                      else "Höchstzahl der Schritte erreicht, ohne die Toleranz zu erreichen.",
                      [(_L(r"x_N"), _n(xk))], rule="Newton-Verfahren"))
    res = NewtonResult(expr, dexpr, float(x0), tol, rows, xk, converged)
    res.steps = steps
    if not converged:
        res.warnings.append("Das Verfahren ist nicht konvergiert: anderen Startwert versuchen.")
    return res


# ==========================================================================
# Numerisch differenzieren
# ==========================================================================

def optimal_step(kind: str = "zentral") -> float:
    """Optimale Schrittweite: √ε_Mach (vorwärts/rückwärts), ∛ε_Mach (zentral). Folie 11."""
    return EPS_MACH ** (1 / DIFF_KINDS_ROOT[_kind(kind)])


#: Wurzelexponent der optimalen Schrittweite je Differenz (2 -> √, 3 -> ∛).
DIFF_KINDS_ROOT = {"vorwaerts": 2, "rueckwaerts": 2, "zentral": 3}


def _difference(fn, x0: float, h: float, kind: str) -> float:
    if kind == "vorwaerts":
        return (fn(x0 + h) - fn(x0)) / h
    if kind == "rueckwaerts":
        return (fn(x0) - fn(x0 - h)) / h
    return (fn(x0 + h) - fn(x0 - h)) / (2 * h)


_FORMULA = {"vorwaerts": r"\frac{f(x_0+h)-f(x_0)}{h}",
            "rueckwaerts": r"\frac{f(x_0)-f(x_0-h)}{h}",
            "zentral": r"\frac{f(x_0+h)-f(x_0-h)}{2h}"}


class NumDiffResult(Result):
    """Näherung von f'(x₀) durch eine Differenz (Folie 11)."""

    title = "Numerisch differenzieren"

    def __init__(self, expr, x0, h, kind, approx, exact):
        super().__init__()
        self.f, self.x0, self.h, self.kind = expr, x0, h, kind
        self.approx, self.exact = approx, exact
        self.error = abs(approx - float(exact)) if exact is not None else None
        self.order = DIFF_KINDS[kind]
        self.title = _KIND_NAME[kind]
        self.summary = (f"Fehlerordnung h^{self.order}, optimale Schrittweite "
                        f"{'√' if DIFF_KINDS_ROOT[kind] == 2 else '∛'}ε_Mach ≈ {_n(optimal_step(kind), 3)}.")
        self.highlights = [(_L(r"f'(x_0) \approx"), _n(approx))]
        if self.error is not None:
            self.highlights.append(("Fehler", _n(self.error, 3)))

    @property
    def value(self) -> float:
        return self.approx


def numeric_diff(f, x0, h: float | None = None, kind: str = "zentral") -> NumDiffResult:
    """
    f'(x0) numerisch: kind 'vorwaerts', 'rueckwaerts' oder 'zentral'. Ohne h
    wird die optimale Schrittweite genommen. Zum Vergleich steht f'(x0) exakt da.

        numeric_diff("x^3 - 5", 1, 1e-6).approx      # ≈ 3.0
        numeric_diff("sin(x)", 0, kind="vorwaerts")
    """
    expr, fn = _func(f)
    kind = _kind(kind)
    x0 = float(x0)
    h = optimal_step(kind) if h is None else float(h)
    if h <= 0:
        raise ValueError("Die Schrittweite h muss positiv sein.")
    approx = _finite(_difference(fn, x0, h, kind), "Die Differenz")
    try:
        exact = sp.diff(expr, x).subs(x, sp.nsimplify(x0, rational=True))
        exact = sp.simplify(exact) if exact.is_number else None
        if exact is not None and not math.isfinite(float(exact)):
            exact = None
    except Exception:
        exact = None
    steps = [
        Step("Formel", "Gegeben $f$, $x_0$ und die kleine Hilfsgrösse $h > 0$.",
             [(_L(f"{_KIND_NAME[kind]}"), _L(r"f'(x_0) \approx " + _FORMULA[kind]))],
             rule=_KIND_NAME[kind]),
        Step("Einsetzen", "", [(_L("x_0"), _n(x0)), ("h", _n(h, 6))]),
    ]
    if kind == "vorwaerts":
        vals = [(_L("f(x_0+h)"), _n(fn(x0 + h))), (_L("f(x_0)"), _n(fn(x0)))]
    elif kind == "rueckwaerts":
        vals = [(_L("f(x_0)"), _n(fn(x0))), (_L("f(x_0-h)"), _n(fn(x0 - h)))]
    else:
        vals = [(_L("f(x_0+h)"), _n(fn(x0 + h))), (_L("f(x_0-h)"), _n(fn(x0 - h)))]
    steps.append(Step("Funktionswerte", "", vals))
    steps.append(Step("Näherung", "", [(_L(r"f'(x_0) \approx"), _n(approx))]))
    if exact is not None:
        steps.append(Step("Vergleich mit der exakten Ableitung", "",
                          [(_L("f'(x_0)"), exact), ("Fehler", _n(abs(approx - float(exact)), 3))],
                          note=f"Fehlerordnung h^{DIFF_KINDS[kind]}: halbe Schrittweite, "
                               f"{'halber' if DIFF_KINDS[kind] == 1 else 'ein Viertel'} Fehler "
                               "(solange Rundungsfehler nicht überwiegen)."))
    res = NumDiffResult(expr, x0, h, kind, approx, exact)
    res.steps = steps
    return res


class DiffErrorTableResult(Result):
    """Fehler der Differenz für fallende Schrittweiten h = 10⁻¹ … 10⁻¹⁶."""

    title = "Fehler und Schrittweite"

    def __init__(self, expr, x0, kind, rows):
        super().__init__()
        self.f, self.x0, self.kind = expr, x0, kind
        #: Zeilen (h, Näherung, Fehler)
        self.rows = rows
        self.best_h, _, self.best_error = min(rows, key=lambda r: r[2])
        self.optimal_h = optimal_step(kind)
        self.title = f"Fehler der {_KIND_NAME[kind]}"
        self.summary = ("Verkleinert man h, sinkt zuerst der Verfahrensfehler, dann steigen die "
                        "Rundungsfehler: beste Schrittweite in der Tabelle "
                        f"h = {_n(self.best_h, 3)} (Theorie ≈ {_n(self.optimal_h, 3)}).")
        self.highlights = [("beste Schrittweite h", _n(self.best_h, 3)),
                           ("kleinster Fehler", _n(self.best_error, 3)),
                           ("optimal (Theorie)", _n(self.optimal_h, 3))]


def diff_error_table(f, x0, kind: str = "zentral", exponents=range(1, 17)) -> DiffErrorTableResult:
    """
    Tabelle h = 10^-k mit Näherung und Fehler gegen f'(x0) exakt.

        diff_error_table("exp(x)", 1, "vorwaerts").best_h    # ≈ 1e-8
    """
    expr, fn = _func(f)
    kind = _kind(kind)
    x0 = float(x0)
    exact = float(sp.diff(expr, x).subs(x, sp.nsimplify(x0, rational=True)))
    rows = []
    for k in exponents:
        hh = 10.0 ** (-k)
        approx = _difference(fn, x0, hh, kind)
        rows.append((hh, approx, abs(approx - exact)))
    res = DiffErrorTableResult(expr, x0, kind, rows)
    res.steps = [Step("Tabelle", f"Exakt: f'(x₀) = {fmt(sp.nsimplify(exact))}.",
                      [(_L(f"h = 10^{{-{k}}}"), f"{_n(a, 12)}   Fehler {_n(e, 3)}")
                       for (hh, a, e), k in zip(rows, exponents)])]
    return res
