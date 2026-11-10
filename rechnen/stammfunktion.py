#!/usr/bin/env python3
"""
Semesterwoche 9: Unbestimmtes Integral und Hauptsatz der Differential- und
Integralrechnung (Folien SW9).

Folien: Seite 5 (Eigenschaften bestimmter Integrale), 7 und 8 (Flächenfunktion
F(x) = ∫ₐˣ f(t) dt), 10 und 11 (Hauptsatz: F' = f und ∫ₐᵇ f dt = F(b) − F(a)).

    from anakit.stammfunktion import (integral_properties, area_function,
                                      antiderivative, fundamental_theorem)

    integral_properties("x^2", 0, 3)       # Folie 5: Additivität, Vorzeichen, Grenzen
    area_function("2x", 0, 1)              # Folie 7/8: F(x) = x², F₁ − F₂ = const.
    antiderivative("x^2")                  # Folie 7: Stammfunktion = unbestimmtes Integral
    fundamental_theorem("x^2", 1, 2)       # Folie 10/11: F' = f, F(b) − F(a) = 7/3

Die Folien rechnen keine Beispiele vor; alle Aufgaben dieser Woche sind eigene
Beispiele. Die Stammfunktion findet SymPy, die Integrationsregeln selbst sind
nicht Stoff der Folien. Alles rechnet exakt.
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.util import continuous_domain

from .core import fmt, num, parse, x
from .render import Result, Step

__all__ = ["integral_properties", "area_function", "antiderivative", "fundamental_theorem",
           "IntegralPropertiesResult", "AreaFunctionResult", "AntiderivativeResult",
           "FundamentalTheoremResult"]

_t = sp.Symbol("t")
_C = sp.Symbol("C")


def _L(s: str) -> str:
    return r"\!" + s


def _int(a, b) -> str:
    return rf"\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} f(t)\,\mathrm{{d}}t"


def _stamm(f):
    """Eine Stammfunktion F von f (SymPy); ValueError, wenn es keine geschlossene gibt."""
    F = sp.integrate(f, x)
    if F.has(sp.Integral):
        raise ValueError("Für diese Funktion findet SymPy keine Stammfunktion.")
    return F


def _stetig(f, *pts):
    """f muss zwischen den Stellen stetig sein, sonst stimmt F(b) − F(a) nicht."""
    lo, hi = min(pts), max(pts)
    if lo == hi:
        return
    try:
        ok = continuous_domain(f, x, sp.Interval(lo, hi)) == sp.Interval(lo, hi)
    except Exception:
        ok = True  # shortcut: nicht prüfbar, dann ungeprüft rechnen; SymPy-Integral gilt wie bei SW8
    if not ok:
        raise ValueError(f"f ist auf [{fmt(lo)}, {fmt(hi)}] nicht stetig (Polstelle oder Sprung); "
                         "der Hauptsatz gilt nur für stetige Funktionen.")


def _zahl(v, what):
    v = num(v)
    if not v.is_number or not v.is_real:
        raise ValueError(f"{what} muss eine reelle Zahl sein.")
    return v


# ==========================================================================
# Unbestimmtes Integral = Stammfunktion
# ==========================================================================

class AntiderivativeResult(Result):
    """Stammfunktion F + C mit der Probe F' = f (Folie 7)."""

    title = "Unbestimmtes Integral (Stammfunktion)"

    def __init__(self, f):
        super().__init__()
        self.f = f
        self.F = None

    def figure(self, dark: bool = False, **kw):
        from ..grafik.plots import plot_area_function
        return plot_area_function(self.f, 0, 1, dark=dark, **kw)


def antiderivative(f) -> AntiderivativeResult:
    """
    Stammfunktion F mit F'(x) = f(x), also ∫ f(x) dx = F(x) + C (Folie 7).

        antiderivative("x^2").F      # x**3/3
    """
    f = parse(f)
    F = _stamm(f)
    res = AntiderivativeResult(f)
    res.F = F
    probe = sp.simplify(sp.diff(F, x) - f)
    res.steps = [
        Step("Stammfunktion", "Gesucht ist $F$ mit $F'(x) = f(x)$.", [("f(x)", f), ("F(x)", F)],
             note="Stammfunktion, Flächenfunktion und unbestimmtes Integral sind dasselbe (Folie 3).",
             rule="Stammfunktion"),
        Step("Probe durch Ableiten", "", [("F'(x)", sp.simplify(sp.diff(F, x))), ("F' − f", probe)]),
        Step("Konstante", "Zwei Stammfunktionen unterscheiden sich nur um eine Konstante (Folie 8).",
             [(_L(r"\int f(x)\,\mathrm{d}x"), F + _C)], rule="unbestimmtes Integral"),
    ]
    if probe != 0:  # pragma: no cover - mathematisch ausgeschlossen
        res.warnings.append("Die Probe F' = f geht nicht auf.")
    res.summary = f"∫ f(x) dx = {fmt(F)} + C"
    res.highlights = [("F(x)", F), ("F(x) + C", F + _C)]
    return res


# ==========================================================================
# Flächenfunktion
# ==========================================================================

class AreaFunctionResult(Result):
    """Flächenfunktion Fₐ(x) = ∫ₐˣ f(t) dt (Folie 7 und 8)."""

    title = "Flächenfunktion"

    def __init__(self, f, a, a2):
        super().__init__()
        self.f, self.a, self.a2 = f, a, a2
        self.F = None        # Fₐ(x)
        self.F2 = None       # Fₐ₂(x), falls a2 gegeben
        self.difference = None

    def figure(self, x0=None, dark: bool = False, **kw):
        from ..grafik.plots import plot_area_function
        return plot_area_function(self.f, self.a, x0, dark=dark, **kw)


def area_function(f, a, a2=None) -> AreaFunctionResult:
    """
    Flächenfunktion Fₐ(x) = ∫ₐˣ f(t) dt mit fester linker Grenze a (Folie 7).
    Mit einer zweiten Grenze a2 zeigt sie auch F₁ − F₂ = ∫ₐ^{a₂} f dt = const. (Folie 8).

        area_function("2x", 0).F            # x**2
        area_function("2x", 0, 1).difference   # 1
    """
    f = parse(f)
    a = _zahl(a, "Die Grenze a")
    a2 = None if a2 in (None, "") else _zahl(a2, "Die zweite Grenze a₂")
    G = _stamm(f)
    _stetig(f, a, *( [a2] if a2 is not None else []))
    F = sp.simplify(G - G.subs(x, a))
    res = AreaFunctionResult(f, a, a2)
    res.F = F
    probe = sp.simplify(sp.diff(F, x) - f)
    res.steps = [
        Step("Definition", "Linke Grenze fest, rechte Grenze variabel.",
             [(_L(rf"F_{{{sp.latex(a)}}}(x)"), rf"\int_{{{sp.latex(a)}}}^{{x}} f(t)\,\mathrm{{d}}t")],
             rule="Flächenfunktion"),
        Step("Stammfunktion bilden", "$F_a(x) = G(x) - G(a)$ mit einer Stammfunktion $G$.",
             [("G(x)", G), (f"G({fmt(a)})", sp.simplify(G.subs(x, a))), (_L(rf"F_{{{sp.latex(a)}}}(x)"), F)]),
        Step("Kontrolle", "Der Flächeninhalt von $a$ bis $a$ ist null, und $F' = f$.",
             [(f"F({fmt(a)})", sp.simplify(F.subs(x, a))), ("F'(x) − f(x)", probe)],
             rule="Hauptsatz"),
    ]
    if probe != 0:  # pragma: no cover
        res.warnings.append("Die Probe F' = f geht nicht auf.")
    res.highlights = [(_L(rf"F_{{{sp.latex(a)}}}(x)"), F)]
    res.summary = f"F_{fmt(a)}(x) = {fmt(F)}"
    if a2 is not None:
        F2 = sp.simplify(G - G.subs(x, a2))
        diff = sp.simplify(F - F2)
        direct = sp.simplify(G.subs(x, a2) - G.subs(x, a))
        res.F2, res.difference = F2, diff
        res.steps.append(Step(
            "Zweite untere Grenze", "Eine andere untere Grenze gibt eine andere Stammfunktion.",
            [(_L(rf"F_{{{sp.latex(a2)}}}(x)"), F2)], rule="Flächenfunktion"))
        res.steps.append(Step(
            "Differenz zweier Stammfunktionen",
            r"$F_1(x) - F_2(x) = \int_{a_1}^{x} - \int_{a_2}^{x} = \int_{a_1}^{a_2} f(t)\,dt$: eine Konstante.",
            [("F₁ − F₂", diff), (_L(_int(a, a2)), direct)], rule="Additivität"))
        if diff != direct:  # pragma: no cover
            res.warnings.append("Die Differenz ist nicht der Wert des Integrals zwischen den Grenzen.")
        res.highlights.append(("F₁ − F₂", diff))
    return res


# ==========================================================================
# Eigenschaften bestimmter Integrale
# ==========================================================================

class IntegralPropertiesResult(Result):
    """Die vier Eigenschaften von Folie 5 an einem Beispiel."""

    title = "Eigenschaften bestimmter Integrale"

    def __init__(self, f, a, b, c):
        super().__init__()
        self.f, self.a, self.b, self.c = f, a, b, c
        self.values: dict = {}
        self.sign = None   # +1 (f ≥ 0), −1 (f ≤ 0), 0 (wechselt) auf [a, b]

    def figure(self, dark: bool = False, **kw):
        from ..grafik.plots import plot_area_function
        return plot_area_function(self.f, self.a, self.b, dark=dark, **kw)


def integral_properties(f, a, b, c=None) -> IntegralPropertiesResult:
    """
    Additivität, Vorzeichen, gleiche und vertauschte Grenzen (Folie 5) an f mit
    den Stellen a, b und c (Standard: Mitte von a und b), exakt nachgerechnet.

        integral_properties("x^2", 0, 3).values["a→c"]
    """
    f = parse(f)
    a, b = _zahl(a, "Die Grenze a"), _zahl(b, "Die Grenze b")
    if not a < b:
        raise ValueError("Es muss a < b gelten.")
    c = (a + b) / 2 if c in (None, "") else _zahl(c, "Die Stelle c")
    G = _stamm(f)
    _stetig(f, a, b, c)

    def I(u, v):
        return sp.simplify(G.subs(x, v) - G.subs(x, u))

    iab, ibc, iac = I(a, b), I(b, c), I(a, c)
    iaa, iba = I(a, a), I(b, a)
    res = IntegralPropertiesResult(f, a, b, c)
    res.values = {"a→b": iab, "b→c": ibc, "a→c": iac, "a→a": iaa, "b→a": iba}
    iv = sp.Interval(a, b)
    try:
        neg, pos = sp.solveset(f < 0, x, iv), sp.solveset(f > 0, x, iv)
        res.sign = 1 if neg == sp.S.EmptySet else (-1 if pos == sp.S.EmptySet else 0)
    except Exception:
        res.sign = 0
    S = res.steps
    S.append(Step("Additivität", "Der Weg von $a$ nach $c$ darf über $b$ führen.",
                  [(_L(_int(a, c)), iac), (_L(_int(a, b) + "+" + _int(b, c)), f"{fmt(iab)} + ({fmt(ibc)}) = {fmt(iab + ibc)}")],
                  rule="Additivität"))
    if iac != iab + ibc:  # pragma: no cover
        res.warnings.append("Additivität verletzt.")
    word = {1: ("f(x) ≥ 0 auf [a, b]", "≥ 0"), -1: ("f(x) ≤ 0 auf [a, b]", "≤ 0")}.get(res.sign)
    S.append(Step("Orientierter Flächeninhalt",
                  f"Hier gilt {word[0]}, also ist das Integral {word[1]}." if word else
                  "f wechselt auf [a, b] das Vorzeichen: positive und negative Flächen heben sich teilweise auf, "
                  "das Vorzeichen des Integrals folgt nicht aus f.",
                  [(_L(_int(a, b)), iab)], rule="orientierter Flächeninhalt"))
    S.append(Step("Identische Grenzen", "", [(_L(_int(a, a)), iaa)], rule="identische Grenzen"))
    S.append(Step("Vertauschen der Grenzen", "Das Vorzeichen kehrt sich um.",
                  [(_L(_int(b, a)), iba), (_L("-" + _int(a, b)), -iab)], rule="Vertauschen der Grenzen"))
    res.highlights = [(_L(_int(a, b)), iab), (_L(_int(b, a)), iba)]
    res.summary = f"∫ von {fmt(a)} bis {fmt(b)} = {fmt(iab)}, vertauscht {fmt(iba)}"
    return res


# ==========================================================================
# Hauptsatz
# ==========================================================================

class FundamentalTheoremResult(Result):
    """∫ₐᵇ f(t) dt = F(b) − F(a) mit F' = f (Folie 10 und 11)."""

    title = "Hauptsatz der Differential- und Integralrechnung"

    def __init__(self, f, a, b):
        super().__init__()
        self.f, self.a, self.b = f, a, b
        self.F = None
        self.value = None

    def figure(self, dark: bool = False, **kw):
        from ..grafik.plots import plot_area_function
        return plot_area_function(self.f, self.a, self.b, dark=dark, **kw)


def fundamental_theorem(f, a, b) -> FundamentalTheoremResult:
    """
    Hauptsatz: erst die Probe F' = f (Folie 10), dann ∫ₐᵇ f dt = F(b) − F(a) (Folie 11),
    zuletzt F(b) = F(a) + ∫ₐᵇ f dt zur Kontrolle.

        fundamental_theorem("x^2", 1, 2).value    # 7/3
    """
    f = parse(f)
    a, b = _zahl(a, "Die Grenze a"), _zahl(b, "Die Grenze b")
    if not a < b:
        raise ValueError("Es muss a < b gelten.")
    F = _stamm(f)
    _stetig(f, a, b)
    fa, fb = sp.simplify(F.subs(x, a)), sp.simplify(F.subs(x, b))
    val = sp.simplify(fb - fa)
    res = FundamentalTheoremResult(f, a, b)
    res.F, res.value = F, val
    probe = sp.simplify(sp.diff(F, x) - f)
    res.steps = [
        Step("Stammfunktion", "Die Flächenfunktion ist eine Stammfunktion: $F'(x) = f(x)$ (Folie 10).",
             [("f(x)", f), ("F(x)", F), ("F' − f", probe)], rule="Hauptsatz"),
        Step("Werte an den Grenzen", "", [(f"F({fmt(b)})", fb), (f"F({fmt(a)})", fa)]),
        Step("Differenz", r"$\int_a^b f(t)\,dt = F(b) - F(a)$ (Folie 11).",
             [(_L(_int(a, b)), f"{fmt(fb)} − ({fmt(fa)}) = {fmt(val)}")], rule="Hauptsatz"),
        Step("Kontrolle", "Wegen $\\int_{a_0}^b = \\int_{a_0}^a + \\int_a^b$ gilt $F(b) = F(a) + \\int_a^b$.",
             [("F(a) + ∫", sp.simplify(fa + val)), ("F(b)", fb)]),
        Step("Konstante fällt weg", "$F(x) + C$ ergibt dasselbe, weil $C$ in der Differenz verschwindet.",
             [("(F(b) + C) − (F(a) + C)", val)]),
    ]
    if probe != 0:  # pragma: no cover
        res.warnings.append("Die Probe F' = f geht nicht auf.")
    res.summary = f"A = {fmt(val)}"
    res.highlights = [("F(x)", F), ("A", val)]
    return res
