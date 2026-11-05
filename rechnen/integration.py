#!/usr/bin/env python3
"""
Semesterwoche 8: Bestimmtes Integral über Rechtecksummen (Folien SW8).

Gesucht ist der Flächeninhalt A = ∫ₐᵇ f(x) dx zwischen dem Graphen von f,
der x-Achse und den Vertikalen bei a und b. Idee der Folien (Seite 5, 7, 8):
die Fläche in n Rechtecke zerlegen, Ober- und Untersumme bilden, den
Grenzwert für n → ∞ nehmen.

    from anakit.integration import (rectangle_sums, integral_by_limit,
                                    faulhaber, definite_integral)

    rectangle_sums("x^2", 1, 2, 4)          # Unter-/Obersumme mit 4 Rechtecken
    integral_by_limit("x^2", 1, 2)          # Folie 8: Ō(n) = 7/3 + 3/(2n) + 1/(6n²) → 7/3
    faulhaber(2)                            # Folie 10: n³/3 + n²/2 + n/6
    definite_integral("x^2", 1, 2)          # Folie 11 (Ausblick): [x³/3] = 7/3

Δx = (b − a)/n,  x_k = a + k·Δx.  Für streng monoton steigendes f gilt
Obersumme = Δx · Σ f(x_k) (k = 1 … n) und Untersumme = Δx · Σ f(x_{k−1});
bei fallendem f vertauschen sich beide. Die Verfahren rechnen exakt.
"""

from __future__ import annotations

import sympy as sp

from .core import fmt, num, parse, x
from .render import Result, Step

__all__ = ["rectangle_sums", "integral_by_limit", "faulhaber", "definite_integral",
           "RectangleSumsResult", "IntegralLimitResult", "FaulhaberResult",
           "DefiniteIntegralResult"]

_n = sp.Symbol("n", positive=True, integer=True)
_k = sp.Symbol("k", positive=True, integer=True)


def _L(s: str) -> str:
    return s if "\\" in s else r"\!" + s


def _monotony(f, a, b) -> int:
    """+1 steigend, −1 fallend auf [a, b] (über das Vorzeichen von f'); sonst ValueError."""
    iv = sp.Interval(a, b)
    try:
        d = sp.diff(f, x)
        neg = sp.solveset(d < 0, x, iv)
        pos = sp.solveset(d > 0, x, iv)
        if neg == sp.S.EmptySet:
            return 1
        if pos == sp.S.EmptySet:
            return -1
    except Exception:
        pass
    raise ValueError("Ober- und Untersumme rechnet anakit nur für auf [a, b] monotone Funktionen "
                     "(Folie 8: f streng monoton steigend). TODO: sonst in monotone Teilintervalle zerlegen.")


def _interval(a, b):
    a, b = num(a), num(b)
    if not (a.is_number and b.is_number) or not a < b:
        raise ValueError("Die Grenzen müssen Zahlen mit a < b sein.")
    return a, b


# ==========================================================================
# Faulhaber
# ==========================================================================

class FaulhaberResult(Result):
    """Summe 1ᵖ + 2ᵖ + … + nᵖ als Polynom in n (Folie 10)."""

    title = "Faulhabersche Formeln"

    def __init__(self, p: int, formula, factored, rows):
        super().__init__()
        self.p, self.formula, self.factored = p, formula, factored
        #: Formeln für alle Exponenten 0 … p (Liste von (q, Polynom in n))
        self.rows = rows
        self.n = _n


def faulhaber(p: int, n=None) -> FaulhaberResult:
    """
    Formel für 1ᵖ + 2ᵖ + … + nᵖ (Folie 10) als Polynom in n.

        faulhaber(2).formula          # n**3/3 + n**2/2 + n/6
        faulhaber(3, 10).highlights   # Wert für n = 10: 3025
    """
    if not isinstance(p, int) or isinstance(p, bool) or p < 0:
        raise ValueError("Der Exponent p muss eine ganze Zahl ≥ 0 sein.")
    rows = [(q, sp.expand(sp.summation(_k ** q, (_k, 1, _n)))) for q in range(p + 1)]
    formula = rows[-1][1]
    fact = sp.factor(formula)
    res = FaulhaberResult(p, formula, fact, rows)
    res.steps = [Step(f"Summe der {p}-ten Potenzen",
                      "Bis zu diesem Exponenten gelten (Folie 10):",
                      [(_L(f"\\sum_{{k=1}}^{{n}} k^{{{q}}}"), f) for q, f in rows],
                      note="Die Formeln gehen auf Johannes Faulhaber (1580 – 1635) zurück.",
                      rule="Faulhabersche Formeln")]
    res.highlights = [(_L(f"1^{{{p}}} + 2^{{{p}}} + \\ldots + n^{{{p}}}"), formula),
                      ("faktorisiert", fact)]
    res.summary = f"Σ k^{p} (k = 1 … n) = {fmt(formula)}"
    if n is not None:
        n_ = int(n)
        direct = sum(i ** p for i in range(1, n_ + 1))
        val = formula.subs(_n, n_)
        res.steps.append(Step(f"Kontrolle für n = {n_}", "Formel und Summe direkt ausrechnen.",
                              [("Formel", val), ("Summe", direct)]))
        if val != direct:  # pragma: no cover - mathematisch ausgeschlossen
            res.warnings.append("Formel und Summe stimmen nicht überein.")
        res.highlights.append((f"n = {n_}", val))
    return res


# ==========================================================================
# Rechtecksummen mit festem n
# ==========================================================================

class RectangleSumsResult(Result):
    """Unter- und Obersumme mit n Rechtecken (Folie 5 und 7)."""

    title = "Rechtecksummen"

    def __init__(self, f, a, b, ns):
        super().__init__()
        self.f, self.a, self.b, self.ns = f, a, b, ns
        #: Zeilen (n, Untersumme, Obersumme, Differenz)
        self.rows: list[tuple] = []
        self.lower = self.upper = self.error = None

    def figure(self, n=None, dark: bool = False, **kw):
        from ..grafik.plots import plot_riemann
        return plot_riemann(self.f, self.a, self.b, n or self.ns[0], dark=dark, **kw)


def _sums(f, a, b, n: int, sense: int):
    dx = (b - a) / n
    xs = [a + i * dx for i in range(n + 1)]
    ys = [sp.simplify(f.subs(x, v)) for v in xs]
    right = dx * sum(ys[1:])
    left = dx * sum(ys[:-1])
    lower, upper = (left, right) if sense > 0 else (right, left)
    return dx, xs, sp.nsimplify(sp.simplify(lower)), sp.nsimplify(sp.simplify(upper))


def rectangle_sums(f, a, b, n=4) -> RectangleSumsResult:
    """
    Untersumme und Obersumme mit n Rechtecken für ein auf [a, b] monotones f
    (Folien SW8, Seite 5 und 7). `n` darf eine Liste sein (Tabelle wie im Notebook
    integration-rechtecke.ipynb): dann zeigt der Rechenweg das erste n ausführlich.

        rectangle_sums("x^2", 1, 2, 4).upper      # 87/32 (exakter Bruch)
        rectangle_sums("x^2", 1, 2, [4, 10, 100]).rows
    """
    f = parse(f)
    a, b = _interval(a, b)
    ns = [n] if isinstance(n, int) else list(n) if isinstance(n, (list, tuple, range)) else [0]
    if not ns or any((not isinstance(v, int)) or isinstance(v, bool) or v < 1 for v in ns):
        raise ValueError("n muss eine ganze Zahl ≥ 1 sein (oder eine Liste davon).")
    sense = _monotony(f, a, b)
    res = RectangleSumsResult(f, a, b, ns)
    n0 = ns[0]
    dx, xs, lo0, up0 = _sums(f, a, b, n0, sense)
    shown = xs[:8] + (["…"] if len(xs) > 8 else [])
    S = res.steps
    S.append(Step("Fläche und Idee",
                  "Gesucht: Fläche zwischen Graph, x-Achse und den Vertikalen bei a und b. "
                  "Sie wird durch Rechtecke angenähert.",
                  [("A", _L(rf"\int_{{{fmt(a)}}}^{{{fmt(b)}}} f(x)\,\mathrm{{d}}x")), ("f(x)", f)],
                  rule="bestimmtes Integral"))
    S.append(Step("Intervall aufteilen",
                  "$\\Delta x = \\frac{b-a}{n}$ und $x_k = a + k\\,\\Delta x$.",
                  [("Δx", dx), ("n", n0)] + [(f"x{_sub(i)}", v) for i, v in enumerate(shown)
                                                          if v != "…"] ,
                  rule="Zerlegung"))
    which = ("f steigt: Obersumme mit den rechten Randwerten f(x_k), Untersumme mit den linken f(x_{k−1})."
             if sense > 0 else
             "f fällt: Obersumme mit den linken Randwerten f(x_{k−1}), Untersumme mit den rechten f(x_k).")
    S.append(Step("Ober- und Untersumme", which,
                  [("Obersumme", up0), ("Untersumme", lo0)],
                  rule="Obersumme / Untersumme"))
    S.append(Step("Fehlerschätzung", "Die Differenz Obersumme − Untersumme begrenzt den Fehler; "
                  "der gesuchte Flächeninhalt liegt dazwischen.",
                  [("Obersumme − Untersumme", sp.simplify(up0 - lo0)),
                   ("Schranken", f"{fmt(lo0)} ≤ A ≤ {fmt(up0)}")],
                  rule="Fehlerschätzung"))
    for v in ns:
        _dx, _xs, lo, up = _sums(f, a, b, v, sense)
        res.rows.append((v, lo, up, sp.simplify(up - lo)))
    if len(ns) > 1:
        S.append(Step("Mehr Rechtecke", "Je grösser n, desto kleiner die Differenz.",
                      [(f"n = {v}", f"{fmt(lo)} ≤ A ≤ {fmt(up)}  (Differenz {fmt(d)})")
                       for v, lo, up, d in res.rows]))
    _n0, res.lower, res.upper, res.error = res.rows[0]
    res.summary = f"{fmt(lo0)} ≤ A ≤ {fmt(up0)} für n = {n0}."
    res.highlights = [("Untersumme", res.lower), ("Obersumme", res.upper),
                      ("Differenz", res.error)]
    return res


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


# ==========================================================================
# Grenzwert der Obersumme
# ==========================================================================

class IntegralLimitResult(Result):
    """Grenzwert der Ober- bzw. Untersumme für n → ∞ (Folie 8)."""

    title = "Bestimmtes Integral als Grenzwert der Obersumme"

    def __init__(self, f, a, b):
        super().__init__()
        self.f, self.a, self.b = f, a, b
        self.n = _n
        self.upper = self.lower = None      # Ō(n) und Ū(n) als Terme in n
        self.limit = None                   # A

    def figure(self, n: int = 8, dark: bool = False, **kw):
        from ..grafik.plots import plot_riemann
        return plot_riemann(self.f, self.a, self.b, n, dark=dark, **kw)


def integral_by_limit(f, a, b) -> IntegralLimitResult:
    """
    Flächeninhalt eines Polynoms über den Grenzwert der Obersumme, mit den
    Faulhaberschen Formeln (Folie 8 und 10).

        r = integral_by_limit("x^2", 1, 2)
        r.upper     # 7/3 + 3/(2n) + 1/(6n²)
        r.limit     # 7/3

    Nur für Polynome (die Faulhaber-Formeln liefern die Summen); f muss auf [a, b] monoton sein.
    """
    f = parse(f)
    a, b = _interval(a, b)
    if not f.is_polynomial(x):
        raise ValueError("integral_by_limit rechnet nur Polynome (Faulhabersche Formeln, Folie 10). "
                         "Für andere Funktionen: rectangle_sums (Näherung) oder definite_integral (Hauptsatz).")
    sense = _monotony(f, a, b)
    res = IntegralLimitResult(f, a, b)
    S = res.steps
    dx = (b - a) / _n
    S.append(Step("Zu berechnen", "Flächeninhalt als bestimmtes Integral.",
                  [("A", _L(rf"\int_{{{fmt(a)}}}^{{{fmt(b)}}} f(x)\,\mathrm{{d}}x")), ("f(x)", f)],
                  rule="bestimmtes Integral"))
    S.append(Step("Integrationsintervall aufteilen",
                  "$\\Delta x = \\frac{b-a}{n}$, $x_k = a + k\\,\\Delta x$.",
                  [("Δx", dx), (_L("x_k"), sp.simplify(a + _k * dx))], rule="Zerlegung"))
    term = sp.expand(f.subs(x, a + _k * dx))
    kind = "Obersumme" if sense > 0 else "Untersumme"
    S.append(Step(f"{kind} mit den rechten Randwerten",
                  "f ist monoton " + ("steigend" if sense > 0 else "fallend") +
                  f": die rechten Randwerte $f(x_k)$, $k = 1 \\ldots n$, liefern die {kind}.",
                  [(_L(r"\sum_{k=1}^{n} f(x_k)"), _L(r"\sum_{k=1}^{n}\left(" + sp.latex(term) + r"\right)")),
                   ("Δx", dx)], rule="Obersumme" if sense > 0 else "Untersumme"))
    # Summe über k mit Faulhaber: term ist Polynom in k mit Koeffizienten in n
    poly = sp.Poly(term, _k)
    parts, total = [], sp.S.Zero
    for (p,), c in sorted(poly.terms(), key=lambda t: t[0]):
        sk = sp.expand(sp.summation(_k ** p, (_k, 1, _n)))
        parts.append((_L(rf"\sum_{{k=1}}^{{n}} k^{{{p}}}"), sk))
        total += c * sk
    S.append(Step("Summen mit den Faulhaberschen Formeln",
                  "Summanden nach Potenzen von $k$ sortieren, konstante Faktoren vor die Summe ziehen "
                  "und $\\sum k^p$ nach Folie 10 einsetzen.",
                  parts, rule="Faulhabersche Formeln"))
    right = sp.apart(sp.simplify(dx * total), _n)
    left = sp.apart(sp.simplify(right - dx * (f.subs(x, b) - f.subs(x, a))), _n)
    upper, lower = (right, left) if sense > 0 else (left, right)
    res.upper, res.lower = upper, lower
    S.append(Step("Obersumme und Untersumme als Term in n",
                  "Ausmultiplizieren und nach Potenzen von $1/n$ ordnen.",
                  [("Obersumme Ō(n)", upper), ("Untersumme Ū(n)", lower)],
                  rule="Obersumme"))
    lim_up, lim_lo = sp.limit(upper, _n, sp.oo), sp.limit(lower, _n, sp.oo)
    S.append(Step("Grenzwert n → ∞",
                  "Beide Summen streben gegen denselben Wert: das ist der Flächeninhalt.",
                  [("lim Obersumme", lim_up),
                   ("lim Untersumme", lim_lo)], rule="Grenzwert"))
    if lim_up != lim_lo:  # pragma: no cover - ausgeschlossen bei stetigem f
        res.warnings.append("Ober- und Untersumme haben verschiedene Grenzwerte.")
    res.limit = lim_up
    check = sp.integrate(f, (x, a, b))
    S.append(Step("Kontrolle (Ausblick Hauptsatz)", "Mit der Stammfunktion gerechnet (Folie 11).",
                  [(_L(rf"\int_{{{fmt(a)}}}^{{{fmt(b)}}} f(x)\,\mathrm{{d}}x"), check)]))
    res.summary = f"A = {fmt(lim_up)}"
    res.highlights = [("Obersumme Ō(n)", upper), ("Untersumme Ū(n)", lower), ("A", lim_up)]
    return res


# ==========================================================================
# Hauptsatz (Ausblick)
# ==========================================================================

class DefiniteIntegralResult(Result):
    """∫ₐᵇ f dx = F(b) − F(a) (Ausblick auf Folie 11)."""

    title = "Bestimmtes Integral mit dem Hauptsatz"

    def __init__(self, f, a, b):
        super().__init__()
        self.f, self.a, self.b = f, a, b
        self.antiderivative = None
        self.value = None

    def figure(self, n: int = 8, dark: bool = False, **kw):
        from ..grafik.plots import plot_riemann
        return plot_riemann(self.f, self.a, self.b, n, dark=dark, **kw)


def definite_integral(f, a, b) -> DefiniteIntegralResult:
    """
    «Ingenieurmässig» mit dem Hauptsatz der Differential- und Integralrechnung
    (Ausblick, Folie 11): Stammfunktion F bilden, F(b) − F(a) rechnen.

        definite_integral("x^2", 1, 2).value     # 7/3

    Die Stammfunktion bestimmt SymPy; die Integrationsregeln selbst sind Stoff der nächsten Woche.
    """
    f = parse(f)
    a, b = _interval(a, b)
    F = sp.integrate(f, x)
    if F.has(sp.Integral):
        raise ValueError("Für diese Funktion findet SymPy keine Stammfunktion.")
    res = DefiniteIntegralResult(f, a, b)
    res.antiderivative = F
    fa, fb = sp.simplify(F.subs(x, a)), sp.simplify(F.subs(x, b))
    res.value = sp.simplify(fb - fa)
    res.steps = [
        Step("Stammfunktion", "$F$ mit $F'(x) = f(x)$.", [("f(x)", f), ("F(x)", F)],
             note="Ausblick: das Integrationsverfahren folgt in einer späteren Woche.",
             rule="Hauptsatz der Differential- und Integralrechnung"),
        Step("Werte an den Grenzen", "", [(f"F({fmt(b)})", fb), (f"F({fmt(a)})", fa)]),
        Step("Differenz", "$\\int_a^b f(x)\\,dx = F(b) - F(a)$.",
             [("A", f"{fmt(fb)} − ({fmt(fa)}) = {fmt(res.value)}")], rule="Hauptsatz"),
    ]
    res.summary = f"A = {fmt(res.value)}"
    res.highlights = [("A", res.value)]
    return res
