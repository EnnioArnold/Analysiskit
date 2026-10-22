#!/usr/bin/env python3
"""
Semesterwoche 6: Taylor- und MacLaurin-Polynome.

Herleitung der Folien: f(x) = a₀ + a₁(x−x₀) + a₂(x−x₀)² + …; wiederholtes
Ableiten und Einsetzen von x₀ gibt f⁽ᵏ⁾(x₀) = k!·aₖ, also

    Taylor-Reihe:      f(x) = Σ_{k=0}^{∞} f⁽ᵏ⁾(x₀)/k! · (x − x₀)ᵏ
    MacLaurin-Reihe:   der Fall x₀ = 0:  Σ f⁽ᵏ⁾(0)/k! · xᵏ

`taylor` bricht nach dem Glied n-ten Grades ab (Taylor-Polynom Tₙ),
`maclaurin` tut das für x₀ = 0:

    from anakit.reihen import taylor, maclaurin

    maclaurin("exp(x)", 4).polynomial   # 1 + x + x²/2 + x³/6 + x⁴/24
    maclaurin("sin(x)", 5).polynomial   # x − x³/6 + x⁵/120
    taylor("ln(x)", 1, 3).polynomial    # (x−1) − (x−1)²/2 + (x−1)³/3
"""

from __future__ import annotations

import sympy as sp

from .core import fmt, num, parse, tex, x
from .render import Result, Step

__all__ = ["taylor", "maclaurin", "TaylorResult"]


def _L(s: str) -> str:
    return s if "\\" in s else r"\!" + s


def _sub(k: int) -> str:
    return str(k).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


class TaylorResult(Result):
    """Taylor-Polynom Tₙ um x₀ mit Ableitungswerten und Koeffizienten."""

    title = "Taylor-Polynom"

    def __init__(self, f, x0, n, derivs, coeffs, poly, maclaurin_form):
        super().__init__()
        self.f, self.x0, self.n = f, x0, n
        #: f⁽ᵏ⁾(x₀) für k = 0..n
        self.derivative_values = derivs
        #: aₖ = f⁽ᵏ⁾(x₀)/k!
        self.coefficients = coeffs
        #: Tₙ(x) in Potenzen von (x − x₀), nicht ausmultipliziert
        self.polynomial = poly
        self.maclaurin = maclaurin_form
        self.title = f"{'MacLaurin' if maclaurin_form else 'Taylor'}-Polynom {n}. Grades"
        self.summary = (f"T_{n}(x) = Σ f⁽ᵏ⁾({fmt(x0)})/k! · (x − {fmt(x0)})ᵏ"
                        if not maclaurin_form else f"T_{n}(x) = Σ f⁽ᵏ⁾(0)/k! · xᵏ  (k = 0 … {n})")
        self.highlights = [(_L(f"T_{{{n}}}(x)"), poly)]

    @property
    def expr(self):
        return self.polynomial

    @property
    def value(self):
        return self.polynomial

    def evaluate(self, at):
        """Tₙ an einer Stelle (exakt)."""
        return self.polynomial.subs(x, num(at))

    def error_at(self, at) -> float:
        """|f(at) − Tₙ(at)| als Gleitkommazahl."""
        a = num(at)
        return abs(float(sp.N(self.f.subs(x, a) - self.polynomial.subs(x, a), 30)))

    def figure(self, xlim=None, orders=None, dark: bool = False, **kw):
        from ..grafik.plots import plot_taylor
        orders = orders or sorted({o for o in (1, 2, 3, self.n) if 0 <= o <= self.n})
        return plot_taylor(self.f, self.x0, orders, xlim=xlim, dark=dark, **kw)


def taylor(f, x0=0, n: int = 3) -> TaylorResult:
    """
    Taylor-Polynom n-ten Grades von f um x₀ (Folien SW6, Seite 7/8).

        taylor("exp(x)", 0, 3).polynomial     # 1 + x + x²/2 + x³/6
        taylor("1/x", 1, 3).coefficients      # [1, -1, 1, -1]
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Der Grad n muss eine ganze Zahl ≥ 0 sein.")
    f = parse(f)
    a = num(x0)
    derivs, cur = [], f
    for k in range(n + 1):
        if k:
            cur = sp.diff(cur, x)
        v = sp.simplify(cur.subs(x, a))
        if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError(f"f⁽{k}⁾ ist an der Stelle x₀ = {fmt(a)} nicht definiert: "
                             "dort gibt es keine Taylor-Entwicklung.")
        derivs.append(v)
    coeffs = [sp.simplify(d / sp.factorial(k)) for k, d in enumerate(derivs)]
    poly = sum((c * (x - a) ** k for k, c in enumerate(coeffs)), sp.S.Zero)
    mac = a == 0
    name = "MacLaurin" if mac else "Taylor"
    A = tex(a)

    steps = [
        Step("Ansatz", "Potenzreihe um $x_0$; die Koeffizienten folgen durch Ableiten und "
             "Einsetzen von $x_0$ (die anderen Summanden fallen weg).",
             [("f(x)", _L(r"a_0 + a_1 (x-x_0) + a_2 (x-x_0)^2 + a_3 (x-x_0)^3 + \ldots"))]),
        Step("Koeffizienten", "$f^{(k)}(x_0) = k!\\cdot a_k$, also $a_k = \\frac{f^{(k)}(x_0)}{k!}$.",
             [(_L(r"a_k"), _L(r"\frac{f^{(k)}(x_0)}{k!}"))], rule=f"{name}-Reihe"),
    ]
    cur = f
    for k in range(n + 1):
        label = f"f^{{({k})}}(x)" if k > 3 else "f" + "'" * k + "(x)"
        if k:
            cur = sp.diff(cur, x)
        steps.append(Step(
            f"Ableitung {k}",
            "" if k else "Funktionswert an der Entwicklungsstelle.",
            [(_L(label), cur), (_L(label.replace("(x)", f"({A})")), derivs[k]),
             (_L(f"a_{{{k}}}"), coeffs[k])]))
    series = _L(r"\sum_{k=0}^{" + str(n) + r"} \frac{f^{(k)}(" + A + r")}{k!}\,(x-" + (A if a != 0 else "0")
                + r")^k") if not mac else _L(r"\sum_{k=0}^{" + str(n) + r"} \frac{f^{(k)}(0)}{k!}\,x^k")
    steps.append(Step(f"{name}-Polynom aufstellen", f"Reihe nach dem Glied {n}. Grades abbrechen.",
                      [(_L(f"T_{{{n}}}(x)"), series), (_L(f"T_{{{n}}}(x)"), poly)],
                      rule=f"{name}-Polynom"))
    res = TaylorResult(f, a, n, derivs, coeffs, poly, mac)
    res.steps = steps
    return res


def maclaurin(f, n: int = 3) -> TaylorResult:
    """MacLaurin-Polynom (x₀ = 0): ``maclaurin("cos(x)", 4)`` → 1 − x²/2 + x⁴/24."""
    return taylor(f, 0, n)
