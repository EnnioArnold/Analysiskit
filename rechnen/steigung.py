#!/usr/bin/env python3
"""
Semesterwoche 1: Sekanten und Tangenten.

Geradensteigung, Geradengleichungen, Sekantensteigung (Differenzenquotient),
Tangentensteigung (Differentialquotient) und die Ableitungsfunktion aus der
Definition. Jedes Verfahren liefert ein `Result` mit dem Rechenweg, so wie er
in der Prüfung auf dem Papier stünde.

    from anakit.steigung import slope, secant, tangent, derivative_by_definition
    slope((-1, 2), (2, 4))                 # m = 2/3
    secant("x^2", -1, 3)                   # m = 2
    tangent("1/x", 1)                      # y = 2 - x
    derivative_by_definition("x^3")        # 3x^2 mit binomischem Satz
"""

from __future__ import annotations

import sympy as sp

from .core import fmt, h, num, parse, tex, x
from .render import Result, Step

__all__ = [
    "SlopeResult", "LineResult", "SecantResult", "TangentResult",
    "DerivativeDefResult", "DifferenceQuotientResult", "SecantTableResult",
    "slope", "line_through", "line_point_slope", "line_slope_intercept",
    "secant", "secant_h", "difference_quotient", "secant_table",
    "tangent_slope", "tangent", "derivative_by_definition",
]


# ==========================================================================
# Hilfsfunktionen
# ==========================================================================

def _L(s: str) -> str:
    """Zeichenkette als LaTeX markieren (render.py wickelt sonst in \\text{})."""
    return s if "\\" in s else r"\!" + s


def _p(e) -> str:
    """LaTeX eines Terms, bei Bedarf in Klammern (negative Zahlen, Summen)."""
    e = sp.sympify(e)
    s = tex(e)
    if e.is_Add or (e.is_number and e.could_extract_minus_sign()) or \
            (e.is_Mul and e.could_extract_minus_sign()):
        return r"\left(" + s + r"\right)"
    return s


def _frac(a: str, b: str) -> str:
    return r"\frac{" + a + "}{" + b + "}"


def _fx(a) -> str:
    """f(a) als LaTeX-Text, a in tex()."""
    return "f(" + tex(sp.sympify(a)) + ")"


def _val(f, a):
    """f(a), exakt vereinfacht."""
    return sp.simplify(f.subs(x, a))


def _pt(p):
    return tuple(num(c) for c in p)


def _pt_tex(name: str, p) -> str:
    return f"{name}({tex(p[0])}/{tex(p[1])})"


def _eq(lhs: str, rhs) -> str:
    return _L(lhs + " = " + (rhs if isinstance(rhs, str) else tex(rhs)))


def _num_str(e) -> str:
    return fmt(e)


def _line_expr(m, anchor_x, anchor_y):
    """y = m (x − anchor_x) + anchor_y, ausmultipliziert."""
    return sp.expand(m * (x - anchor_x) + anchor_y)


def _is_bad(v) -> bool:
    return v is None or v.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or v.has(sp.Limit)


def _limit(expr, var, point):
    """
    Grenzwert mit Prüfung beider Seiten.

    Gibt (wert, links, rechts, status) zurück; status 'ok', 'sprung'
    (einseitige Grenzwerte verschieden) oder 'fehler'.
    """
    try:
        lp = sp.limit(expr, var, point, "+")
        lm = sp.limit(expr, var, point, "-")
    except Exception:
        return None, None, None, "fehler"
    if lp.has(sp.Limit) or lm.has(sp.Limit):
        return None, lm, lp, "fehler"
    if sp.simplify(lp - lm) != 0:
        return None, lm, lp, "sprung"
    return sp.simplify(lp), lm, lp, "ok"


def _simplify_quotient(q, var):
    """Kürzen: erst cancel, dann ggf. simplify. Gibt (term, methode) zurück."""
    s = sp.cancel(sp.together(q))
    if s != q:
        return s, "cancel"
    s2 = sp.simplify(q)
    if s2 != q:
        return s2, "simplify"
    return q, ""


def _need_fig(module_attr):
    from ..grafik import plots
    return getattr(plots, module_attr)


# ==========================================================================
# Geradensteigung
# ==========================================================================

class SlopeResult(Result):
    """Steigung der Geraden durch zwei Punkte (Folie 5)."""

    title = "Geradensteigung"

    def __init__(self, p1, p2, dx, dy, m, decimal: bool = False):
        super().__init__(decimal)
        self.p1, self.p2, self.dx, self.dy, self.m = p1, p2, dx, dy, m
        self.vertical = m is None
        if self.vertical:
            self.summary = "Δx = 0: senkrechte Gerade, die Steigung ist nicht definiert."
            self.highlights = [("Δx", dx), ("Gerade", f"x = {fmt(p1[0])}")]
        else:
            self.summary = "m = Δy/Δx = (y₂ − y₁)/(x₂ − x₁)."
            self.highlights = [("m", m)]

    @property
    def value(self):
        return self.m

    def figure(self, xlim=None, dark: bool = False, **kw):
        return line_through(self.p1, self.p2).figure(xlim=xlim, dark=dark, **kw)


def slope(p1, p2) -> SlopeResult:
    """
    Steigung der Geraden durch P₁(x₁/y₁) und P₂(x₂/y₂): m = Δy/Δx.

        slope((-1, 2), (2, 4))   # m = 2/3
    """
    p1, p2 = _pt(p1), _pt(p2)
    (x1_, y1_), (x2_, y2_) = p1, p2
    dx, dy = sp.simplify(x2_ - x1_), sp.simplify(y2_ - y1_)
    steps = [
        Step("Punkte notieren", "Gegeben sind zwei Punkte der Geraden.",
             [(_L("P_1"), _L(_pt_tex("", p1))), (_L("P_2"), _L(_pt_tex("", p2)))]),
        Step("Formel", "Die Steigung ist Höhenunterschied durch Horizontalabstand.",
             [("m", _L(r"\frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1}"))],
             rule="Steigung"),
        Step("Differenzen bilden", "",
             [(_L(r"\Delta y"), _L(f"{tex(y2_)} - {_p(y1_)} = {tex(dy)}")),
              (_L(r"\Delta x"), _L(f"{tex(x2_)} - {_p(x1_)} = {tex(dx)}"))]),
    ]
    if dx == 0:
        steps.append(Step("Senkrechte Gerade",
                          r"$\Delta x = 0$: Durch Null darf man nicht teilen. "
                          "Die Gerade ist senkrecht, sie hat keine Steigung "
                          f"und die Gleichung $x = {tex(x1_)}$.",
                          [("x", x1_)]))
        res = SlopeResult(p1, p2, dx, dy, None)
        res.steps = steps
        res.warnings.append("Δx = 0: senkrechte Gerade, die Steigung ist nicht definiert "
                            "(sie ist kein Funktionsgraph).")
        return res
    m = sp.simplify(dy / dx)
    steps.append(Step("Einsetzen und kürzen", "",
                      [("m", _L(_frac(tex(dy), tex(dx)) + " = " + tex(m)))],
                      note=("Die Gerade ist waagrecht." if m == 0 else "")))
    res = SlopeResult(p1, p2, dx, dy, m)
    res.steps = steps
    return res


# ==========================================================================
# Geradengleichungen
# ==========================================================================

class LineResult(Result):
    """Gleichung einer Geraden y = m·x + t (Folie 7)."""

    title = "Geradengleichung"

    def __init__(self, m, t, points=(), vertical_x=None, decimal: bool = False):
        super().__init__(decimal)
        self.points = tuple(points)
        self.vertical = vertical_x is not None
        self.x_const = vertical_x
        if self.vertical:
            self.m = self.t = self.line = None
            self.summary = f"Senkrechte Gerade x = {fmt(vertical_x)} (keine Funktion von x)."
            self.highlights = [("Gerade", f"x = {fmt(vertical_x)}")]
        else:
            self.m, self.t = m, t
            self.line = sp.expand(m * x + t)
            self.summary = "y(x) = m·x + t mit Steigung m und y-Achsenabschnitt t."
            self.highlights = [("m", m), ("t", t), ("y(x)", self.line)]

    @property
    def eq(self):
        return None if self.vertical else sp.Eq(sp.Symbol("y"), self.line)

    def __call__(self, at):
        """Wert y(at) der Geraden."""
        if self.vertical:
            raise ValueError("Senkrechte Gerade: kein Funktionswert y(x).")
        return self.line.subs(x, num(at))

    def figure(self, xlim=None, dark: bool = False, **kw):
        pts = [(float(a), float(b)) for a, b in self.points]
        if xlim is None:
            xs = [p[0] for p in pts] or [0.0]
            lo, hi = min(xs), max(xs)
            pad = max(2.0, (hi - lo) * 0.6)
            xlim = (lo - pad, hi + pad)
        if self.vertical:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axvline(float(self.x_const), color="tab:blue")
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], color="tab:red", zorder=3)
            ax.set_xlim(*xlim)
            ax.grid(alpha=.3)
            ax.set_title(f"x = {fmt(self.x_const)}")
            return fig
        return _need_fig("plot_function")(self.line, xlim=xlim, points=pts,
                                          title=f"y = {fmt(self.line)}", dark=dark, **kw)


def line_through(p1, p2) -> LineResult:
    """
    Gerade durch zwei Punkte, Zwei-Punkte-Form (Folie 7):

        y(x) = (y₂ − y₁)/(x₂ − x₁) · (x − x₁) + y₁

        line_through((-1, 2), (2, 4))   # y = 2/3·x + 8/3
    """
    p1, p2 = _pt(p1), _pt(p2)
    (x1_, y1_), (x2_, y2_) = p1, p2
    if sp.simplify(x2_ - x1_) == 0:
        res = LineResult(None, None, (p1, p2), vertical_x=x1_)
        res.steps = [
            Step("Punkte notieren", "",
                 [(_L("P_1"), _L(_pt_tex("", p1))), (_L("P_2"), _L(_pt_tex("", p2)))]),
            Step("Senkrechte Gerade",
                 r"$x_1 = x_2$, also $\Delta x = 0$: es gibt keine Steigung. "
                 f"Die Gerade ist senkrecht: $x = {tex(x1_)}$.", [("x", x1_)]),
        ]
        res.warnings.append("x₁ = x₂: senkrechte Gerade, keine Darstellung als y = m·x + t.")
        return res
    sl = slope(p1, p2)
    m = sl.m
    t = sp.simplify(y1_ - m * x1_)
    steps = [
        Step("Punkte notieren", "",
             [(_L("P_1"), _L(_pt_tex("", p1))), (_L("P_2"), _L(_pt_tex("", p2)))]),
        Step("Zwei-Punkte-Form",
             "Für jeden Punkt $P(x/y)$ der Geraden gilt dieselbe Steigung.",
             [(_L("y(x)"), _L(_frac("y_2 - y_1", "x_2 - x_1") + r"\cdot (x - x_1) + y_1"))],
             rule="Zwei-Punkte-Form"),
        Step("Steigung berechnen", "",
             [("m", _L(_frac(f"{tex(y2_)} - {_p(y1_)}", f"{tex(x2_)} - {_p(x1_)}")
                       + " = " + tex(m)))]),
        Step("Einsetzen", "Steigung und Punkt P₁ in die Punkt-Steigungs-Form.",
             [(_L("y(x)"), _L(tex(m) + r"\cdot \left(x - " + _p(x1_) + r"\right) + " + _p(y1_)))]),
        Step("Ausmultiplizieren", r"Auf die Form $y = m\cdot x + t$ bringen: "
             r"$t = y_1 - m\cdot x_1$.",
             [("t", _L(f"{tex(y1_)} - {_p(m)}\\cdot {_p(x1_)} = {tex(t)}")),
              (_L("y(x)"), sp.expand(m * x + t))]),
        Step("Probe", "Beide Punkte müssen auf der Geraden liegen.",
             [(_L("y(" + tex(x1_) + ")"), sp.simplify(m * x1_ + t)),
              (_L("y(" + tex(x2_) + ")"), sp.simplify(m * x2_ + t))]),
    ]
    res = LineResult(m, t, (p1, p2))
    res.steps = steps
    return res


def line_point_slope(p0, m) -> LineResult:
    """
    Gerade durch P₀(x₀/y₀) mit Steigung m (Punkt-Steigungs-Form):

        y(x) = m·(x − x₀) + y₀ = m·x + (y₀ − m·x₀)

        line_point_slope((1, 2), 3)   # y = 3x − 1
    """
    p0 = _pt(p0)
    m = num(m)
    x0_, y0_ = p0
    t = sp.simplify(y0_ - m * x0_)
    res = LineResult(m, t, (p0,))
    res.steps = [
        Step("Gegeben", "", [(_L("P_0"), _L(_pt_tex("", p0))), ("m", m)]),
        Step("Punkt-Steigungs-Form",
             r"Aus $m = \frac{y - y_0}{x - x_0}$ für alle Punkte $P(x/y)$ der Geraden folgt:",
             [(_L("y(x)"), _L(r"m\cdot (x - x_0) + y_0 = m\cdot x + (y_0 - m\cdot x_0)"))],
             rule="Punkt-Steigungs-Form"),
        Step("Einsetzen", "",
             [(_L("y(x)"), _L(tex(m) + r"\cdot \left(x - " + _p(x0_) + r"\right) + " + _p(y0_)))]),
        Step("Ausmultiplizieren", "",
             [("t", _L(f"{tex(y0_)} - {_p(m)}\\cdot {_p(x0_)} = {tex(t)}")),
              (_L("y(x)"), res.line)]),
    ]
    return res


def line_slope_intercept(m, t) -> LineResult:
    """
    Gerade aus Steigung m und y-Achsenabschnitt t: y(x) = m·x + t.
    Nennt zusätzlich Achsenabschnitte (y-Achse, Nullstelle).

        line_slope_intercept(2, -3)
    """
    m, t = num(m), num(t)
    res = LineResult(m, t, ((sp.Integer(0), t),))
    steps = [
        Step("Gegeben", "Steigung und y-Achsenabschnitt.", [("m", m), ("t", t)]),
        Step("Geradengleichung", "", [(_L("y(x)"), res.line)], rule="y = m·x + t"),
        Step("Achsenabschnitte", "y-Achse bei $x = 0$, Nullstelle aus $y = 0$.",
             [(_L("y(0)"), t)] + ([(_L("x_N"), sp.simplify(-t / m))] if m != 0 else []),
             note=("" if m != 0 else "Waagrechte Gerade: keine Nullstelle (ausser t = 0).")),
    ]
    res.steps = steps
    return res


# ==========================================================================
# Sekante (Differenzenquotient)
# ==========================================================================

class SecantResult(Result):
    """Sekantensteigung und -gleichung."""

    title = "Sekante"

    def __init__(self, f, x1_, x2_, f1, f2, m, line, h_=None, decimal: bool = False):
        super().__init__(decimal)
        self.f, self.x1, self.x2, self.y1, self.y2 = f, x1_, x2_, f1, f2
        self.m, self.line, self.h = m, line, h_
        self.variant = 2 if h_ is not None else 1
        self.summary = ("m = (f(x₂) − f(x₁))/(x₂ − x₁): Steigung der Sekante durch "
                        "P₁ und P₂." if h_ is None else
                        "m = (f(x₀ + h) − f(x₀))/h: Differenzenquotient.")
        self.highlights = [("m", m), ("s(x)", line)]

    @property
    def value(self):
        return self.m

    @property
    def p1(self):
        return (self.x1, self.y1)

    @property
    def p2(self):
        return (self.x2, self.y2)

    def figure(self, xlim=None, dark: bool = False, **kw):
        return _need_fig("plot_secant")(self.f, float(self.x1), float(self.x2),
                                        xlim=xlim, dark=dark, **kw)


def _secant_result(f, a, b, variant_h=None):
    """Gemeinsamer Kern für secant() und secant_h()."""
    ya, yb = _val(f, a), _val(f, b)
    dxv = sp.simplify(b - a)
    if dxv == 0:
        raise ValueError("Die beiden Stellen müssen verschieden sein (Δx = 0 ergäbe "
                         "eine Division durch Null).")
    m = sp.simplify((yb - ya) / dxv)
    line = _line_expr(m, a, ya)
    return ya, yb, dxv, m, line


def secant(f, x1, x2) -> SecantResult:
    """
    Sekante durch (x₁/f(x₁)) und (x₂/f(x₂)) (Folie 9):

        m = (f(x₂) − f(x₁)) / (x₂ − x₁)

        secant("x^2", -1, 3)   # m = 2, s(x) = 2x + 3
    """
    f = parse(f)
    a, b = num(x1), num(x2)
    ya, yb, dxv, m, line = _secant_result(f, a, b)
    res = SecantResult(f, a, b, ya, yb, m, line)
    res.steps = [
        Step("Funktion und Stellen", "Gegeben sind die Funktion und zwei feste Stellen.",
             [("f(x)", f), (_L("x_1"), a), (_L("x_2"), b)]),
        Step("Funktionswerte", "Die Funktionswerte liefern die beiden Punkte der Sekante.",
             [(_fx(a), ya), (_fx(b), yb)]),
        Step("Punkte", "",
             [(_L("P_1"), _L(_pt_tex("", (a, ya)))), (_L("P_2"), _L(_pt_tex("", (b, yb))))]),
        Step("Differenzenquotient",
             r"$m = \frac{\Delta y}{\Delta x} = \frac{f(x_2) - f(x_1)}{x_2 - x_1}$",
             [("m", _L(_frac(f"{tex(yb)} - {_p(ya)}", f"{tex(b)} - {_p(a)}")
                       + " = " + _frac(tex(sp.simplify(yb - ya)), tex(dxv))
                       + " = " + tex(m)))],
             rule="Differenzenquotient"),
        Step("Sekantengleichung", "Punkt-Steigungs-Form mit P₁ und der Steigung m.",
             [(_L("s(x)"), _L(tex(m) + r"\cdot \left(x - " + _p(a) + r"\right) + " + _p(ya))),
              (_L("s(x)"), line)],
             rule="Punkt-Steigungs-Form"),
    ]
    return res


def secant_h(f, x0, h) -> SecantResult:  # noqa: A002 (Name wie in der Vorlesung)
    """
    Sekante durch (x₀/f(x₀)) und (x₀+h/f(x₀+h)) (Folie 10):

        m = (f(x₀ + h) − f(x₀)) / h

        secant_h("x^2", 1, "1/2")   # m = 3/2
    """
    f = parse(f)
    a, hv = num(x0), num(h)
    if hv == 0:
        raise ValueError("h = 0 ist nicht erlaubt: der Differenzenquotient braucht h ≠ 0.")
    b = sp.simplify(a + hv)
    ya, yb, dxv, m, line = _secant_result(f, a, b)
    res = SecantResult(f, a, b, ya, yb, m, line, h_=hv)
    res.steps = [
        Step("Funktion, Stelle und Abstand", "",
             [("f(x)", f), (_L("x_0"), a), ("h", hv)]),
        Step("Zweite Stelle", r"$x_0 + h$", [(_L("x_0 + h"), b)]),
        Step("Funktionswerte", "",
             [(_L("f(x_0)"), ya), (_L("f(x_0 + h)"), yb)]),
        Step("Differenzenquotient",
             r"$m = \frac{f(x_0 + h) - f(x_0)}{h}$",
             [("m", _L(_frac(f"{tex(yb)} - {_p(ya)}", tex(hv))
                       + " = " + _frac(tex(sp.simplify(yb - ya)), tex(hv))
                       + " = " + tex(m)))],
             rule="Differenzenquotient"),
        Step("Sekantengleichung", "Punkt-Steigungs-Form durch $P_0$.",
             [(_L("s(x)"), line)], rule="Punkt-Steigungs-Form"),
    ]
    return res


class DifferenceQuotientResult(Result):
    """Differenzenquotient als Term in h."""

    title = "Differenzenquotient"

    def __init__(self, f, x0_, expr, decimal: bool = False):
        super().__init__(decimal)
        self.f, self.x0, self.expr = f, x0_, expr
        self.summary = "m(h) = (f(x₀ + h) − f(x₀))/h als vereinfachter Term in h."
        self.highlights = [("m(h)", expr)]

    def _sympy_(self):
        return self.expr

    def __call__(self, hv):
        return self.expr.subs(h, num(hv))

    @property
    def limit(self):
        """Grenzwert h → 0 (die Tangentensteigung), falls er existiert."""
        return _limit(self.expr, h, 0)[0]


def difference_quotient(f, x0) -> DifferenceQuotientResult:
    """
    Differenzenquotient an der Stelle x₀ als vereinfachter Term in h:

        difference_quotient("x^2", 1).expr   # h + 2
    """
    f = parse(f)
    a = num(x0)
    fa, fah = _val(f, a), f.subs(x, a + h)
    zaehler = sp.expand(fah - fa)
    simp, how = _simplify_quotient(zaehler / h, h)
    steps = [
        Step("Differenzenquotient", r"$m(h) = \frac{f(x_0 + h) - f(x_0)}{h}$",
             [(_L("x_0"), a)], rule="Differenzenquotient"),
        Step("Funktionswerte", "",
             [(_L("f(" + tex(a) + ")"), fa),
              (_L("f(" + tex(a) + " + h)"), sp.expand(fah))]),
        Step("Einsetzen", "",
             [("m(h)", _L(_frac(f"{tex(sp.expand(fah))} - {_p(fa)}", "h")))]),
        Step("Zähler zusammenfassen", "", [("Zähler", zaehler)]),
        Step("Kürzen", r"Mit $h \neq 0$ durch $h$ kürzen.", [("m(h)", simp)]),
    ]
    res = DifferenceQuotientResult(f, a, simp)
    res.steps = steps
    return res


class SecantTableResult(Result):
    """Sekantensteigungen für kleiner werdende h."""

    title = "Sekantensteigungen für kleiner werdendes h"

    def __init__(self, f, x0_, rows, limit, decimal: bool = False):
        super().__init__(decimal)
        self.f, self.x0, self.rows, self.limit = f, x0_, rows, limit
        self.summary = ("Die Sekantensteigungen nähern sich der Tangentensteigung "
                        "f'(x₀) = " + (fmt(limit) if limit is not None else "?") + " an.")
        self.highlights = [("f'(x_0)", limit)] if limit is not None else []

    def table(self):
        """Liste von (h, m) als exakte Werte."""
        return list(self.rows)

    def figure(self, xlim=None, dark: bool = False, **kw):
        return _need_fig("plot_tangent")(self.f, float(self.x0), xlim=xlim,
                                         secants_h=tuple(float(r[0]) for r in self.rows),
                                         dark=dark, **kw)


def secant_table(f, x0, hs=(1, sp.Rational(1, 2), sp.Rational(1, 10),
                            sp.Rational(1, 100), sp.Rational(1, 1000))) -> SecantTableResult:
    """
    Tabelle der Sekantensteigungen m(h) für kleiner werdendes h.

        secant_table("x^2", 1).rows   # (1, 3), (1/2, 5/2), ... → Tangentensteigung 2
    """
    f = parse(f)
    a = num(x0)
    hs = [num(v) for v in hs]
    steps = [Step("Idee", "Rückt die zweite Stelle $x_0 + h$ immer näher an $x_0$, "
                  "dreht sich die Sekante in die Tangente.",
                  [(_L("x_0"), a), ("f(x)", f)])]
    rows = []
    fa = _val(f, a)
    for hv in hs:
        if hv == 0:
            raise ValueError("h = 0 ist nicht erlaubt.")
        fb = _val(f, a + hv)
        m = sp.simplify((fb - fa) / hv)
        rows.append((hv, m))
        steps.append(Step(f"h = {fmt(hv)}",
                          r"$m = \frac{f(x_0 + h) - f(x_0)}{h}$",
                          [(_L(f"f({tex(a + hv)})"), fb),
                           ("m", _L(_frac(f"{tex(fb)} - {_p(fa)}", tex(hv)) + " = " + tex(m))),
                           (_L(r"m \approx"), sp.Float(float(m), 6))]))
    lim = tangent_slope(f, a, "h").value
    steps.append(Step("Tendenz", "Grenzübergang $h \\to 0$ liefert die Tangentensteigung.",
                      [(_L("f'(" + tex(a) + ")"), lim if lim is not None else "existiert nicht")],
                      rule="Differentialquotient"))
    res = SecantTableResult(f, a, rows, lim)
    res.steps = steps
    return res


# ==========================================================================
# Tangente (Differentialquotient)
# ==========================================================================

class TangentResult(Result):
    """Tangentensteigung f'(x₀) und – wenn verlangt – die Tangentengleichung."""

    title = "Tangentensteigung"

    def __init__(self, f, x0_, variant, value, fx0, line=None, decimal: bool = False):
        super().__init__(decimal)
        self.f, self.x0, self.variant = f, x0_, variant
        self.value, self.fx0, self.line = value, fx0, line
        self.differentiable = value is not None
        if line is not None:
            self.title = "Tangente"
        if value is None:
            self.summary = f"f ist an der Stelle x₀ = {fmt(x0_)} nicht differenzierbar."
            self.highlights = []
        else:
            self.summary = ("f'(x₀) = lim (f(x) − f(x₀))/(x − x₀) für x → x₀." if variant == "x"
                            else "f'(x₀) = lim (f(x₀ + h) − f(x₀))/h für h → 0.")
            self.highlights = [("f'(x_0)", value)]
            if line is not None:
                self.highlights.append(("t(x)", line))

    @property
    def m(self):
        return self.value

    @property
    def point(self):
        return (self.x0, self.fx0)

    def figure(self, xlim=None, secants_h=(), dark: bool = False, **kw):
        return _need_fig("plot_tangent")(self.f, float(self.x0), xlim=xlim,
                                         secants_h=secants_h, dark=dark, **kw)


def _limit_steps(f, a, variant):
    """Rechenweg für f'(x₀). Gibt (steps, wert, fa, warnings) zurück."""
    if variant not in ("x", "h"):
        raise ValueError("variant muss 'x' oder 'h' sein.")
    fa = _val(f, a)
    steps, warns = [], []
    A = tex(a)
    if variant == "x":
        var, point = x, a
        steps.append(Step("Definition (x → x₀)",
                          "Die Tangentensteigung ist der Grenzwert der Sekantensteigungen:",
                          [(_L("f'(x_0)"), _L(r"\lim_{x \to x_0} \frac{f(x) - f(x_0)}{x - x_0}"))],
                          rule="Differentialquotient"))
        steps.append(Step("Funktionswert", f"An der Stelle $x_0 = {A}$:",
                          [(_L(f"f({A})"), fa)]))
        q = (f - fa) / (x - a)
        steps.append(Step("Einsetzen", "",
                          [(_L("f'(" + A + ")"),
                            _L(r"\lim_{x \to " + A + "}" + _frac(f"{tex(f)} - {_p(fa)}",
                                                                 f"x - {_p(a)}"))
                            )]))
        zaehler = f - fa
        zf = sp.factor(sp.together(zaehler))
        if zf != zaehler:
            steps.append(Step("Zähler faktorisieren",
                              r"Der Faktor $(x - x_0)$ muss im Zähler auftauchen.",
                              [("Zähler", zf)]))
        simp, how = _simplify_quotient(q, x)
    else:
        var, point = h, sp.Integer(0)
        fah = f.subs(x, a + h)
        steps.append(Step("Definition (h → 0)",
                          "Die Tangentensteigung ist der Grenzwert des Differenzenquotienten:",
                          [(_L("f'(x_0)"), _L(r"\lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}"))],
                          rule="Differentialquotient"))
        steps.append(Step("Funktionswerte", f"An der Stelle $x_0 = {A}$:",
                          [(_L(f"f({A})"), fa), (_L(f"f({A} + h)"), fah)]))
        steps.append(Step("Einsetzen", "",
                          [(_L("f'(" + A + ")"),
                            _L(r"\lim_{h \to 0}" + _frac(f"{tex(fah)} - {_p(fa)}", "h")))]))
        zaehler = sp.expand(fah - fa)
        if zaehler != fah - fa:
            steps.append(Step("Zähler ausmultiplizieren", "",
                              [("Zähler", zaehler)]))
        simp, how = _simplify_quotient(zaehler / h, h)
        q = (fah - fa) / h
    steps.append(Step("Kürzen und vereinfachen",
                      "Mit $" + ("x" if variant == "x" else "h") + r" \neq " +
                      ("x_0" if variant == "x" else "0") +
                      "$ darf gekürzt werden (der Grenzwert fragt nur nach der Umgebung).",
                      [(_L("Differenzenquotient"), simp)],
                      rule={"cancel": "Kürzen", "simplify": "Vereinfachen"}.get(how, "")))
    # Grenzwert
    try:
        direct = simp.subs(var, point)
    except Exception:
        direct = None
    if direct is not None and not _is_bad(sp.sympify(direct)) and \
            not simp.has(sp.Abs, sp.sign, sp.Piecewise, sp.Heaviside):
        value = sp.simplify(direct)
        steps.append(Step("Grenzwert bilden",
                          "Jetzt darf direkt eingesetzt werden (kein Ausdruck 0/0 mehr).",
                          [(_L("f'(" + A + ")"),
                            _L(r"\lim_{" + ("x" if variant == "x" else "h") + r" \to " +
                               tex(point) + "}" + _p(simp) + " = " + tex(value)))],
                          rule="Grenzwert"))
    else:
        value, lm, lp, status = _limit(simp, var, point)
        if status == "ok":
            steps.append(Step("Grenzwert bilden",
                              "Einsetzen ergibt einen unbestimmten Ausdruck; der Grenzwert "
                              "wird mit SymPy bestimmt (links = rechts).",
                              [(_L("f'(" + A + ")"), value)], rule="Grenzwert"))
        elif status == "sprung":
            steps.append(Step("Einseitige Grenzwerte",
                              "Links- und rechtsseitiger Grenzwert sind verschieden.",
                              [(_L(r"\lim_{\to 0^-}"), lm), (_L(r"\lim_{\to 0^+}"), lp)]))
            warns.append(f"Linksseitiger ({fmt(lm)}) und rechtsseitiger Grenzwert ({fmt(lp)}) "
                         "sind verschieden: f ist an dieser Stelle nicht differenzierbar "
                         "(Knick oder Sprung).")
            value = None
        else:
            warns.append("SymPy konnte den Grenzwert nicht bestimmen.")
            value = None
    return steps, value, fa, warns


def tangent_slope(f, x0, variant: str = "x") -> TangentResult:
    """
    Tangentensteigung f'(x₀) über den Grenzwert (Folien 12/13).

    variant 'x':  lim_{x→x₀} (f(x) − f(x₀))/(x − x₀)
    variant 'h':  lim_{h→0}  (f(x₀+h) − f(x₀))/h

        tangent_slope("x^2", 2).value        # 4
        tangent_slope("x^2", 2, "h").value   # 4
    """
    f = parse(f)
    a = num(x0)
    steps, value, fa, warns = _limit_steps(f, a, variant)
    res = TangentResult(f, a, variant, value, fa)
    res.steps = steps
    res.warnings.extend(warns)
    return res


def tangent(f, x0, variant: str = "x") -> TangentResult:
    """
    Tangente an der Stelle x₀: y = f'(x₀)·(x − x₀) + f(x₀).

        tangent("1/x", 1).line   # 2 − x
    """
    f = parse(f)
    a = num(x0)
    steps, value, fa, warns = _limit_steps(f, a, variant)
    if value is None:
        res = TangentResult(f, a, variant, None, fa)
        res.steps = steps
        res.warnings.extend(warns)
        res.warnings.append("Keine Tangente: f'(x₀) existiert nicht.")
        return res
    line = _line_expr(value, a, fa)
    A = tex(a)
    steps += [
        Step("Berührpunkt und Steigung", "Die Tangente berührt den Graphen in $P_0$ "
             "und hat dort die Steigung $f'(x_0)$.",
             [(_L("P_0"), _L(_pt_tex("", (a, fa)))), (_L("f'(" + A + ")"), value)]),
        Step("Tangentengleichung", "Punkt-Steigungs-Form:",
             [(_L("t(x)"), _L(r"f'(x_0)\cdot (x - x_0) + f(x_0)")),
              (_L("t(x)"), _L(tex(value) + r"\cdot \left(x - " + _p(a) + r"\right) + " + _p(fa))),
              (_L("t(x)"), line)],
             rule="Tangente"),
    ]
    res = TangentResult(f, a, variant, value, fa, line=line)
    res.steps = steps
    res.warnings.extend(warns)
    return res


# ==========================================================================
# Ableitungsfunktion aus der Definition
# ==========================================================================

class DerivativeDefResult(Result):
    """Ableitungsfunktion f'(x), hergeleitet aus dem Differentialquotienten."""

    title = "Ableitungsfunktion aus der Definition"

    def __init__(self, f, expr, method: str = "allgemein", decimal: bool = False):
        super().__init__(decimal)
        self.f, self.expr, self.method = f, expr, method
        self.summary = ("f'(x) = lim_{h→0} (f(x + h) − f(x))/h — die Tangentensteigung "
                        "an einer beliebigen Stelle x.")
        self.highlights = [("f'(x)", expr)]

    @property
    def value(self):
        return self.expr

    @property
    def derivative(self):
        return self.expr

    def __call__(self, at):
        return self.expr.subs(x, num(at))

    def figure(self, xlim=None, dark: bool = False, **kw):
        return _need_fig("plot_derivative")(self.f, xlim=xlim, dark=dark, **kw)


_DEF = Step("Definition", "Die Ableitungsfunktion ist der Grenzwert des Differenzenquotienten "
            "an einer beliebigen Stelle $x$:",
            [("f'(x)", _L(r"\lim_{h \to 0} \frac{f(x + h) - f(x)}{h}"))],
            rule="Differentialquotient")


def _binom_terms(n: int, maxshow: int = 8):
    """LaTeX-Entwicklung von (x+h)^n (binomischer Satz) und die Einzelterme."""
    terms = [sp.binomial(n, k) * x ** (n - k) * h ** k for k in range(n + 1)]
    if n + 1 <= maxshow:
        s = " + ".join(tex(t) for t in terms)
    else:
        s = " + ".join(tex(t) for t in terms[:3]) + r" + \ldots + " + tex(terms[-1])
    return s, terms


def _power_lehrweg(f, n: int) -> DerivativeDefResult:
    expr = sp.Integer(n) * x ** (n - 1)
    full, terms = _binom_terms(n)
    rest_terms = terms[1:]
    rest = (" + ".join(tex(t / h) for t in rest_terms) if len(rest_terms) <= 7
            else " + ".join(tex(t / h) for t in rest_terms[:2]) + r" + \ldots + " +
            tex(rest_terms[-1] / h))
    steps = [
        _DEF,
        Step("Differenzenquotient aufstellen", f"Mit $f(x) = x^{{{n}}}$:",
             [("f'(x)", _L(r"\lim_{h \to 0}" + _frac(f"(x + h)^{{{n}}} - x^{{{n}}}", "h")))]),
        Step("Binomischer Satz",
             "Allgemein: (a + b)ⁿ = Σ (n über k) · aⁿ⁻ᵏ · bᵏ für k = 0 … n. "
             f"Hier mit $a = x$, $b = h$ und $n = {n}$:",
             [(_L(f"(x + h)^{{{n}}}"), _L(full))], rule="Binomischer Satz"),
        Step("Einsetzen und x^n wegheben",
             f"Der Term $x^{{{n}}}$ hebt sich auf; im Zähler bleibt ein Faktor $h$ in jedem Summanden.",
             [("Zähler", _L(" + ".join(tex(t) for t in rest_terms) if len(rest_terms) <= 7
                                       else " + ".join(tex(t) for t in rest_terms[:2]) +
                                       r" + \ldots + " + tex(rest_terms[-1])))]),
        Step("Durch h kürzen", r"Mit $h \neq 0$ ausklammern und kürzen:",
             [("m(h)", _L(rest))], rule="Kürzen"),
        Step("Grenzwert h → 0",
             "Alle Summanden mit dem Faktor $h$ gehen gegen $0$, es bleibt nur der erste.",
             [("f'(x)", _L(r"\lim_{h \to 0}\left(" + rest + r"\right) = " + tex(expr)))],
             rule="Grenzwert"),
        Step("Vermutung bestätigt",
             f"Für $f(x) = x^{{{n}}}$ gilt $f'(x) = n\\cdot x^{{n-1}}$ mit $n = {n}$.",
             [(_L(f"(x^{{{n}}})'"), expr)]),
    ]
    res = DerivativeDefResult(f, expr, "binomisch")
    res.steps = steps
    return res


def _power_symbolic(f, n) -> DerivativeDefResult:
    expr = n * x ** (n - 1)
    res = DerivativeDefResult(f, expr, "potenz")
    res.steps = [
        _DEF,
        Step("Differenzenquotient aufstellen", "",
             [("f'(x)", _L(r"\lim_{h \to 0}" + _frac(f"(x + h)^{{{tex(n)}}} - x^{{{tex(n)}}}", "h")))]),
        Step("Binomischer Satz (für natürliches n)",
             "Nur die ersten beiden Summanden werden gebraucht, alle weiteren enthalten $h^2$:",
             [(_L(f"(x + h)^{{{tex(n)}}}"),
               _L(f"x^{{{tex(n)}}} + {tex(n)}\\, x^{{{tex(n)}-1}}\\, h + "
                  f"\\frac{{{tex(n)}({tex(n)}-1)}}{{2}} x^{{{tex(n)}-2}} h^2 + \\ldots + h^{{{tex(n)}}}"))],
             rule="Binomischer Satz"),
        Step("Kürzen", r"$x^n$ hebt sich weg, durch $h$ kürzen:",
             [("m(h)", _L(f"{tex(n)}\\, x^{{{tex(n)}-1}} + \\frac{{{tex(n)}({tex(n)}-1)}}{{2}} x^{{{tex(n)}-2}} h"
                          r" + \ldots + h^{" + tex(n) + "-1}"))], rule="Kürzen"),
        Step("Grenzwert h → 0", "Alle Terme mit $h$ verschwinden:",
             [("f'(x)", expr)], rule="Grenzwert"),
        Step("Vermutung", r"$(x^n)' = n\cdot x^{n-1}$ (hier hergeleitet für $n \in \mathbb{N}$).", []),
    ]
    return res


def _reciprocal_lehrweg(f) -> DerivativeDefResult:
    expr = -1 / x ** 2
    steps = [
        _DEF,
        Step("Differenzenquotient aufstellen", r"Mit $r(x) = \frac{1}{x}$:",
             [("m(h)", _L(_frac(r"\frac{1}{x + h} - \frac{1}{x}", "h")))]),
        Step("Gleichnamig machen", "Hauptnenner $x\\,(x + h)$ im Zähler:",
             [("m(h)", _L(_frac(_frac("x - (x + h)", r"x\,(x + h)"), "h")))],
             rule="Bruchrechnen"),
        Step("Zähler vereinfachen", "",
             [("m(h)", _L(_frac(_frac("-h", r"x\,(x + h)"), "h")))]),
        Step("Durch h kürzen", r"Mit $h \neq 0$:",
             [("m(h)", _L(_frac("-1", r"x\,(x + h)")))], rule="Kürzen"),
        Step("Grenzwert h → 0", "Jetzt darf $h = 0$ eingesetzt werden:",
             [("f'(x)", _L(r"\lim_{h \to 0}" + _frac("-1", r"x\,(x + h)") + " = " +
                           _frac("-1", "x^2")))], rule="Grenzwert"),
    ]
    res = DerivativeDefResult(f, expr, "kehrwert")
    res.steps = steps
    return res


def _trig_lehrweg(f, kind: str) -> DerivativeDefResult:
    sinus = kind == "sin"
    if sinus:
        theorem = r"\sin(x + h) = \sin x\,\cos h + \cos x\,\sin h"
        quot = _frac(r"\sin x\,\cos h + \cos x\,\sin h - \sin x", "h")
        split = (r"-\sin x \cdot \frac{1 - \cos h}{h} + \cos x \cdot \frac{\sin h}{h}")
        final = (r"-\sin x \cdot 0 + \cos x \cdot 1 = \cos x")
        expr = sp.cos(x)
        pre = r"\sin x\,(\cos h - 1) + \cos x\,\sin h"
    else:
        theorem = r"\cos(x + h) = \cos x\,\cos h - \sin x\,\sin h"
        quot = _frac(r"\cos x\,\cos h - \sin x\,\sin h - \cos x", "h")
        split = (r"-\cos x \cdot \frac{1 - \cos h}{h} - \sin x \cdot \frac{\sin h}{h}")
        final = (r"-\cos x \cdot 0 - \sin x \cdot 1 = -\sin x")
        expr = -sp.sin(x)
        pre = r"\cos x\,(\cos h - 1) - \sin x\,\sin h"
    name = r"\sin" if sinus else r"\cos"
    steps = [
        _DEF,
        Step("Additionstheorem", "Die Funktion an der Stelle $x + h$ aufspalten:",
             [(_L("Additionstheorem"), _L(theorem))], rule="Additionstheorem"),
        Step("Differenzenquotient aufstellen", "",
             [("m(h)", _L(quot))]),
        Step("Ausklammern", f"$\\{name[1:]}\\,x$ ausklammern (Zähler umordnen):",
             [("m(h)", _L(_frac(pre, "h")))]),
        Step("In bekannte Grenzwerte aufspalten", "Zwei Brüche, jeweils nur mit $h$ im Nenner:",
             [("m(h)", _L(split))]),
        Step("Bekannte Grenzwerte",
             r"Aus dem Einschnürungssatz folgt $\cos h < \frac{\sin h}{h} < 1$ für kleine $h > 0$, "
             r"also $\lim_{h\to 0}\frac{\sin h}{h} = 1$. Mit "
             r"$\frac{1 - \cos h}{h} = \frac{\sin h}{h}\cdot\frac{\sin h}{1 + \cos h}$ "
             r"folgt $\lim_{h\to 0}\frac{1 - \cos h}{h} = 1\cdot\frac{0}{2} = 0$.",
             [(_L(r"\lim_{h \to 0} \frac{\sin h}{h}"), sp.Integer(1)),
              (_L(r"\lim_{h \to 0} \frac{1 - \cos h}{h}"), sp.Integer(0))],
             rule="Einschnürungssatz"),
        Step("Grenzwert h → 0", "", [("f'(x)", _L(final))], rule="Grenzwert"),
    ]
    res = DerivativeDefResult(f, expr, "trigonometrisch")
    res.steps = steps
    return res


def derivative_by_definition(f) -> DerivativeDefResult:
    """
    Ableitungsfunktion f'(x) = lim_{h→0} (f(x+h) − f(x))/h aus der Definition.

    Mit ausführlichem Lehrweg für xⁿ (binomischer Satz, n ∈ ℕ≥2, Folie 15),
    1/x, sin x und cos x. Für jede andere Funktion: Differenzenquotient
    aufstellen, vereinfachen, Grenzwert h → 0 (SymPy).

        derivative_by_definition("x^3").expr    # 3x^2
        derivative_by_definition("sin(x)").expr # cos(x)
    """
    f = parse(f)
    # --- Lehrwege ---
    if f.is_Pow and f.base == x:
        if f.exp.is_Integer and f.exp >= 2:
            return _power_lehrweg(f, int(f.exp))
        if f.exp == -1:
            return _reciprocal_lehrweg(f)
        if f.exp.is_Symbol:
            return _power_symbolic(f, f.exp)
    if f == sp.sin(x):
        return _trig_lehrweg(f, "sin")
    if f == sp.cos(x):
        return _trig_lehrweg(f, "cos")

    # --- allgemeiner Weg ---
    fx_h = f.subs(x, x + h)
    steps = [
        _DEF,
        Step("Differenzenquotient aufstellen", "",
             [("f(x)", f), ("f(x + h)", fx_h),
              ("m(h)", _L(_frac(f"{tex(fx_h)} - {_p(f)}", "h")))]),
    ]
    zaehler = sp.expand(fx_h - f)
    if zaehler != fx_h - f:
        steps.append(Step("Zähler ausmultiplizieren", "", [("Zähler", zaehler)],
                          rule="Ausmultiplizieren"))
    simp, how = _simplify_quotient(zaehler / h, h)
    steps.append(Step("Kürzen und vereinfachen", r"Mit $h \neq 0$ durch $h$ kürzen:",
                      [("m(h)", simp)],
                      rule={"cancel": "Kürzen", "simplify": "Vereinfachen"}.get(how, "")))
    warns = []
    ref = sp.diff(f, x)
    value, lm, lp, status = (None, None, None, "fehler")
    try:
        direct = simp.subs(h, 0)
    except Exception:
        direct = None
    if direct is not None and not _is_bad(sp.sympify(direct)) and not simp.has(sp.Piecewise):
        value, status = sp.simplify(direct), "ok"
        steps.append(Step("Grenzwert h → 0", "Jetzt darf $h = 0$ eingesetzt werden:",
                          [("f'(x)", value)], rule="Grenzwert"))
    else:
        value, lm, lp, status = _limit(simp, h, 0)
        if status == "ok":
            steps.append(Step("Grenzwert h → 0",
                              "Einsetzen von $h = 0$ gibt $0/0$; der Grenzwert wird mit SymPy bestimmt.",
                              [("f'(x)", value)], rule="Grenzwert"))
    if status != "ok":
        warns.append("SymPy konnte den Grenzwert nicht bestimmen; das Ergebnis stammt aus "
                     "sympy.diff (Ableitungsregeln), nicht aus der Definition.")
        value = ref
        steps.append(Step("Ergebnis über Ableitungsregeln", "", [("f'(x)", ref)]))
    else:
        try:
            ok = sp.simplify(value - ref) == 0
        except Exception:
            ok = True
        if not ok:
            warns.append("Der Grenzwert weicht von sympy.diff ab; verwendet wird sympy.diff.")
            value = ref
            steps.append(Step("Kontrolle mit sympy.diff", "", [("f'(x)", ref)]))
    res = DerivativeDefResult(f, value, "allgemein")
    res.steps = steps
    res.warnings.extend(warns)
    return res
