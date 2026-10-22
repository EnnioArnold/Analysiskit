#!/usr/bin/env python3
"""
Semesterwoche 6: Kurvendiskussion, Ergänzungen zu SW4.

SW4 (`analysis.py`) behandelt Monotonie, kritische Punkte, Krümmung, Extrema
und Wendepunkte. Die Lernziele von SW6 nennen zusätzlich Definitionsbereich,
Wertebereich, Symmetrie, Nullstellen, Pole und Asymptoten sowie die
Sonderfälle mit höheren Ableitungen:

    from anakit.kurve import curve_basics, higher_order_test

    curve_basics("x^2/(x-1)")      # Pol bei 1, schräge Asymptote y = x + 1
    higher_order_test("x^4", 0)    # erste nichtverschwindende Ableitung: k = 4 => Minimum

Kriterium Folie 5: Ist f⁽ᵏ⁾(x₀) die erste nichtverschwindende Ableitung an
der Stelle x₀ (ab f'), so hat f dort

    ein Minimum,     falls k gerade und f⁽ᵏ⁾(x₀) > 0,
    ein Maximum,     falls k gerade und f⁽ᵏ⁾(x₀) < 0,
    einen Wendepunkt, falls k ungerade (k ≥ 3: Sattelpunkt, waagerechte Tangente).
"""

from __future__ import annotations

import sympy as sp

from .analysis import _find_roots
from .core import fmt, parse, x
from .render import Result, Step

__all__ = ["curve_basics", "higher_order_test", "CurveBasicsResult", "HigherOrderResult"]

_REAL = sp.S.Reals


def _L(s: str) -> str:
    return s if "\\" in s else r"\!" + s


def _set_text(s) -> str:
    """Eine SymPy-Menge lesbar: (−∞, 1) ∪ (1, ∞), ℝ, ∅."""
    if s == _REAL:
        return "ℝ"
    if s == sp.S.EmptySet:
        return "∅"
    if isinstance(s, sp.Union):
        return " ∪ ".join(_set_text(a) for a in sorted(s.args, key=lambda a: float(a.inf)))
    if isinstance(s, sp.Interval):
        lo, hi = fmt(s.inf), fmt(s.sup)
        return f"{'(' if s.left_open else '['}{lo}, {hi}{')' if s.right_open else ']'}"
    if isinstance(s, sp.FiniteSet):
        return "{" + ", ".join(fmt(a) for a in s) + "}"
    return str(s)


# ==========================================================================
# Definitions- und Wertebereich, Symmetrie, Nullstellen, Pole, Asymptoten
# ==========================================================================

class CurveBasicsResult(Result):
    """Grundangaben zu f ohne Ableitungen."""

    title = "Funktionsüberblick"

    def __init__(self, f):
        super().__init__()
        self.f = f
        self.domain = None
        self.range = None
        self.symmetry = "keine"
        self.zeros: list = []
        self.poles: list = []          # (Stelle, 'Pol mit VZW' | 'Pol ohne VZW' | 'Lücke')
        self.asymptotes: list = []     # Texte; Geraden als Terme in self.lines
        self.lines: list = []

    def figure(self, xlim=None, dark: bool = False, **kw):
        from ..grafik.plots import plot_function
        pts = [z for z in self.zeros if z.is_real and z.is_number][:8]
        return plot_function(self.f, xlim or self._xlim(), points=[float(z) for z in pts],
                             title=f"f(x) = {fmt(self.f)}", dark=dark, **kw)

    def _xlim(self):
        xs = [float(v) for v in self.zeros] + [float(p) for p, _ in self.poles]
        if not xs:
            return (-5, 5)
        lo, hi = min(xs), max(xs)
        pad = max(2.0, 0.5 * (hi - lo))
        return (lo - pad, hi + pad)


def _limit(expr, point, direction):
    try:
        v = sp.limit(expr, x, point, direction)
    except Exception:
        return None
    return v


def curve_basics(f) -> CurveBasicsResult:
    """
    Definitionsbereich, Wertebereich, Symmetrie, Nullstellen, Pole und
    Asymptoten (waagrecht, schräg, senkrecht) — das «Kopfstück» einer
    Kurvendiskussion.

        curve_basics("x^2/(x-1)").lines      # [x + 1]
    """
    f = parse(f)
    res = CurveBasicsResult(f)
    S = res.steps

    # 1. Definitionsbereich
    try:
        dom = sp.calculus.util.continuous_domain(f, x, _REAL)
    except Exception:
        dom = None
    res.domain = dom
    S.append(Step("Definitionsbereich", "Nenner ≠ 0, Argument von ln und Wurzel im erlaubten Bereich.",
                  [("D", _set_text(dom) if dom is not None else "TODO: nicht bestimmt")],
                  rule="Definitionsbereich"))

    # 2. Wertebereich (nur rationale Funktionen, sonst nicht automatisch)
    if f.is_rational_function(x) and dom is not None:
        try:
            res.range = sp.calculus.util.function_range(f, x, dom)
        except Exception:
            res.range = None
    S.append(Step("Wertebereich", "", [("W", _set_text(res.range) if res.range is not None
                                        else "nicht automatisch bestimmt")],
                  note="" if res.range is not None else
                  "Für Nicht-Polynome bzw. Nicht-Bruchterme bestimmt anakit den Wertebereich nicht; "
                  "aus dem Graph und den Extrema ablesen."))

    # 3. Symmetrie
    try:
        even = sp.simplify(f.subs(x, -x) - f) == 0
        odd = sp.simplify(f.subs(x, -x) + f) == 0
    except Exception:
        even = odd = False
    res.symmetry = "gerade" if even and not odd else "ungerade" if odd and not even else \
        "gerade und ungerade (f = 0)" if even and odd else "keine"
    S.append(Step("Symmetrie", "Gerade: $f(-x) = f(x)$ (achsensymmetrisch zur y-Achse); "
                  "ungerade: $f(-x) = -f(x)$ (punktsymmetrisch zum Ursprung).",
                  [(_L("f(-x)"), f.subs(x, -x)),
                   ("Ergebnis", {"gerade": "f(−x) = f(x): gerade, achsensymmetrisch",
                                 "ungerade": "f(−x) = −f(x): ungerade, punktsymmetrisch zum Ursprung",
                                 "keine": "weder gerade noch ungerade"}.get(res.symmetry, res.symmetry))],
                  rule="Symmetrie"))

    # 4. Nullstellen
    zeros, exact, info = _find_roots(f, -sp.oo, sp.oo)
    if dom is not None:
        zeros = [z for z in zeros if dom.contains(z) == sp.true or not z.is_number]
    res.zeros = zeros
    S.append(Step("Nullstellen", "$f(x) = 0$ lösen (Zähler null, Nenner nicht null).",
                  [(_L(f"x_{{{i + 1}}}"), z) for i, z in enumerate(zeros)] or [("", "keine")],
                  note=info, rule="Nullstellen"))
    res.warnings += [info] if info else []

    # 5. Pole und Lücken
    try:
        sing = sp.singularities(f, x)
    except Exception:
        sing = sp.S.EmptySet
    pts = sorted([p for p in sing if p.is_real is not False and p.is_number], key=float) \
        if isinstance(sing, sp.FiniteSet) else []
    rows = []
    for p in pts:
        left, right = _limit(f, p, "-"), _limit(f, p, "+")
        inf = (sp.oo, -sp.oo)
        if left in inf or right in inf:
            kind = "Pol mit Vorzeichenwechsel" if (left in inf and right in inf and left != right) \
                else "Pol ohne Vorzeichenwechsel" if (left == right) else "Pol (einseitig)"
            res.asymptotes.append(f"senkrechte Asymptote x = {fmt(p)}")
        else:
            kind = "Lücke (hebbar)" if left == right and left is not None else "Singularität"
        res.poles.append((p, kind))
        rows.append((_L(f"x = {fmt(p)}"), f"{kind}: links {fmt(left)}, rechts {fmt(right)}"))
    S.append(Step("Pole", "Stellen, an denen f nicht definiert ist, und das Verhalten daneben.",
                  rows or [("", "keine")], rule="Pole und Asymptoten"))

    # 6. Asymptoten für x → ±∞
    lines = []
    for sign, name in ((sp.oo, "+∞"), (-sp.oo, "−∞")):
        m = _limit(f / x, sign, "-" if sign == sp.oo else "+")
        if m is None or m in (sp.oo, -sp.oo) or m.has(sp.nan, sp.zoo, sp.AccumBounds):
            continue
        n = _limit(f - m * x, sign, "-" if sign == sp.oo else "+")
        if n is None or n in (sp.oo, -sp.oo) or n.has(sp.nan, sp.zoo, sp.AccumBounds):
            continue
        line = sp.simplify(m * x + n)
        kind = "waagrechte" if m == 0 else "schräge"
        lines.append((name, kind, line))
    for name, kind, line in lines:
        res.asymptotes.append(f"{kind} Asymptote y = {fmt(line)} für x → {name}")
    unique = []
    for _n, _k, line in lines:
        if line not in unique:
            unique.append(line)
    res.lines = unique
    S.append(Step("Verhalten im Unendlichen",
                  "$m = \\lim f(x)/x$, $n = \\lim (f(x) - m\\,x)$; $m = 0$: waagrechte, $m \\neq 0$: schräge Asymptote "
                  "$y = m\\,x + n$.",
                  [(f"x → {name}", f"{kind} Asymptote y = {fmt(line)}") for name, kind, line in lines]
                  or [("", "keine Asymptote")], rule="Pole und Asymptoten"))

    res.summary = (f"D = {_set_text(dom) if dom is not None else '?'}; Symmetrie: {res.symmetry}; "
                   f"{len(zeros)} Nullstelle(n); {len(res.poles)} Pol-/Lückenstelle(n).")
    res.highlights = [("Definitionsbereich", _set_text(dom) if dom is not None else "?"),
                      ("Symmetrie", res.symmetry),
                      ("Nullstellen", zeros if zeros else "keine"),
                      ("Pole", [p for p, k in res.poles if k.startswith("Pol")] or "keine"),
                      ("Asymptoten", "; ".join(res.asymptotes) or "keine")]
    return res


# ==========================================================================
# Sonderfälle: höhere Ableitungen
# ==========================================================================

class HigherOrderResult(Result):
    """Erste nichtverschwindende Ableitung an x₀ und der Schluss daraus (Folie 5)."""

    title = "Höhere Ableitungen: Sonderfälle"

    def __init__(self, f, x0, derivs):
        super().__init__()
        self.f, self.x0 = f, x0
        #: Werte f'(x0), f''(x0), … bis zur ersten nichtverschwindenden
        self.values = derivs
        self.k = len(derivs) if derivs and derivs[-1] != 0 else None
        self.value_k = derivs[-1] if self.k else None
        self.stationary = bool(derivs) and derivs[0] == 0
        self.kind = None
        self.inflection = False

    def figure(self, xlim=None, dark: bool = False, **kw):
        from ..grafik.plots import plot_function
        x0 = float(self.x0)
        return plot_function(self.f, xlim or (x0 - 3, x0 + 3), points=[x0], dark=dark, **kw)


def higher_order_test(f, x0, kmax: int = 12) -> HigherOrderResult:
    """
    Sonderfälle der Extremwert- und Wendepunktsuche (Folie 5).

    Ist f'(x0) = 0, wird so lange weiter abgeleitet, bis f⁽ᵏ⁾(x0) ≠ 0.
    k gerade: Minimum (f⁽ᵏ⁾ > 0) bzw. Maximum (f⁽ᵏ⁾ < 0); k ungerade (k ≥ 3):
    Wendepunkt mit waagerechter Tangente (Sattelpunkt). Zusätzlich gilt ab
    f'': erste nichtverschwindende Ableitung ungerader Ordnung ⇒ Wendepunkt.

        higher_order_test("x^4", 0).kind      # 'relatives Minimum'
        higher_order_test("x^3", 0).kind      # 'Sattelpunkt'
        higher_order_test("-x^6", 0).k        # 6
    """
    f = parse(f)
    a = sp.nsimplify(x0) if not isinstance(x0, sp.Basic) else x0
    res_steps = []
    derivs, cur = [], f
    for k in range(1, kmax + 1):
        cur = sp.diff(cur, x)
        v = sp.simplify(cur.subs(x, a))
        if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError(f"f⁽{k}⁾ ist an der Stelle x₀ = {fmt(a)} nicht definiert.")
        derivs.append(v)
        res_steps.append((_L(f"f^{{({k})}}({fmt(a)})") if k > 3 else
                          _L(f"f{'′' * k}({fmt(a)})"), v))
        if v != 0:
            break
    res = HigherOrderResult(f, a, derivs)
    res.steps = [Step("Ableitungen an der Stelle bilden",
                      "So lange ableiten und $x_0$ einsetzen, bis $f^{(k)}(x_0) \\neq 0$.",
                      res_steps, rule="Höhere Ableitungen")]
    if res.k is None:
        res.summary = f"Bis zur Ordnung {kmax} ist jede Ableitung 0: keine Aussage (z. B. konstante Funktion)."
        res.kind = "unbestimmt"
        res.highlights = [("k", "nicht gefunden")]
        return res
    k, vk = res.k, res.value_k
    if not res.stationary:
        res.kind = "kein kritischer Punkt"
        res.summary = f"f'(x₀) = {fmt(vk)} ≠ 0: keine waagrechte Tangente, also weder Extremum noch Sattelpunkt."
    elif k % 2 == 0:
        res.kind = "relatives Minimum" if vk > 0 else "relatives Maximum"
        res.summary = (f"Erste nichtverschwindende Ableitung: k = {k} (gerade), f⁽{k}⁾(x₀) = {fmt(vk)} "
                       f"{'> 0' if vk > 0 else '< 0'} ⇒ {res.kind}.")
    else:
        res.kind = "Sattelpunkt"
        res.summary = (f"Erste nichtverschwindende Ableitung: k = {k} (ungerade) ⇒ Wendepunkt mit "
                       "waagerechter Tangente (Sattelpunkt).")
    # Wendepunkt: erste nichtverschwindende Ableitung ab f'' hat ungerade Ordnung
    cur = sp.diff(f, x)
    for j in range(2, kmax + 1):
        cur = sp.diff(cur, x)
        if sp.simplify(cur.subs(x, a)) != 0:
            res.inflection = j % 2 == 1
            break
    res.steps.append(Step("Schluss", "Folie 5: k gerade ⇒ Extremum (Vorzeichen von $f^{(k)}$), "
                          "k ungerade ⇒ Wendepunkt.",
                          [("k", k), (_L(f"f^{{({k})}}(x_0)"), vk), ("Art", res.kind)],
                          rule="Höhere Ableitungen"))
    res.highlights = [("k", k), (_L(f"f^{{({k})}}(x_0)"), vk), ("Art", res.kind),
                      ("Wendepunkt", "ja" if res.inflection else "nein")]
    return res
