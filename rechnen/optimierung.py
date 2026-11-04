#!/usr/bin/env python3
"""
Semesterwoche 7: Optimierungsprobleme (Folien SW7 und Notizen im Unterricht).

Verfahren wie im Unterricht (Zylinder in Kugel, Zaun an der Wand, Holzstück):

    1. Zielfunktion aufstellen (hängt in der Regel von zwei Variablen ab,
       soll Maximum oder Minimum werden),
    2. Nebenbedingung aufstellen und damit eine Variable eliminieren,
    3. erste Ableitung null setzen (kritische Stellen),
    4. kritische Stellen mit der zweiten Ableitung prüfen
       (Z'' < 0 ⇒ Maximum, Z'' > 0 ⇒ Minimum),
    5. Antwort: Variablen und Wert der Zielfunktion.

    from anakit.optimierung import optimize

    # Zylinder grösstes Volumen in Kugel mit Radius R (Notizen SW7)
    optimize("r^2*pi*h", "R^2 = r^2 + (h/2)^2", eliminate="r^2", var="h",
             domain=(0, "2R"))                                 # h = 2R/√3

    # Holzstück (Folien 9/10): Zielfunktion A(x) = (4 − x)(4 + 2x − x²)
    optimize("(4-x)*(4+2x-x^2)", var="x")                      # Max bei 2 − (2/3)√6

Auf Folie 10 steht x₂ und B(x₁) als 3.3630; richtig ist 2 + (2/3)·√6 ≈ 3.6330
(Zahlendreher auf der Folie, anakit rechnet richtig).

Namen dürfen aus mehreren Zeichen bestehen (L1, L2, R). Alle Variablen ausser
`var` und der eliminierten gelten als positive Parameter.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication,
                                        parse_expr, standard_transformations)

from .core import _NAMESPACE, _UNSAFE, fmt
from .render import Result, Step

__all__ = ["optimize", "OptimizeResult"]

_TRANSFORMS = standard_transformations + (implicit_multiplication, convert_xor)
_MAX, _MIN = "Maximum", "Minimum"


# ==========================================================================
# Einlesen
# ==========================================================================

def _read(text) -> sp.Expr:
    """Term einlesen; Namen wie L1 bleiben ein Symbol (kein Aufspalten in L·1)."""
    if isinstance(text, sp.Basic):
        return text
    text = str(text).strip().replace("−", "-").replace("·", "*").replace("²", "^2") \
        .replace("³", "^3").replace("√", "sqrt")
    if _UNSAFE.search(text):
        raise ValueError(f"Unzulässige Zeichen im Term: {text!r}")
    return parse_expr(text, local_dict=_NAMESPACE, transformations=_TRANSFORMS, evaluate=True)


def _real(expr, positive=()):
    """Alle Symbole reell machen (nach Name); die Parameter in `positive` positiv."""
    sub = {s: sp.Symbol(s.name, real=True, positive=True if s.name in positive else None)
           for s in expr.free_symbols}
    return expr.xreplace(sub)


def _split_equation(text):
    if isinstance(text, sp.Equality):
        return text.lhs - text.rhs
    s = str(text)
    if s.count("=") != 1 or any(op in s for op in ("<=", ">=", "==", "!=")):
        raise ValueError("Die Nebenbedingung braucht genau ein Gleichheitszeichen, "
                         "z. B. 'L = 2 L1 + L2'.")
    left, right = s.split("=")
    return _read(left) - _read(right)


def _num_sign(e, params) -> int:
    """Vorzeichen von e; bei Parametern entscheidet notfalls der Wert mit allen Parametern = 1."""
    e = sp.simplify(e)
    if e.is_zero:
        return 0
    if e.is_positive:
        return 1
    if e.is_negative:
        return -1
    v = complex(sp.N(e.subs({p: 1 for p in params}), 30))
    if abs(v.imag) > 1e-12 or abs(v.real) < 1e-12:
        return 0
    return 1 if v.real > 0 else -1


def _inside(s, dom, params) -> bool:
    if dom is None:
        return True
    lo, hi = dom
    return _num_sign(s - lo, params) > 0 and _num_sign(hi - s, params) > 0


# ==========================================================================
# Ergebnis
# ==========================================================================

class OptimizeResult(Result):
    """Zielfunktion, Nebenbedingung, kritische Stellen und Antwort eines Optimierungsproblems."""

    title = "Optimierungsproblem"

    def __init__(self, goal):
        super().__init__()
        self.goal = goal
        self.var = None
        self.target = None          #: Zielfunktion mit allen Variablen
        self.objective = None       #: Zielfunktion Z(var) nach der Elimination
        self.derivative = None
        self.second = None
        self.domain = None
        self.params: list = []
        #: Liste von dicts: x0, value, z2, kind, inside
        self.critical: list[dict] = []
        self.best = None            #: optimale Stelle
        self.best_value = None      #: Wert der Zielfunktion dort
        self.others: dict = {}      #: eliminierte Variable(n) an der optimalen Stelle

    def figure(self, xlim=None, dark: bool = False, **kw):
        from ..grafik.plots import plot_function
        from ..rechnen.core import x as X
        z = self.objective.subs({p: 1 for p in self.params}).subs(self.var, X)
        pts = []
        for c in self.critical:
            if c["inside"]:
                pts.append((float(sp.N(c["x0"].subs({p: 1 for p in self.params}))),
                            float(sp.N(c["value"].subs({p: 1 for p in self.params}))), c["kind"]))
        if xlim is None:
            if self.domain is not None:
                lo, hi = (float(sp.N(b.subs({p: 1 for p in self.params}))) for b in self.domain)
            else:
                xs = [p[0] for p in pts] or [0.0]
                lo, hi = min(xs) - 2, max(xs) + 2
            pad = 0.1 * (hi - lo)
            xlim = (lo - pad, hi + pad)
        title = (f"Zielfunktion Z({self.var}), alle Parameter = 1" if self.params
                 else f"Z({self.var}) = {fmt(self.objective)}")
        return plot_function(z, xlim, points=pts, dark=dark, title=title, **kw)


# ==========================================================================
# Verfahren
# ==========================================================================

def optimize(target, constraint=None, eliminate=None, var=None, domain=None,
             goal: str = "max", closed: bool = False) -> OptimizeResult:
    """
    Optimierungsproblem lösen: Zielfunktion, Nebenbedingung, Z'(var) = 0, Test mit Z''.

    target      Zielfunktion als Term (z. B. "r^2*pi*h").
    constraint  Nebenbedingung als Gleichung (z. B. "R^2 = r^2 + (h/2)^2"); weglassen,
                wenn `target` schon nur von `var` abhängt.
    eliminate   Was aus der Nebenbedingung berechnet wird: eine Variable ("L1") oder
                eine Potenz ("r^2"; im Unterricht wird r² = R² − (h/2)² gelöst).
    var         Die Variable, die übrig bleibt (z. B. "h"). Ohne Angabe: "x" bzw. die
                einzige Variable der Zielfunktion.
    domain      (untere, obere) Grenze von `var`, z. B. (0, "2R"). Kritische Stellen
                ausserhalb werden verworfen (wie das negative h im Unterricht).
    goal        "max" oder "min".
    closed      True, wenn die Grenzen von `domain` angenommen werden (Randwerte zählen mit).

        r = optimize("r^2*pi*h", "R^2 = r^2 + (h/2)^2", eliminate="r^2", var="h",
                     domain=(0, "2R"))
        r.best          # 2*sqrt(3)*R/3
        r.best_value    # 4*sqrt(3)*pi*R**3/9
    """
    goal = str(goal).lower()
    if goal in ("max", "maximum", "maximal"):
        goal = _MAX
    elif goal in ("min", "minimum", "minimal"):
        goal = _MIN
    else:
        raise ValueError("goal muss 'max' oder 'min' sein.")
    res = OptimizeResult(goal)

    tgt = _read(target)
    con = _split_equation(constraint) if constraint is not None else None
    elim = _read(eliminate) if eliminate is not None else None
    if (con is None) != (elim is None):
        raise ValueError("Nebenbedingung und `eliminate` gehören zusammen: beide angeben oder keins.")
    names = {s.name for s in tgt.free_symbols | (con.free_symbols if con is not None else set())}
    if var is None:
        names_t = {s.name for s in tgt.free_symbols}
        var = "x" if "x" in names_t or len(names_t) != 1 else next(iter(names_t))
    var_name = str(var).strip()
    if var_name not in names:
        raise ValueError(f"Die Variable {var_name!r} kommt nicht vor; mit var=... angeben, "
                         "welche Variable übrig bleiben soll.")
    elim_names = {s.name for s in elim.free_symbols} if elim is not None else set()
    params = sorted(names - {var_name} - elim_names)
    tgt = _real(tgt, params)
    if con is not None:
        con = _real(con, params)
        elim = _real(elim, params)
    V = sp.Symbol(var_name, real=True)
    P = [sp.Symbol(n, real=True, positive=True) for n in params]
    res.var, res.params, res.target = V, P, tgt
    dom = None
    if domain is not None:
        dom = tuple(_real(_read(b), params) if not isinstance(b, (int, float)) else sp.nsimplify(b)
                    for b in domain)
        res.domain = dom
    S = res.steps

    # 1. Zielfunktion
    S.append(Step("Zielfunktion",
                  f"Die Grösse, die {'maximal' if goal == _MAX else 'minimal'} werden soll, als Term "
                  "aufschreiben; sie hängt in der Regel von zwei Variablen ab.",
                  [("Z", tgt), ("Ziel", "Max!" if goal == _MAX else "Min!")],
                  rule="Zielfunktion"))

    # 2. Nebenbedingung und Elimination
    Z = tgt
    if con is not None:
        d = sp.Dummy("d")
        # eliminate ist eine Variable oder eine Potenz wie r**2: dafür ein Platzhalter-Symbol
        con_d = con.subs(elim, d)
        if con_d.free_symbols & elim.free_symbols:
            raise ValueError(f"Aus der Nebenbedingung lässt sich {fmt(elim)} nicht herauslösen "
                             "(es kommt noch in anderer Form vor): `eliminate` anders wählen, "
                             "z. B. die Variable statt ihres Quadrats.")
        sols = sp.solve(con_d, d)
        if not sols:
            raise ValueError(f"Die Nebenbedingung lässt sich nicht nach {fmt(elim)} auflösen.")
        if len(sols) > 1:
            mid = (sp.N(sum(dom).subs({p: 1 for p in P})) / 2 if dom else 1)
            ok = [s for s in sols if _num_sign(s.subs(V, mid), P) > 0]
            sols = ok or sols
        sol = sp.simplify(sols[0])
        Zc = tgt.subs(elim, sol)
        if Zc.free_symbols & elim.free_symbols:
            raise ValueError(f"Die Zielfunktion enthält {fmt(elim)} in anderer Form; "
                             "wähle `eliminate` passend (z. B. die Variable statt ihres Quadrats).")
        Z = sp.expand(sp.simplify(Zc)) if sp.expand(Zc).is_polynomial(V) else sp.simplify(Zc)
        res.eliminated = (elim, sol)
        S.append(Step("Nebenbedingung",
                      "Die Variablen hängen zusammen; die Nebenbedingung nach einer Grösse auflösen.",
                      [("NB", sp.Eq(con, 0, evaluate=False)), (fmt(elim), sol)],
                      rule="Nebenbedingung"))
        S.append(Step("Eine Variable eliminieren",
                      f"Die aufgelöste Grösse in die Zielfunktion einsetzen: Z hängt nur noch von ${var_name}$ ab.",
                      [("Z(" + var_name + ")", Z)], rule="Elimination"))
    if con is None and sp.expand(Z).is_polynomial(V):
        Z = sp.expand(Z)
        if Z != tgt:
            S.append(Step("Zielfunktion ausmultiplizieren", "Für das Ableiten als Polynom schreiben.",
                          [("Z(" + var_name + ")", Z)]))
    res.objective = Z
    if Z.free_symbols - {V} - set(P):
        raise ValueError("In der Zielfunktion bleibt eine zweite Variable stehen: "
                         "Nebenbedingung und `eliminate` anpassen.")

    # 3. Erste Ableitung
    d1 = sp.simplify(sp.diff(Z, V))
    d2 = sp.simplify(sp.diff(d1, V))
    res.derivative, res.second = d1, d2
    cands = sorted({sp.simplify(s) for s in sp.solve(d1, V) if s.is_real is not False},
                   key=lambda s: float(sp.N(s.subs({p: 1 for p in P}))))
    S.append(Step("Erste Ableitung null setzen",
                  f"Kritische Stellen: $Z'({var_name}) = 0$.",
                  [("Z'(" + var_name + ")", d1)] + [(f"{var_name}{_sub(i)}", c) for i, c in enumerate(cands, 1)]
                  or [("", "keine kritische Stelle")],
                  rule="kritische Stellen"))

    # 4. Prüfen
    rows = []
    for i, c in enumerate(cands, 1):
        inside = _inside(c, dom, P)
        z2 = sp.simplify(d2.subs(V, c))
        sg = _num_sign(z2, P)
        kind = _MAX if sg < 0 else _MIN if sg > 0 else "unbestimmt"
        val = sp.simplify(Z.subs(V, c))
        rows.append(dict(x0=c, value=val, z2=z2, kind=kind, inside=inside))
        lab = f"{var_name}{_sub(i)}"
        if not inside:
            S.append(Step(f"Prüfung bei {lab}", f"${lab}$ liegt nicht im zulässigen Bereich "
                          f"({fmt(dom[0])} < {var_name} < {fmt(dom[1])}) und wird verworfen.",
                          [(lab, c)]))
            continue
        rel = "<" if sg < 0 else ">" if sg > 0 else "="
        concl = (f"⇒ {kind}" if sg else "⇒ keine Aussage möglich (Z'' = 0)")
        S.append(Step(f"Prüfung bei {lab} mit der zweiten Ableitung",
                      f"$Z''({lab}) {rel} 0$ {concl}.",
                      [("Z''(" + var_name + ")", d2), (f"Z''({lab})", z2), (f"Z({lab})", val)],
                      rule="Z'' < 0 ⇒ Maximum, Z'' > 0 ⇒ Minimum"))
    res.critical = rows
    good = [r for r in rows if r["inside"] and r["kind"] == goal]

    # 5. Randwerte (nur bei angenommenen Grenzen)
    pool = list(good)
    if dom is not None and closed:
        items = []
        for b in dom:
            val = sp.simplify(Z.subs(V, b))
            items.append((f"Z({fmt(b)})", val))
            pool.append(dict(x0=b, value=val, kind="Rand", inside=True, z2=None))
        S.append(Step("Randwerte", "Die Grenzen gehören zum Bereich: ihre Werte der Zielfunktion vergleichen.",
                      items, rule="Randwerte prüfen"))
    if not pool:
        res.summary = "Keine Lösung mit dem gesuchten Extremum im zulässigen Bereich."
        res.warnings.append(res.summary)
        return res
    best = (max if goal == _MAX else min)(pool, key=lambda r: float(sp.N(r["value"].subs({p: 1 for p in P}))))
    res.best, res.best_value = best["x0"], best["value"]

    # 6. Antwort
    items = [(var_name, res.best)]
    if con is not None:
        elim_v, sol = res.eliminated
        ev = sp.simplify(sol.subs(V, res.best))
        res.others[fmt(elim_v)] = ev
        items.append((fmt(elim_v), ev))
        if elim_v.is_Pow and elim_v.exp == 2 and elim_v.base.is_Symbol:
            root = sp.simplify(sp.sqrt(ev))
            res.others[fmt(elim_v.base)] = root
            items.append((fmt(elim_v.base), root))
    items.append(("Z", res.best_value))
    note = ""
    if dom is None:
        note = ("Ohne angegebenen Bereich ist das ein relatives Extremum; ob es auch das grösste bzw. "
                "kleinste im zulässigen Bereich ist, hängt von den Randwerten ab.")
    elif not closed:
        note = "Die Grenzen des Bereichs werden nicht angenommen (offenes Intervall), also zählen keine Randwerte."
    S.append(Step("Antwort", "Optimale Stelle, die eliminierte(n) Grösse(n) dort und der Wert der Zielfunktion.",
                  items, note=note))
    res.summary = f"{goal} bei {var_name} = {fmt(res.best)}: Z = {fmt(res.best_value)}."
    res.highlights = [(k, v) for k, v in items]
    return res


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
