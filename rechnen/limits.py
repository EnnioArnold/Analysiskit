#!/usr/bin/env python3
"""
Grenzwert und Stetigkeit (Semesterwoche 2).

Jedes Verfahren liefert ein Ergebnisobjekt mit Rechenweg:

    limit(f, x0, side='both')      Grenzwert an einer Stelle (links/rechts/beide)
    limit_infinity(f, '+')         Verhalten für x -> +oo bzw. -oo
    limit_laws(f, g, x0)           Grenzwertsätze an einem Beispiel
    continuity(f, x0)              Stetigkeit an einer Stelle
    is_continuous(f, domain=None)  Stetigkeit der ganzen Funktion
    limit_table(f, x0, side)       Wertetabelle zur Annäherung

Betrag, Signum und Heaviside werden intern abschnittsweise geschrieben
(|x|, sgn x, H(x) mit H(0) = 1, wie auf der Folie), damit man den
passenden Ast je Seite wählen kann.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .core import fmt, num, parse, tex, x
from .render import Result, Step

OO = sp.oo
_EPS = sp.Rational(1, 10 ** 9)          # Testabstand zur Wahl des Astes
_SUP = {1: "⁺", -1: "⁻", 0: ""}


# ==========================================================================
# Hilfen
# ==========================================================================

def _S(title, detail="", **kw):
    """Step; sorgt dafür, dass $...$ im Konsolentext aufgelöst wird."""
    if "$" in detail and "\\" not in detail:
        detail += "\\,"
    return Step(title, detail, **kw)


def _show(e):
    """Darstellung für Items: abschnittsweise Funktionen als Fallunterscheidung."""
    if isinstance(e, sp.Piecewise):
        rows, prev = [], []
        for ex, c in e.args:
            ct = _neg_cond_text(prev) if c in (True, sp.true) else tex(c)
            rows.append(tex(ex) + " & " + (r"\text{sonst}" if ct == "sonst" else ct))
            prev.append(c)
        return r"\begin{cases} " + r" \\ ".join(rows) + r" \end{cases}"
    return e


def _T(e) -> str:
    """Term als Inline-LaTeX für Fliesstext."""
    return "$" + tex(e) + "$"


def _prep(f) -> sp.Expr:
    """Funktion einlesen; Betrag, Signum, Heaviside -> abschnittsweise."""
    e = parse(f)
    e = e.replace(lambda t: isinstance(t, sp.Abs),
                  lambda t: sp.Piecewise((-t.args[0], t.args[0] < 0), (t.args[0], True)))
    e = e.replace(lambda t: isinstance(t, sp.sign),
                  lambda t: sp.Piecewise((-1, t.args[0] < 0),
                                         (0, sp.Eq(t.args[0], 0)), (1, True)))
    e = e.replace(lambda t: isinstance(t, sp.Heaviside),
                  lambda t: sp.Piecewise((0, t.args[0] < 0), (1, True)))
    return e


def _x0(value):
    """Stelle einlesen: Zahl, sp.oo, '-oo', 'pi/2' ..."""
    if isinstance(value, str):
        v = value.strip().replace("∞", "oo").replace("−", "-")
        if v in ("+oo", "+inf", "inf"):
            return OO
        return parse(v)
    return num(value) if not isinstance(value, sp.Basic) else value


def _is_inf(x0) -> bool:
    return x0 in (OO, -OO)


def _side_arg(side) -> str:
    s = str(side).strip().lower()
    if s in ("+", "rechts", "right", "r", "plus"):
        return "+"
    if s in ("-", "−", "links", "left", "l", "minus"):
        return "-"
    if s in ("both", "beide", "beidseitig", "", "b"):
        return "both"
    raise ValueError("side muss '+', '-' oder 'both' sein")


def _testpoint(x0, d):
    if x0 == OO:
        return sp.Integer(10 ** 9)
    if x0 == -OO:
        return -sp.Integer(10 ** 9)
    return x0 + d * _EPS


def _label(x0, d, name="f") -> str:
    if _is_inf(x0):
        arrow = tex(x0)
    else:
        arrow = tex(x0) + {1: "^{+}", -1: "^{-}", 0: ""}[d]
    return rf"\lim_{{x \to {arrow}}} {name}(x)"


def _title_side(x0, d) -> str:
    if _is_inf(x0):
        return f"Grenzwert (x → {fmt(x0)})"
    if d < 0:
        return f"Linksseitiger Grenzwert (x → {fmt(x0)}⁻)"
    if d > 0:
        return f"Rechtsseitiger Grenzwert (x → {fmt(x0)}⁺)"
    return f"Grenzwert (x → {fmt(x0)})"


def _even_monomial(p) -> bool:
    try:
        if not p.is_polynomial(x):
            return False
        q = sp.Poly(p, x)
        return q.is_monomial and q.degree() > 0 and q.degree() % 2 == 0
    except Exception:
        return False


def _sgn(v) -> int:
    v = sp.N(v, 40)
    return 1 if v > 0 else -1 if v < 0 else 0


def _bad(v) -> bool:
    return (v is None or v.has(sp.nan, sp.zoo, OO, -OO, sp.AccumBounds)
            or not v.is_number or v.is_real is False)


def _value_at(e, x0):
    """Funktionswert f(x0) oder None, falls nicht definiert."""
    if _is_inf(x0):
        return None
    try:
        v = sp.simplify(e.subs(x, x0))
    except Exception:
        return None
    return None if _bad(v) else v


def _defined_near(e, pt) -> bool:
    try:
        v = sp.N(e.subs(x, pt), 20)
    except Exception:
        return False
    return bool(v.is_number and v.is_real and not v.has(sp.nan, sp.zoo, OO, -OO))


def _sy_limit(e, x0, d):
    try:
        if _is_inf(x0):
            return sp.limit(e, x, x0)
        return sp.limit(e, x, x0, dir="+" if d > 0 else "-")
    except Exception:
        return None


def _cls(L) -> str:
    if L is None:
        return "unknown"
    if L.has(sp.AccumBounds):
        return "osc"
    if L == OO:
        return "pinf"
    if L == -OO:
        return "minf"
    if L.is_number and L.is_real and not L.has(sp.nan, sp.zoo):
        return "finite"
    return "unknown"


def _same(a, b) -> bool:
    try:
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return a == b


def _neg_cond_text(prev_conds):
    if not prev_conds:
        return "sonst"
    neg = [sp.Not(c) for c in prev_conds]
    return tex(neg[0] if len(neg) == 1 else sp.And(*neg))


def _localize(e, x0, d):
    """Ersetzt jede abschnittsweise Funktion durch ihren Ast auf dieser Seite."""
    pt = _testpoint(x0, d)
    info = []

    def pick(p):
        prev = []
        for ex, c in p.args:
            ok = (c is True or c == sp.true)
            if not ok:
                try:
                    ok = bool(c.subs(x, pt))
                except Exception:
                    ok = False
            if ok:
                info.append((ex, c, _neg_cond_text(prev) if c == sp.true else tex(c)))
                return ex
            prev.append(c)
        return sp.nan

    loc = e.replace(lambda t: isinstance(t, sp.Piecewise), pick)
    return loc, info


# ==========================================================================
# Rechenweg für eine Seite
# ==========================================================================

@dataclass
class _Side:
    d: int
    value: object
    kind: str            # eigentlich | uneigentlich | existiert nicht | nicht definiert
    steps: list = field(default_factory=list)
    osc: bool = False


def _std_steps(e, x0, d):
    """Standardgrenzwerte bei 0 mit Einschnürungssatz; (steps, wert) oder None."""
    if x0 != 0:
        return None
    lab = _label(x0, d, "")
    if _same(e, sp.sin(x) / x):
        return [_S("Standardgrenzwert", "Am Einheitskreis gilt für $0<x<\\pi/2$: "
                     "$\\cos x < \\frac{\\sin x}{x} < 1$. Beide Schranken streben gegen 1 "
                     "(auch von links, da $\\frac{\\sin x}{x}$ gerade ist).",
                     items=[("", sp.cos(x) < sp.sin(x) / x), ("", sp.sin(x) / x < 1)],
                     rule="Einschnürungssatz"),
                _S("Folgerung", "Wird $\\frac{\\sin x}{x}$ von zwei Funktionen mit Grenzwert 1 "
                     "eingeschnürt, hat auch sie den Grenzwert 1.",
                     items=[("Grenzwert", 1)],
                     rule="Einschnürungssatz")], sp.Integer(1)
    if _same(e, (1 - sp.cos(x)) / x):
        return [_S("Erweitern mit 1 + cos x",
                     "Mit $1-\\cos^2 x=\\sin^2 x$ wird "
                     "$\\frac{1-\\cos x}{x}=\\frac{\\sin x}{x}\\cdot\\frac{\\sin x}{1+\\cos x}$.",
                     items=[("f(x)", sp.sin(x) / x * sp.sin(x) / (1 + sp.cos(x)))],
                     rule="Erweitern"),
                _S("Grenzwerte einzeln", "$\\frac{\\sin x}{x}\\to 1$ (Standardgrenzwert, "
                     "Einschnürungssatz), $\\frac{\\sin x}{1+\\cos x}\\to\\frac{0}{2}=0$.",
                     items=[("1 \\cdot 0", 0)], rule="Produkt der Grenzwerte")], sp.Integer(0)
    if _same(e, (1 - sp.cos(x)) / x ** 2):
        return [_S("Erweitern mit 1 + cos x",
                     "Es wird $\\frac{1-\\cos x}{x^2}=\\left(\\frac{\\sin x}{x}\\right)^2"
                     "\\cdot\\frac{1}{1+\\cos x}$.",
                     items=[("f(x)", (sp.sin(x) / x) ** 2 / (1 + sp.cos(x)))],
                     rule="Erweitern"),
                _S("Grenzwerte einzeln", "$\\left(\\frac{\\sin x}{x}\\right)^2\\to 1$ "
                     "(Einschnürungssatz), $\\frac{1}{1+\\cos x}\\to\\frac12$.",
                     items=[("1 \\cdot \\frac12", sp.Rational(1, 2))],
                     rule="Produkt der Grenzwerte")], sp.Rational(1, 2)
    return None


def _derive(e, x0, d, L, cls, depth=0):
    """Rechenschritte, die zum Grenzwert L (von SymPy bestätigt) führen."""
    steps: list[Step] = []
    lab = _label(x0, d)
    pt = _testpoint(x0, d)

    # ---------------------------------------------------------------- x0 endlich
    if not _is_inf(x0):
        direct = _value_at(e, x0)
        if cls == "finite" and direct is not None and _same(direct, L):
            steps.append(_S(
                "Einsetzen", f"Der Term ist bei {_T(x0)} definiert und stetig "
                "(aus elementaren Funktionen zusammengesetzt), also darf man einsetzen.",
                items=[(lab, direct)], rule="Einsetzen"))
            return steps

        e1 = e
        if e.has(sp.tan, sp.cot):
            e1 = e.replace(sp.tan, lambda a: sp.sin(a) / sp.cos(a)) \
                  .replace(sp.cot, lambda a: sp.cos(a) / sp.sin(a))
            steps.append(_S("Tangens umschreiben",
                              "$\\tan x=\\frac{\\sin x}{\\cos x}$, so sieht man Zähler und Nenner getrennt.",
                              items=[("f(x)", e1)], rule="tan = sin/cos"))
        n, dd = sp.fraction(sp.together(e1))
        try:
            n0, d0 = sp.simplify(n.subs(x, x0)), sp.simplify(dd.subs(x, x0))
        except Exception:
            n0 = d0 = None

        if d0 is not None and d0 == 0 and n0 == 0:
            std = _std_steps(e, x0, d)
            if std:
                return steps + std[0]
            steps.append(_S("Einsetzen ergibt 0/0",
                              "Zähler und Nenner werden beide 0: unbestimmter Ausdruck, "
                              "es muss umgeformt werden.",
                              items=[("Zähler bei x\u2080", n0), ("Nenner bei x\u2080", d0)]))
            c = sp.cancel(sp.together(e1))
            if depth < 4 and c != sp.together(e1) and not c.has(sp.nan):
                steps.append(_S("Faktorisieren", "Zähler und Nenner in Linearfaktoren zerlegen.",
                                  items=[("Zähler", sp.factor(n)), ("Nenner", sp.factor(dd))],
                                  rule="Faktorisieren"))
                try:
                    g = sp.gcd(n, dd)
                except Exception:
                    g = None
                steps.append(_S("Kürzen", "Der gemeinsame Faktor " + (_T(sp.factor(g)) + " " if g is not None else "")
                                  + "darf gekürzt werden, weil beim Grenzwert $x\\neq x_0$ gilt "
                                  "(x erreicht $x_0$ nie).",
                                  items=[("f(x)", c)], rule="Kürzen"))
                return steps + _derive(c, x0, d, L, cls, depth + 1)
            steps.append(_S("Grenzwert bestimmen",
                              "Keine Vereinfachung durch Kürzen möglich; der Grenzwert folgt aus "
                              "bekannten Grenzwerten bzw. der Regel von de l'Hôpital (später in der Vorlesung).",
                              items=[(lab, L)]))
            return steps

        if d0 is not None and d0 == 0 and n0 is not None and n0 != 0 and cls in ("pinf", "minf"):
            sn, sd = _sgn(n0), _sgn(dd.subs(x, pt))
            if sn * sd != 0 and (sn * sd > 0) == (cls == "pinf"):
                nz = "positiv" if sn > 0 else "negativ"
                dz = "positiv" if sd > 0 else "negativ"
                steps.append(_S(
                    "Nenner → 0, Zähler → Zahl ≠ 0",
                    f"Der Zähler strebt gegen {_T(n0)} (also {nz}), der Nenner gegen 0. "
                    f"Auf dieser Seite ({'gerade vor' if d<0 else 'gerade nach'} {_T(x0)}) ist der "
                    f"Nenner {dz}, d. h. {'→ 0⁺' if sd>0 else '→ 0⁻'}. "
                    "Der Betrag des Bruchs wächst über alle Grenzen.",
                    items=[("Zähler", n0), ("Nenner", "0⁺" if sd > 0 else "0⁻")],
                    rule="Bruch mit Nenner → 0"))
                steps.append(_S("Vorzeichen",
                                  f"{nz.capitalize()} durch {dz} ergibt "
                                  f"{'positiv' if sn*sd>0 else 'negativ'}: uneigentlicher Grenzwert.",
                                  items=[(lab, L)],
                                  note="Gerade Potenz im Nenner: gleiches Vorzeichen auf beiden Seiten."
                                  if _even_monomial(dd) else ""))
                return steps

        # Schwingung
        if cls == "osc":
            lo, hi = L.min, L.max
            note = ""
            detail = ("Das Argument durchläuft in jeder Umgebung von $x_0$ unendlich oft alle "
                      "Werte, die Funktion pendelt immer wieder zwischen den Werten "
                      f"{_T(lo)} und {_T(hi)} hin und her. Sie nähert sich keiner Zahl.")
            if _same(e, sp.sin(1 / x)) and x0 == 0:
                detail = ("Für $x_n=\\frac{1}{\\pi/2+2n\\pi}$ ist $\\sin(1/x_n)=1$, für "
                          "$x_n=\\frac{1}{3\\pi/2+2n\\pi}$ ist $\\sin(1/x_n)=-1$. Beide Folgen streben "
                          "gegen 0, die Funktionswerte aber gegen verschiedene Zahlen.")
            steps.append(_S("Schwingung statt Annäherung", detail,
                              items=[("Schwankung", f"zwischen {fmt(lo)} und {fmt(hi)}")],
                              note="Kein Grenzwert: es gibt keine Zahl L, der sich die Werte nähern.",
                              rule="Folgenkriterium"))
            return steps

        if cls in ("pinf", "minf"):
            steps.append(_S("Werte wachsen über alle Grenzen",
                              "Der Funktionswert wird auf dieser Seite beliebig "
                              + ("gross." if cls == "pinf" else "klein (negativ)."),
                              items=[(lab, L)], note="uneigentlicher Grenzwert"))
            return steps
        steps.append(_S("Grenzwert bestimmen", "Aus bekannten Grenzwerten bzw. Rechenregeln.",
                          items=[(lab, L)]))
        return steps

    # ---------------------------------------------------------------- x0 = +-oo
    try:
        n, dd = sp.fraction(sp.cancel(sp.together(e)))
    except Exception:
        n, dd = e, sp.Integer(1)
    if n.is_polynomial(x) and dd.is_polynomial(x) and dd != 0 and (n.has(x) or dd.has(x)):
        k = sp.degree(n, x) if n != 0 else -1
        m = sp.degree(dd, x)
        steps.append(_S("Grade vergleichen",
                          "Zähler und Nenner durch die höchste Nennerpotenz "
                          f"{_T(x**m)} teilen (ausklammern).",
                          items=[("Grad Zähler", k), ("Grad Nenner", m)], rule="Höchste Potenz ausklammern"))
        if k <= m:
            zn, nn = sp.expand(n / x ** m), sp.expand(dd / x ** m)
            steps.append(_S("Durch " + fmt(x ** m) + " teilen", "",
                              items=[("Zähler", zn), ("Nenner", nn),
                                     ("f(x)", sp.Mul(zn, sp.Pow(nn, -1, evaluate=False), evaluate=False))]))
            lz, ln_ = sp.limit(zn, x, x0), sp.limit(nn, x, x0)
            steps.append(_S("Terme 1/x\u1d4f streben gegen 0",
                              "Für $k>0$ gilt $\\frac{1}{x^k}\\to0$ bei $x\\to\\pm\\infty$; "
                              "übrig bleiben die Koeffizienten der höchsten Potenz.",
                              items=[("Grenzwert Zähler", lz), ("Grenzwert Nenner", ln_)],
                              rule="$1/x^k \\to 0$"))
        else:
            lc = sp.Poly(n, x).LC() / sp.Poly(dd, x).LC()
            steps.append(_S("Zähler wächst schneller",
                              f"Der Zähler hat höheren Grad ({k}>{m}); der Bruch verhält sich wie "
                              f"{_T(lc * x ** (k - m))}. Das Vorzeichen folgt aus dem Quotienten der "
                              "Leitkoeffizienten und dem Vorzeichen von $x^{k-m}$ für "
                              + ("$x\\to+\\infty$." if x0 == OO else "$x\\to-\\infty$."),
                              items=[("Leitkoeffizient Zähler/Nenner", lc)],
                              note="uneigentlicher Grenzwert"))
        steps.append(_S("Ergebnis", "", items=[(lab, L)]))
        return steps
    if n.func in (sp.sin, sp.cos) and dd.is_polynomial(x) and sp.degree(dd, x) >= 1:
        steps.append(_S("Beschränkter Zähler, wachsender Nenner",
                          f"Es gilt $|{tex(n)}|\\le1$, also $\\left|f(x)\\right|\\le\\frac{{1}}{{|{tex(dd)}|}}\\to0$.",
                          items=[(lab, L)], rule="Einschnürungssatz"))
        return steps
    steps.append(_S("Bekannte Grenzwerte",
                      "Verhalten der Grundfunktionen für $x\\to\\pm\\infty$ "
                      "(z. B. $e^{x}$, $\\arctan x$, $1/x\\to0$).", items=[(lab, L)]))
    return steps


def _side(e, x0, d) -> _Side:
    """Grenzwert von e für x -> x0 von der Seite d (+1 rechts, -1 links, 0 für ±oo)."""
    steps: list[Step] = []
    title = _title_side(x0, d)
    loc, info = _localize(e, x0, d)
    if info:
        seite = "für" if _is_inf(x0) else "in der Nähe von x0 auf dieser Seite, also für"
        lines = []
        for ex, c, ctext in info:
            lines.append(f"Es gilt {'$' + ctext + '$' if ctext != 'sonst' else 'sonst'}, "
                         f"also der Ast {_T(ex)}.")
        steps.append(_S(f"{title}: passenden Ast wählen",
                          f"Die Funktion ist abschnittsweise definiert. Gesucht ist das Verhalten "
                          f"{seite} $x \\to {tex(x0)}{'^{+}' if d>0 else '^{-}' if d<0 else ''}$. "
                          + " ".join(lines),
                          items=[("f(x) auf diesem Ast", loc)], rule="Ast wählen"))
    pt = _testpoint(x0, d)
    if not _defined_near(loc, pt):
        steps.append(_S(f"{title}: nicht definiert",
                          f"Auf dieser Seite von {_T(x0)} ist die Funktion nicht (reell) definiert; "
                          "dort kann es keinen Grenzwert geben."))
        return _Side(d, None, "nicht definiert", steps)
    L = _sy_limit(loc, x0, d)
    cls = _cls(L)
    sub = _derive(loc, x0, d, L, cls)
    if not info and sub:
        sub[0].title = f"{title}: {sub[0].title}"
    elif sub:
        pass
    steps += sub
    if cls == "finite":
        kind, val = "eigentlich", L
    elif cls in ("pinf", "minf"):
        kind, val = "uneigentlich", L
    elif cls == "osc":
        kind, val = "existiert nicht", None
    else:
        kind, val = "existiert nicht", None
        steps.append(_S("Nicht bestimmbar", "Der Grenzwert konnte nicht bestimmt werden."))
    return _Side(d, val, kind, steps, osc=(cls == "osc"))


# ==========================================================================
# LimitResult
# ==========================================================================

def _kind_text(s: _Side) -> str:
    if s.kind == "eigentlich":
        return fmt(s.value)
    if s.kind == "uneigentlich":
        return fmt(s.value) + " (uneigentlich)"
    if s.kind == "nicht definiert":
        return "nicht definiert"
    return "existiert nicht" + (" (Schwingung)" if s.osc else "")


class LimitResult(Result):
    """
    Grenzwert einer Funktion an einer Stelle.

    `left`/`right`: einseitige Grenzwerte (bei x -> ±oo und bei einseitiger
    Anfrage jeweils None), `value`: der Grenzwert (None, wenn er nicht
    existiert), `exists`: ob ein Grenzwert (auch uneigentlich) existiert,
    `kind`: 'eigentlich' | 'uneigentlich' | 'existiert nicht'.
    """

    def __init__(self, f, expr, x0, side):
        super().__init__()
        self.f = f
        self.expr = expr
        self.x0 = x0
        self.side = side
        self.left = self.right = self.value = None
        self.left_kind = self.right_kind = None
        self.kind = "existiert nicht"
        self.exists = False
        self.reason = ""

    def figure(self, **kw):
        from . import plots
        if _is_inf(self.x0):
            kw.setdefault("xlim", (-10, 10))
            return plots.plot_function(self.expr, **kw)
        return plots.plot_limit(self.expr, self.x0, **kw)


def _limit_core(f, x0, side, title_prefix="Grenzwert") -> LimitResult:
    orig = parse(f)
    e = _prep(orig)
    x0 = _x0(x0)
    side = "both" if _is_inf(x0) else _side_arg(side)
    res = LimitResult(f, e, x0, side)
    res.title = f"{title_prefix}: {_label(x0, {'+':1,'-':-1,'both':0}[side])}"
    has_pw = e.has(sp.Piecewise)

    items = [("f(x)", _show(orig))]
    if e != orig and isinstance(e, sp.Piecewise):
        items.append(("f(x) abschnittsweise", _show(e)))
    res.steps.append(_S("Funktion", f"Gesucht ist ${_label(x0, {'+':1,'-':-1,'both':0}[side])}$."
                          + (" Betrag, Signum bzw. Heaviside werden abschnittsweise geschrieben "
                             "(H(x)=0 für x<0, H(x)=1 für x≥0)." if e != orig else ""),
                          items=items))

    def finish_single(s, d):
        res.steps += s.steps
        res.value, res.kind = s.value, s.kind if s.kind != "nicht definiert" else "existiert nicht"
        res.exists = res.kind in ("eigentlich", "uneigentlich")
        return s

    if _is_inf(x0):
        s = _side(e, x0, 0)
        finish_single(s, 0)
        res.right = res.left = None
        res.summary = f"${_label(x0, 0)}$ = {_kind_text(s)}"
        res.highlights = [("Grenzwert", s.value if s.value is not None else "existiert nicht")]
        if s.kind == "uneigentlich":
            res.steps.append(_S("Ergebnis", "Die Funktionswerte wachsen über alle Schranken: "
                                  "uneigentlicher Grenzwert (kein Grenzwert im Sinne einer Zahl L).",
                                  items=[("", s.value)]))
        return res

    if side in ("+", "-"):
        d = 1 if side == "+" else -1
        s = _side(e, x0, d)
        finish_single(s, d)
        if d > 0:
            res.right, res.right_kind = s.value, s.kind
        else:
            res.left, res.left_kind = s.value, s.kind
        res.summary = f"${_label(x0, d)}$ = {_kind_text(s)}"
        res.highlights = [("rechtsseitig" if d > 0 else "linksseitig",
                           s.value if s.value is not None else "existiert nicht")]
        return res

    # --- beidseitig
    sl, sr = _side(e, x0, -1), _side(e, x0, 1)
    res.left, res.right = sl.value, sr.value
    res.left_kind, res.right_kind = sl.kind, sr.kind
    merged = (not has_pw and sl.kind == sr.kind == "eigentlich" and _same(sl.value, sr.value))
    if merged:
        st = sr.steps
        st[0].title = f"Beidseitige Annäherung: {st[0].title.split(': ')[-1]}"
        st[0].note = (st[0].note + " " if st[0].note else "") + \
            "Gilt für x → x₀⁻ und x → x₀⁺ gleichermassen."
        res.steps += st
    else:
        res.steps += sl.steps + sr.steps

    lab_l, lab_r = _label(x0, -1), _label(x0, 1)
    comp = _S("Vergleich links / rechts",
                "Nur wenn linksseitiger und rechtsseitiger Grenzwert existieren und übereinstimmen, "
                "kann man von einem Grenzwert sprechen.",
                items=[(lab_l, sl.value if sl.value is not None else _kind_text(sl)),
                       (lab_r, sr.value if sr.value is not None else _kind_text(sr))])
    res.steps.append(comp)

    def judge():
        a, b = sl, sr
        if a.kind == "nicht definiert" or b.kind == "nicht definiert":
            return ("existiert nicht", None,
                    "Die Funktion ist nur auf einer Seite von x₀ definiert; ein beidseitiger "
                    "Grenzwert existiert nicht (nur der einseitige).")
        if a.osc or b.osc:
            return ("existiert nicht", None, "Die Funktion schwingt: kein Grenzwert.")
        if a.kind == b.kind == "eigentlich":
            if _same(a.value, b.value):
                return ("eigentlich", a.value, f"links = rechts = {fmt(a.value)}: der Grenzwert existiert.")
            return ("existiert nicht", None,
                    f"links ({fmt(a.value)}) ≠ rechts ({fmt(b.value)}): der Grenzwert existiert nicht "
                    f"(Sprung der Höhe {fmt(sp.simplify(b.value - a.value))}).")
        if a.kind == b.kind == "uneigentlich":
            if a.value == b.value:
                return ("uneigentlich", a.value,
                        f"Beide Seiten streben gegen {fmt(a.value)}: uneigentlicher Grenzwert {fmt(a.value)}.")
            return ("existiert nicht", None,
                    "Links und rechts streben die Werte gegen verschiedene Unendlichkeiten "
                    f"({fmt(a.value)} bzw. {fmt(b.value)}): kein gemeinsamer Grenzwert "
                    "(Polstelle mit Vorzeichenwechsel).")
        return ("existiert nicht", None, "Die beiden einseitigen Grenzwerte sind verschieden "
                "(einer ist endlich, der andere nicht): der Grenzwert existiert nicht.")

    res.kind, res.value, res.reason = judge()
    res.exists = res.kind in ("eigentlich", "uneigentlich")
    verdict = _S("Ergebnis", res.reason, items=[(_label(x0, 0), res.value if res.exists else "existiert nicht")],
                   note="uneigentlicher Grenzwert: f wächst über alle Schranken (kein Grenzwert im Sinne einer Zahl)."
                   if res.kind == "uneigentlich" else "")
    if orig == sp.tan(x) and x0 == sp.pi / 2:
        verdict.note = ("Hinweis zur Folie: Dort sind die Vorzeichen bei tan(x) vertauscht angeschrieben. "
                        "Korrekt ist: von links (x<π/2) ist tan x > 0, also → +∞; von rechts "
                        "(x>π/2) ist tan x < 0, also → −∞. Probe: tan(1.5) ≈ 14.1, tan(1.6) ≈ −34.2.")
    res.steps.append(verdict)
    res.summary = f"${_label(x0, 0)}$: " + (fmt(res.value) if res.exists else "existiert nicht") + \
        (" (uneigentlich)" if res.kind == "uneigentlich" else "")
    if res.exists:
        res.highlights = [("Grenzwert", res.value)]
    else:
        res.highlights = [("linksseitig", sl.value if sl.value is not None else _kind_text(sl)),
                          ("rechtsseitig", sr.value if sr.value is not None else _kind_text(sr)),
                          ("Grenzwert", "existiert nicht")]
    return res


def limit(f, x0, side: str = "both") -> LimitResult:
    """
    Grenzwert von f(x) für x -> x0.

        limit("(x^2-1)/(x-1)", 1)                    # 2
        limit("x^2 if x<1 else 2-x", 1)               # 1
        limit("1/x^2", 0, "+")                        # +oo (uneigentlich)
        limit("tan(x)", "pi/2", "-")                  # +oo

    side: '+' (rechts), '-' (links) oder 'both'. x0 darf auch oo / '-oo' sein.
    """
    return _limit_core(f, x0, side)


def limit_infinity(f, direction="+") -> LimitResult:
    """Verhalten für x -> +oo (direction '+') bzw. x -> -oo ('-')."""
    d = str(direction).strip().replace("−", "-")
    neg = d.startswith("-") or d in ("minus", "links")
    r = _limit_core(f, -OO if neg else OO, "both")
    r.title = f"Verhalten im Unendlichen: {_label(-OO if neg else OO, 0)}"
    return r


# ==========================================================================
# Grenzwertsätze
# ==========================================================================

class LimitLawsResult(Result):
    """Grenzwertsätze an einem Beispiel: `a`, `b`, `checks` (Name -> bool), `all_ok`."""

    def __init__(self):
        super().__init__()
        self.a = self.b = None
        self.checks: dict[str, bool] = {}
        self.f = self.g = self.x0 = None

    @property
    def all_ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def figure(self, **kw):
        from . import plots
        return plots.plot_limit(self.f + self.g, self.x0, **kw)


def limit_laws(f, g, x0) -> LimitLawsResult:
    """
    Grenzwertsätze für f und g an der Stelle x0 durchrechnen.

    Bestimmt a = lim f, b = lim g und zeigt a±b, a·b, a/b (b≠0) und die
    Verkettung f(g(x)) (äussere Funktion f stetig bei b). Jedes Ergebnis wird
    durch den Grenzwert der zusammengesetzten Funktion kontrolliert.
    """
    ef, eg, x0 = _prep(f), _prep(g), _x0(x0)
    res = LimitLawsResult()
    res.f, res.g, res.x0 = ef, eg, x0
    res.title = f"Grenzwertsätze bei x → {fmt(x0)}"
    lf, lg = _limit_core(ef, x0, "both"), _limit_core(eg, x0, "both")
    res.steps.append(_S("Funktionen", "", items=[("f(x)", ef), ("g(x)", eg)]))
    res.steps.append(_S("Einzelne Grenzwerte", "Beide Grenzwerte müssen als Zahlen existieren.",
                          items=[(_label(x0, 0, "f"), lf.value if lf.kind == "eigentlich" else lf.kind),
                                 (_label(x0, 0, "g"), lg.value if lg.kind == "eigentlich" else lg.kind)]))
    if lf.kind != "eigentlich" or lg.kind != "eigentlich":
        res.summary = "Die Grenzwertsätze setzen endliche Grenzwerte von f und g voraus."
        res.warnings.append("Mindestens einer der Grenzwerte ist nicht endlich; Sätze nicht anwendbar.")
        return res
    a, b = lf.value, lg.value
    res.a, res.b = a, b

    def chk(name, rule, comb, expected):
        direct = _limit_core(comb, x0, "both")
        ok = direct.kind == "eigentlich" and _same(direct.value, expected)
        res.checks[name] = ok
        res.steps.append(_S(name, f"Mit $a={tex(a)}$ und $b={tex(b)}$ ist der Wert nach Satz {tex(expected)}. "
                              "Gegenprobe: Grenzwert der zusammengesetzten Funktion direkt bestimmt.",
                              items=[("Satz", expected),
                                     ("Gegenprobe", direct.value if direct.exists else direct.kind),
                                     ("Übereinstimmung", "ja" if ok else "NEIN")], rule=rule))

    chk("Summe", "Summenregel", ef + eg, a + b)
    chk("Differenz", "Summenregel", ef - eg, a - b)
    chk("Produkt", "Produktregel", ef * eg, a * b)
    if b != 0:
        chk("Quotient", "Quotientenregel (b ≠ 0)", ef / eg, a / b)
    else:
        res.steps.append(_S("Quotient", "Wegen $b=0$ ist der Quotientensatz nicht anwendbar "
                              "(Nenner → 0; hier müsste man gesondert untersuchen).",
                              rule="Quotientenregel (b ≠ 0)"))
    # Verkettung: aussen f, innen g
    fb = _value_at(ef, b)
    outer_cont = False
    if fb is not None:
        outer_cont = continuity(ef, b).continuous
    if fb is not None and outer_cont:
        comp = ef.subs(x, eg) if not ef.has(sp.Piecewise) else sp.piecewise_fold(ef.subs(x, eg))
        chk("Verkettung", "Stetige äussere Funktion", comp, fb)
        res.steps[-1].detail = (f"Äussere Funktion f ist bei $b={tex(b)}$ stetig, darum "
                                f"$\\lim f(g(x))=f(\\lim g(x))=f(b)={tex(fb)}$. Gegenprobe direkt.")
    else:
        res.steps.append(_S("Verkettung", "Die äussere Funktion f ist bei b nicht stetig (oder nicht definiert); "
                              "der Satz $\\lim f(g(x))=f(b)$ ist dann nicht anwendbar.",
                              rule="Stetige äussere Funktion"))
    res.summary = (f"a = {fmt(a)}, b = {fmt(b)}; alle Gegenproben stimmen." if res.all_ok
                   else f"a = {fmt(a)}, b = {fmt(b)}; Gegenprobe fehlgeschlagen oder Satz nicht anwendbar.")
    res.highlights = [("a = lim f", a), ("b = lim g", b)] + \
        [(k, "stimmt" if v else "stimmt nicht") for k, v in res.checks.items()]
    return res


# ==========================================================================
# Stetigkeit
# ==========================================================================

class ContinuityResult(Result):
    """
    Stetigkeit an einer Stelle (continuity) bzw. der ganzen Funktion (is_continuous).

    `continuous` (bool), `kind`: 'stetig' | 'Sprungstelle' | 'hebbare Lücke' |
    'Polstelle' | 'Oszillation' | 'einseitig definiert'; bei ganzen Funktionen
    zusätzlich 'stetig auf dem Definitionsbereich' bzw. 'unstetig' mit
    `points`, `discontinuities`, `gaps`.
    """

    def __init__(self, f, expr, x0=None):
        super().__init__()
        self.f, self.expr, self.x0 = f, expr, x0
        self.continuous = False
        self.kind = ""
        self.f_x0 = self.left = self.right = self.limit = self.jump = None
        self.points: list[ContinuityResult] = []
        self.discontinuities: list = []
        self.gaps: list = []
        self.domain = None

    def figure(self, **kw):
        from . import plots
        if self.x0 is None:
            pts = [p.x0 for p in self.points]
            lo = float(min(pts)) - 2 if pts else -5
            hi = float(max(pts)) + 2 if pts else 5
            kw.setdefault("xlim", (lo, hi))
            return plots.plot_function(self.expr, **kw)
        return plots.plot_limit(self.expr, self.x0, **kw)


def continuity(f, x0) -> ContinuityResult:
    """
    Stetigkeit an der Stelle x0 nach Definition:
    linksseitiger Grenzwert = rechtsseitiger Grenzwert = f(x0).
    """
    orig = parse(f)
    e = _prep(orig)
    x0 = _x0(x0)
    res = ContinuityResult(f, e, x0)
    res.title = f"Stetigkeit bei x₀ = {fmt(x0)}"
    has_pw = e.has(sp.Piecewise)
    res.steps.append(_S("Definition",
                          "f ist bei $x_0$ stetig, wenn links- und rechtsseitiger Grenzwert und der "
                          "Funktionswert übereinstimmen: "
                          "$\\lim_{x\\to x_0^-}f(x)=\\lim_{x\\to x_0^+}f(x)=f(x_0)$.",
                          items=[("f(x)", _show(orig))] + ([("f(x) abschnittsweise", _show(e))] if e != orig and isinstance(e, sp.Piecewise) else [])))
    fx = _value_at(e, x0)
    res.f_x0 = fx
    if fx is not None:
        res.steps.append(_S("Funktionswert f(x₀)",
                              "Stelle einsetzen" + (" (passenden Ast nach der Bedingung wählen)." if has_pw else "."),
                              items=[(f"f({tex(x0)})", fx)]))
    else:
        res.steps.append(_S("Funktionswert f(x₀)",
                              f"f ist an der Stelle {_T(x0)} nicht definiert (Einsetzen ergibt keine Zahl).",
                              items=[(f"f({tex(x0)})", "nicht definiert")]))
    sl, sr = _side(e, x0, -1), _side(e, x0, 1)
    res.left, res.right = sl.value, sr.value
    merged = (not has_pw and sl.kind == sr.kind == "eigentlich" and _same(sl.value, sr.value))
    if merged:
        st = sr.steps
        st[0].title = f"Links- und rechtsseitiger Grenzwert: {st[0].title.split(': ')[-1]}"
        st[0].note = "Gilt für x → x₀⁻ und x → x₀⁺ gleichermassen."
        res.steps += st
    else:
        res.steps += sl.steps + sr.steps
    res.steps.append(_S("Vergleich", "",
                          items=[(_label(x0, -1), sl.value if sl.value is not None else _kind_text(sl)),
                                 (_label(x0, 1), sr.value if sr.value is not None else _kind_text(sr)),
                                 (f"f({tex(x0)})", fx if fx is not None else "nicht definiert")]))

    both_fin = sl.kind == sr.kind == "eigentlich"
    if sl.kind == "nicht definiert" or sr.kind == "nicht definiert":
        res.kind, res.continuous = "einseitig definiert", False
        why = ("f ist nur auf einer Seite von x₀ definiert; zweiseitige Stetigkeit ist hier nicht "
               "erklärt (höchstens einseitige).")
    elif sl.osc or sr.osc:
        res.kind = "Oszillation"
        why = "Die Funktion schwingt in jeder Umgebung von x₀: kein Grenzwert, also nicht stetig."
    elif "uneigentlich" in (sl.kind, sr.kind):
        res.kind = "Polstelle"
        why = ("Mindestens ein einseitiger Grenzwert ist uneigentlich (±∞): Polstelle, "
               "damit nicht stetig.")
    elif both_fin and not _same(sl.value, sr.value):
        res.kind = "Sprungstelle"
        res.jump = sp.simplify(sr.value - sl.value)
        why = (f"Links ({fmt(sl.value)}) ≠ rechts ({fmt(sr.value)}): Sprungstelle mit "
               f"Sprunghöhe {fmt(res.jump)}; nicht stetig.")
    elif both_fin:
        res.limit = sl.value
        if fx is not None and _same(fx, sl.value):
            res.kind, res.continuous = "stetig", True
            why = (f"links = rechts = f(x₀) = {fmt(fx)}: alle drei Grössen stimmen überein, "
                   "f ist bei x₀ stetig.")
        else:
            res.kind = "hebbare Lücke"
            if fx is None:
                why = (f"Die Grenzwerte stimmen überein (= {fmt(sl.value)}), aber f(x₀) ist nicht definiert: "
                       f"hebbare Lücke. Mit f(x₀) := {fmt(sl.value)} wäre f stetig fortsetzbar.")
            else:
                why = (f"Die Grenzwerte stimmen überein (= {fmt(sl.value)}), aber f(x₀) = {fmt(fx)} "
                       f"weicht ab: hebbare Lücke. Mit f(x₀) := {fmt(sl.value)} wäre f stetig.")
    else:
        res.kind = "Polstelle"
        why = "Der Grenzwert existiert nicht: nicht stetig."
    note = ""
    if fx is None and res.kind != "stetig":
        note = ("Streng genommen liegt x₀ nicht im Definitionsbereich; "
                "'stetig' wird nur an Stellen des Definitionsbereichs gefragt.")
    res.steps.append(_S("Urteil", why, items=[("stetig", res.continuous), ("Art", res.kind)], note=note))
    res.summary = f"f ist bei x₀ = {fmt(x0)} " + ("stetig." if res.continuous else f"nicht stetig ({res.kind}).")
    res.highlights = [("stetig", res.continuous), ("Art", res.kind)]
    if res.jump is not None:
        res.highlights.append(("Sprunghöhe", res.jump))
    return res


def _real_points(expr, domain):
    lo, hi = domain if domain else (-10, 10)
    try:
        s = sp.solveset(expr, x, sp.Reals)
        if not isinstance(s, sp.FiniteSet):
            s = s.intersect(sp.Interval(lo, hi))
        return list(s) if isinstance(s, sp.FiniteSet) else []
    except Exception:
        return []


def is_continuous(f, domain=None) -> ContinuityResult:
    """
    Stetigkeit der ganzen Funktion: Verdachtsstellen (Nennernullstellen,
    Übergänge bei abschnittsweisen Funktionen, Betrag/Signum/Heaviside,
    tan/cot) finden und jede nach Definition prüfen.

    domain: optional (a, b) – nur Verdachtsstellen im Innern werden geprüft.
    Stetig heisst (wie auf der Folie) stetig an allen Stellen des
    Definitionsbereichs; Stellen ausserhalb (Polstellen, Lücken) werden
    als `gaps` gemeldet.
    """
    orig = parse(f)
    e = _prep(orig)
    res = ContinuityResult(f, e, None)
    res.domain = domain
    res.title = "Stetigkeit der Funktion" + (f" auf ({fmt(domain[0])}, {fmt(domain[1])})" if domain else "")
    res.steps.append(_S("Funktion", "Elementare Funktionen (Polynome, sin, cos, exp, ...) sind auf ihrem "
                          "Definitionsbereich stetig; zu prüfen sind nur die Verdachtsstellen.",
                          items=[("f(x)", _show(orig))] + ([("f(x) abschnittsweise", _show(e))] if e != orig and isinstance(e, sp.Piecewise) else [])))
    sus: dict = {}

    def add(p, why):
        p = sp.nsimplify(p) if isinstance(p, float) else p
        for q in list(sus):
            if _same(q, p):
                sus[q].append(why)
                return
        sus[p] = [why]

    for pw_ in e.atoms(sp.Piecewise):
        for _, c in pw_.args:
            if c in (True, sp.true):
                continue
            for rel in c.atoms(sp.Rel):
                for p in _real_points(rel.lhs - rel.rhs, domain):
                    add(p, f"Übergang bei {fmt(rel.lhs)} {rel.rel_op} {fmt(rel.rhs)} (abschnittsweise)")
    for pw_ in orig.atoms(sp.Abs, sp.sign, sp.Heaviside):
        for p in _real_points(pw_.args[0], domain):
            name = {sp.Abs: "Betrag", sp.sign: "Signum", sp.Heaviside: "Heaviside"}[type(pw_)]
            add(p, f"{name}-Funktion wechselt bei {fmt(pw_.args[0])} = 0")
    for pw_ in e.atoms(sp.Pow):
        if pw_.exp.is_negative:
            for p in _real_points(pw_.base, domain):
                add(p, f"Nenner {fmt(pw_.base)} = 0")
    for t in e.atoms(sp.tan):
        for p in _real_points(sp.cos(t.args[0]), domain):
            add(p, f"tan: cos({fmt(t.args[0])}) = 0")
    for t in e.atoms(sp.cot):
        for p in _real_points(sp.sin(t.args[0]), domain):
            add(p, f"cot: sin({fmt(t.args[0])}) = 0")
    if domain:
        a, b = _x0(domain[0]), _x0(domain[1])
        sus = {p: w for p, w in sus.items() if bool(a < p) and bool(p < b)}
    pts = sorted(sus, key=lambda p: float(p))
    res.steps.append(_S("Verdachtsstellen", "Stellen, an denen die Funktion unstetig sein könnte: "
                          + (", ".join(_T(p) for p in pts) if pts else "keine."),
                          items=[(f"x = {fmt(p)}", "; ".join(sus[p])) for p in pts]))
    for p in pts:
        cr = continuity(e, p)
        res.points.append(cr)
        fx = cr.f_x0
        res.steps.append(_S(f"Prüfung bei x = {fmt(p)}",
                              "",
                              items=[(_label(p, -1), cr.left if cr.left is not None else "kein endlicher Wert"),
                                     (_label(p, 1), cr.right if cr.right is not None else "kein endlicher Wert"),
                                     (f"f({tex(p)})", fx if fx is not None else "nicht definiert"),
                                     ("Urteil", cr.kind)],
                              note=cr.steps[-1].detail))
        if fx is None:
            res.gaps.append((p, cr.kind))
        elif not cr.continuous:
            res.discontinuities.append((p, cr.kind))
    res.continuous = not res.discontinuities
    if res.discontinuities:
        res.kind = "unstetig"
        why = "Unstetig bei " + ", ".join(f"x = {fmt(p)} ({k})" for p, k in res.discontinuities) + "."
    elif res.gaps:
        res.kind = "stetig auf dem Definitionsbereich"
        why = ("An allen Stellen des Definitionsbereichs stetig; nicht definiert (Lücken/Polstellen) bei "
               + ", ".join(f"x = {fmt(p)} ({k})" for p, k in res.gaps) + ".")
    else:
        res.kind = "stetig"
        why = "Alle Verdachtsstellen sind stetig: f ist überall stetig."
    res.steps.append(_S("Urteil", why, items=[("stetig", res.continuous), ("Art", res.kind)]))
    res.summary = why
    res.highlights = [("stetig", res.continuous), ("Art", res.kind)]
    return res


# ==========================================================================
# Wertetabelle
# ==========================================================================

class TableResult(Result):
    """Wertetabelle: `rows_left`, `rows_right` als Listen (x, f(x)) mit sympy-Zahlen/None."""

    def __init__(self, expr, x0):
        super().__init__()
        self.expr, self.x0 = expr, x0
        self.rows_left: list = []
        self.rows_right: list = []

    def figure(self, **kw):
        from . import plots
        if _is_inf(self.x0):
            kw.setdefault("xlim", (-10, 10))
            return plots.plot_function(self.expr, **kw)
        return plots.plot_limit(self.expr, self.x0, **kw)


def _val_str(v) -> str:
    if v is None:
        return "nicht definiert"
    s = f"{float(v):.10g}"
    return s


def limit_table(f, x0, side: str = "both") -> TableResult:
    """Wertetabelle f(x0 ± 10^-k), k = 1..6 (bei x0 = ±oo: x = ±10^k)."""
    e = _prep(f)
    x0 = _x0(x0)
    side = "both" if _is_inf(x0) else _side_arg(side)
    res = TableResult(e, x0)
    res.title = f"Wertetabelle bei x → {fmt(x0)}"
    ds = [0] if _is_inf(x0) else ([-1, 1] if side == "both" else [1 if side == "+" else -1])
    lim = _limit_core(e, x0, side)
    for d in ds:
        rows = []
        for k in range(1, 7):
            if _is_inf(x0):
                pt = sp.Integer(10) ** k * (1 if x0 == OO else -1)
            else:
                pt = x0 + d * sp.Rational(1, 10 ** k)
            try:
                v = sp.N(e.subs(x, pt), 15)
                v = None if _bad(v) else v
            except Exception:
                v = None
            rows.append((pt, v))
        (res.rows_left if d < 0 else res.rows_right).extend(rows)
        label = ("von links" if d < 0 else "von rechts" if d > 0 else "")
        res.steps.append(_S(f"Annäherung {label}".strip() if label else "Annäherung",
                              "Einsetzen von " + (f"x = {fmt(x0)} {'−' if d<0 else '+'} 10⁻ᵏ"
                                                  if not _is_inf(x0) else "x = ±10ᵏ") + " für k = 1, ..., 6.",
                              items=[(f"f({_val_str(p)})", _val_str(v)) for p, v in rows]))
    res.summary = f"Vermutung: ${_label(x0, {'+':1,'-':-1,'both':0}[side])}$ = " + \
        (fmt(lim.value) if lim.exists else "existiert nicht") + \
        (" (uneigentlich)" if lim.kind == "uneigentlich" else "")
    res.steps.append(_S("Vermutung und Prüfung",
                          "Aus der Tabelle liest man die Tendenz ab; sicher ist erst die Rechnung "
                          "(ein Tabellenwert beweist nichts).",
                          items=[("Grenzwert (Rechnung)", lim.value if lim.exists else "existiert nicht"),
                                 ("links", lim.left if lim.left is not None else lim.left_kind or "-"),
                                 ("rechts", lim.right if lim.right is not None else lim.right_kind or "-")]
                          if side == "both" and not _is_inf(x0)
                          else [("Grenzwert (Rechnung)", lim.value if lim.exists else "existiert nicht")]))
    res.highlights = [("Grenzwert", lim.value if lim.exists else "existiert nicht")]
    return res
