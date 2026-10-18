#!/usr/bin/env python3
"""
Ableitungsregeln (Semesterwoche 3): ein Ableiter, der wie auf dem Papier rechnet.

`diff(f)` gibt nicht einfach das Ergebnis aus, sondern steigt rekursiv durch
den Term und nennt jede Regel beim Namen:

    Konstante, konstanter Summand, konstanter Faktor, Summenregel,
    Potenzregel, Formelsammlung (e^x, ln x, sin x, ...), Produktregel,
    Quotientenregel, Kettenregel, Ableitung der Umkehrfunktion.

Am Ende wird (nur wenn nötig) vereinfacht und das Ergebnis mit SymPy
(`sympy.diff`) kontrolliert.

    diff("x*e^x")            # Produktregel, Formelsammlung
    diff("ln(x^2+1)")        # Kettenregel
    diff(f, 2)               # zweite Ableitung
    diff_at("x^2", 3)        # f'(3)
    rule_table()             # Tabelle von Folie 24
    check_derivative("x^2", "2x")   # eigene Rechnung prüfen
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy as sp

from .core import fmt, num, parse, tex, x
from .render import Result, Step

# Hilfsvariable für äussere Funktionen g(y) der Kettenregel
_Y = sp.Symbol("y", real=True)

# ==========================================================================
# Formatierung (LaTeX-Bausteine)
# ==========================================================================

_TEX_FIX = re.compile(r"\\(sinh|cosh|tanh|cot|arcsin|arccos|arctan)(?![a-zA-Z])")


def _T(e) -> str:
    """
    LaTeX für Terme. arcsin, sinh, cot usw. werden als \\operatorname gesetzt,
    damit auch die Textansicht (render.tex_to_text) sie kennt.
    """
    if isinstance(e, sp.Basic):
        s = sp.latex(e, inv_trig_style="full", mul_symbol="dot", ln_notation=True)
    else:
        s = tex(e)
    return _TEX_FIX.sub(lambda m: r"\operatorname{" + m.group(1) + "}", s)


def _ts(obj):
    """Objekt für ein Step-Item: SymPy-Terme als LaTeX mit arcsin statt sin^-1."""
    if isinstance(obj, sp.Basic):
        t = _T(obj)
        return t if "\\" in t else t + r"\!"
    return obj


def _step(title, detail="", items=None, note="", rule=""):
    return Step(title, detail, [(l, _ts(o)) for l, o in (items or [])], note, rule)


def _d(detail: str) -> str:
    """Detailtext: $...$ ohne Backslash würde in der Textansicht roh erscheinen."""
    return detail + r"\," if "$" in detail and "\\" not in detail else detail


def _display(expr):
    """Schöne Form fürs Papier: e^(-x) im Nenner statt negativem Exponenten."""
    if expr.is_Mul:
        neg = [f for f in expr.args if isinstance(f, sp.exp) and f.args[0].could_extract_minus_sign()]
        others = [f for f in expr.args if f not in neg]
        if neg and not any(f.is_Pow and f.exp.is_negative for f in others):
            top = sp.Mul(*others)
            bottom = sp.Mul(*[sp.exp(-f.args[0]) for f in neg])
            return r"\frac{" + _T(top) + "}{" + _T(bottom) + "}"
    return expr


def _P(e) -> str:
    """(e)' als LaTeX."""
    return r"\left(" + (e if isinstance(e, str) else _T(e)) + r"\right)'"


def _par(e) -> str:
    if isinstance(e, str):
        return e
    s = _T(e)
    if e.is_Add or (e.is_Mul and e.could_extract_minus_sign()) or (e.is_Number and e.is_negative):
        return r"\left(" + s + r"\right)"
    return s


def _M(*es) -> str:
    """Produkt als LaTeX, ohne dass SymPy es zusammenfasst: 1 · e^x + x · e^x."""
    return r" \cdot ".join(_par(e) for e in es)


def _S(terms) -> str:
    """Summe aus (negativ?, LaTeX)-Paaren: a - b + c."""
    out = ""
    for index, (neg, s) in enumerate(terms):
        if index == 0:
            out = ("-" if neg else "") + s
        else:
            out += (" - " if neg else " + ") + s
    return out or "0"


def _eq(unevaluated: str, value) -> str:
    """'unausgewertet = Wert', aber ohne Wiederholung, wenn beides gleich aussieht."""
    v = _T(value)
    return unevaluated if v == unevaluated else f"{unevaluated} = {v}"


def _name(k: int) -> str:
    """f(x), f'(x), f''(x), f'''(x), f^{(4)}(x)."""
    if k <= 3:
        return "f" + "'" * k + "(x)"
    return rf"f^{{({k})}}(x)"


def _pow_tex(base, n: int) -> str:
    return _T(sp.Pow(base, n, evaluate=False))


def _subs_tex(expr, var, replacement) -> str:
    """expr(replacement) als LaTeX, ohne Vereinfachung (e^{ln x} bleibt e^{ln x})."""
    marker = sp.Symbol("QQQ")
    s = _T(expr.subs(var, marker))
    inner = _T(replacement)
    s = s.replace(r"\left(QQQ \right)", r"\left(" + inner + r" \right)")
    s = s.replace("^{QQQ}", "^{" + inner + "}")
    if not (replacement.is_Symbol or replacement.is_Number):
        inner = r"\left(" + inner + r"\right)"
    return s.replace("QQQ", inner)


# ==========================================================================
# Tabellenableitungen
# ==========================================================================

_TABLE = {
    sp.exp: lambda y: sp.exp(y),
    sp.log: lambda y: 1 / y,
    sp.sin: lambda y: sp.cos(y),
    sp.cos: lambda y: -sp.sin(y),
    sp.tan: lambda y: 1 / sp.cos(y) ** 2,
    sp.cot: lambda y: -1 / sp.sin(y) ** 2,
    sp.asin: lambda y: 1 / sp.sqrt(1 - y ** 2),
    sp.acos: lambda y: -1 / sp.sqrt(1 - y ** 2),
    sp.atan: lambda y: 1 / (1 + y ** 2),
    sp.acot: lambda y: -1 / (1 + y ** 2),
    sp.sinh: lambda y: sp.cosh(y),
    sp.cosh: lambda y: sp.sinh(y),
    sp.tanh: lambda y: 1 / sp.cosh(y) ** 2,
    sp.asinh: lambda y: 1 / sp.sqrt(y ** 2 + 1),
    sp.acosh: lambda y: 1 / sp.sqrt(y ** 2 - 1),
    sp.atanh: lambda y: 1 / (1 - y ** 2),
    sp.Abs: lambda y: sp.sign(y),
}

_DOMAIN = {
    sp.log: "x > 0", sp.asin: "-1 < x < 1", sp.acos: "-1 < x < 1",
    sp.Abs: "x ≠ 0", sp.acosh: "x > 1", sp.atanh: "-1 < x < 1",
}


@dataclass
class _Leaf:
    """Eine Ableitung, die in einem Zug geht (Konstante, c·x^n, c·sin x, ...)."""
    value: sp.Expr
    rules: list
    composed: str          # c · (Tabellenwert), unausgewertet
    c: sp.Expr = 1
    u: sp.Expr = 1
    du: sp.Expr = 0
    const: bool = False
    note: str = ""


# ==========================================================================
# Der regelbasierte Ableiter
# ==========================================================================

class _Deriver:
    """Steigt rekursiv durch den Term und schreibt den Rechenweg mit."""

    def __init__(self, var=x):
        self.var = var
        self.steps: list[Step] = []
        self.rules: list[str] = []
        self.warnings: list[str] = []
        self.depth = 0

    # -- Buchführung ------------------------------------------------------

    def use(self, *names):
        for n in names:
            if n and n not in self.rules:
                self.rules.append(n)

    def emit(self, title, detail="", items=None, rule="", note="", ctx=""):
        self.steps.append(_step(ctx + title, _d(detail), items or [], note, rule))

    # -- Blätter ------------------------------------------------------------

    def _core(self, u, var):
        """(Ableitung, Regel) für u ohne konstanten Faktor, sonst None."""
        if not u.has(var):
            return sp.S.Zero, "Konstante"
        if u == var:
            return sp.S.One, "Potenzregel"
        if u.is_Pow:
            b, n = u.as_base_exp()
            if b == var and not n.has(var):
                return n * var ** (n - 1), "Potenzregel"
            if not b.has(var) and n == var:
                return u * sp.log(b), "Formelsammlung"
            return None
        if u.func in _TABLE and len(u.args) == 1 and u.args[0] == var:
            return _TABLE[u.func](var), "Formelsammlung"
        return None

    def leaf(self, e, var=None, in_sum=False):
        var = var or self.var
        e = sp.sympify(e)
        if not e.has(var):
            rule = "Konstanter Summand" if in_sum else "Konstante"
            return _Leaf(sp.S.Zero, [rule], "0", c=e, u=sp.S.One, du=sp.S.Zero, const=True)
        c, u = e.as_independent(var, as_Add=False)
        core = self._core(u, var)
        if core is None:
            return None
        du, rule = core
        rules = ([] if c == 1 else ["Konstanter Faktor"]) + [rule]
        composed = _T(du) if c == 1 else _M(c, du)
        note = ""
        if u.func in _DOMAIN and u.args and u.args[0] == var:
            note = f"gilt für {_DOMAIN[u.func]}"
        return _Leaf(c * du, rules, composed, c=c, u=u, du=du, note=note)

    @staticmethod
    def _rule_formula(lf: _Leaf) -> str:
        parts = []
        for r in lf.rules:
            if r == "Konstanter Faktor":
                parts.append(r"Konstanter Faktor: $(c \cdot f)' = c \cdot f'$")
            elif r == "Konstante":
                parts.append(r"Konstante: $(c)' = 0$")
            elif r == "Konstanter Summand":
                parts.append(r"Konstanter Summand: $(c)' = 0$")
            elif r == "Potenzregel":
                s = r"Potenzregel: $(x^n)' = n \cdot x^{(n-1)}$"
                if lf.u.is_Pow:
                    n = lf.u.as_base_exp()[1]
                    if n.is_Rational and not n.is_Integer and n > 0:
                        s += r" (Wurzel als Potenz, $\sqrt{x} = x^{1/2}$)"
                    elif n.is_negative:
                        s += r" (Bruch als Potenz mit negativem Exponenten)"
                parts.append(s)
            elif r == "Formelsammlung":
                parts.append("Formelsammlung: $" + _P(lf.u) + " = " + _T(lf.du) + "$")
        return "; ".join(parts)

    def _emit_leaf(self, e, lf: _Leaf, ctx):
        self.use(*lf.rules)
        title = " und ".join(lf.rules)
        detail = f"Ableitung von ${_T(e)}$. " + self._rule_formula(lf) + "."
        items = [(_P(e), _eq(lf.composed, lf.value))]
        self.emit(title, detail, items, rule=", ".join(lf.rules), note=lf.note, ctx=ctx)

    # -- Einstieg -----------------------------------------------------------

    def d(self, e, ctx=""):
        self.depth += 1
        try:
            return self._d(e, ctx)
        finally:
            self.depth -= 1

    def _d(self, e, ctx=""):
        e = sp.sympify(e)
        lf = self.leaf(e)
        if lf is not None:
            self._emit_leaf(e, lf, ctx)
            return lf.value
        if e.is_Add:
            return self._sum(e, ctx)
        if e.is_Mul:
            return self._mul(e, ctx)
        if e.is_Pow or (e.func in _TABLE and len(e.args) == 1):
            return self._power_or_function(e, ctx)
        return self._fallback(e, ctx)

    # -- Summen -------------------------------------------------------------

    def _sum(self, e, ctx):
        var = self.var
        terms = e.as_ordered_terms()
        infos = []
        for t in terms:
            neg = bool(t.could_extract_minus_sign())
            a = -t if neg else t
            infos.append((neg, a, self.leaf(a, in_sum=True)))
        all_leaf = all(lf is not None for _, _, lf in infos)

        self.use("Summenregel")
        sum_of_primes = _S([(neg, _P(a)) for neg, a, _ in infos])
        self.emit("Summenregel",
                  r"Summe gliedweise ableiten: $(f \pm g)' = f' \pm g'$. "
                  f"Ableitung von ${_T(e)}$.",
                  [(_P(e), sum_of_primes)], rule="Summenregel", ctx=ctx)

        if all_leaf:
            # wie Folie 9: Faktoren vorziehen, dann Tabellenwerte einsetzen
            if any((not lf.const) and lf.c != 1 for _, _, lf in infos):
                self.use("Konstanter Faktor")
                pulled = _S([(neg, _P(a) if (lf.const or lf.c == 1)
                              else _M(lf.c, _P(lf.u))) for neg, a, lf in infos])
                self.emit("Konstante Faktoren vorziehen",
                          r"$(c \cdot f)' = c \cdot f'$",
                          [(_P(e), pulled)], rule="Konstanter Faktor", ctx=ctx)
            names = []
            for _, _, lf in infos:
                for r in lf.rules:
                    if r != "Konstanter Faktor" and r not in names:
                        names.append(r)
                self.use(*lf.rules)
            inserted = _S([(neg, lf.composed) for neg, a, lf in infos])
            value = sum((-lf.value if neg else lf.value) for neg, a, lf in infos)
            self.emit("Ableitungen einsetzen",
                      "; ".join(self._short_rule(r) for r in names),
                      [(_P(e), _eq(inserted, value))], rule=", ".join(names), ctx=ctx)
            return value

        # gemischt: elementare Summanden zusammen, die anderen einzeln
        leafs = [(i, neg, a, lf) for i, (neg, a, lf) in enumerate(infos) if lf is not None]
        values = {}
        if leafs:
            items = []
            for i, neg, a, lf in leafs:
                self.use(*lf.rules)
                items.append((_P(a), _eq(lf.composed, lf.value)))
                values[i] = lf.value
            names = []
            for _, _, _, lf in leafs:
                for r in lf.rules:
                    if r not in names:
                        names.append(r)
            self.emit("Elementare Summanden", "; ".join(self._short_rule(r) for r in names),
                      items, rule=", ".join(names), ctx=ctx)
        for i, (neg, a, lf) in enumerate(infos):
            if lf is None:
                values[i] = self.d(a, ctx=f"Summand {i + 1}: ")
        total = sum((-values[i] if neg else values[i]) for i, (neg, a, lf) in enumerate(infos))
        joined = _S([(neg, _par(values[i])) for i, (neg, a, lf) in enumerate(infos)])
        self.emit("Summanden zusammenfügen", r"$(f \pm g)' = f' \pm g'$",
                  [(_P(e), _eq(joined, total))], rule="Summenregel", ctx=ctx)
        return total

    @staticmethod
    def _short_rule(r):
        return {"Konstante": r"$(c)' = 0$", "Konstanter Summand": r"$(c)' = 0$ (Summand entfällt)",
                "Konstanter Faktor": r"$(c f)' = c f'$",
                "Potenzregel": r"Potenzregel $(x^n)' = n x^{(n-1)}$",
                "Formelsammlung": "Werte aus der Formelsammlung"}.get(r, r)

    # -- Produkte, Quotienten ---------------------------------------------

    def _mul(self, e, ctx):
        c, rest = e.as_independent(self.var, as_Add=False)
        if c != 1:
            self.use("Konstanter Faktor")
            self.emit("Konstanter Faktor",
                      rf"Konstanten Faktor vorziehen: $(c \cdot f)' = c \cdot f'$ mit $c = {_T(c)}$.",
                      [(_P(e), _M(c, _P(rest)))], rule="Konstanter Faktor", ctx=ctx)
            v = self.d(rest, ctx="Rest: ")
            total = c * v
            self.emit("Konstanten Faktor einsetzen", "",
                      [(_P(e), _eq(_M(c, v), total))], rule="Konstanter Faktor", ctx=ctx)
            return total
        numer, denom = [], []
        for f in rest.as_ordered_factors():
            if f.is_Pow and f.exp.is_number and f.exp.is_negative:
                denom.append(sp.Pow(f.base, -f.exp))
            elif isinstance(f, sp.exp) and f.args[0].could_extract_minus_sign():
                denom.append(sp.exp(-f.args[0]))
            else:
                numer.append(f)
        if denom:
            fq, gq = sp.Mul(*numer), sp.Mul(*denom)
            return self._quotient(fq, gq, e, ctx,
                                  whole_tex=r"\frac{" + _T(fq) + "}{" + _T(gq) + "}")
        return self._product(list(rest.as_ordered_factors()), e, ctx)

    def _block(self, ctx, rule, title, detail, defs, subs, compose, whole_label, extra=()):
        """
        Gemeinsames Muster aller Regeln mit Teilableitungen.

        subs: (schlüssel, beschriftung der Ableitung, Term, Vorsatz)
        compose(vals) -> (unausgewertetes LaTeX, Wert)
        Sind alle Teilableitungen elementar, gibt es einen einzigen Schritt,
        sonst: Regel hinschreiben, Teile ableiten, einsetzen.
        """
        self.use(rule)
        vals, items, nonleaf = {}, list(defs), []
        for key, dlabel, term, pre in subs:
            lf = self.leaf(term)
            if lf is None:
                nonleaf.append((key, dlabel, term, pre))
                continue
            self.use(*lf.rules)
            vals[key] = lf.value
            remark = _eq(lf.composed, lf.value) + r" \quad \text{(" + ", ".join(lf.rules) + ")}"
            items.append((dlabel, remark))
        if not nonleaf:
            uneval, value = compose(vals)
            items.append((whole_label, self._shown(uneval, value)))
            self.emit(title, detail, items, rule=rule, ctx=ctx)
            return value
        self.emit(title, detail, items, rule=rule, ctx=ctx)
        late = []
        for key, dlabel, term, pre in nonleaf:
            raw = self.d(term, ctx=pre)
            vals[key] = _simplify(raw)
            late.append((dlabel, _eq(_T(raw), vals[key])))
        uneval, value = compose(vals)
        self.emit("Einsetzen", "Teilableitungen in die Regel einsetzen.",
                  late + [(whole_label, self._shown(uneval, value))], rule=rule, ctx=ctx)
        return value

    def _shown(self, uneval, value):
        """Ganz aussen zeigt erst der Vereinfachen-Schritt das Endergebnis."""
        if self.depth <= 1 and _simplify(value) != value:
            return uneval
        return _eq(uneval, value)

    def _product(self, factors, whole, ctx, whole_tex=None):
        n = len(factors)
        names = ["f", "g", "h"] if n <= 3 else [f"f_{{{i + 1}}}" for i in range(n)]
        if n == 2:
            formula = r"$(f \cdot g)' = f' \cdot g + f \cdot g'$"
        elif n == 3:
            formula = (r"$(f \cdot g \cdot h)' = f' \cdot g \cdot h + f \cdot g' \cdot h"
                       r" + f \cdot g \cdot h'$")
        else:
            formula = (r"$(f_1 \cdots f_n)' = f_1' f_2 \cdots f_n + f_1 f_2' \cdots f_n"
                       r" + \dots + f_1 \cdots f_{n-1} f_n'$")
        body = whole_tex or _T(whole)
        detail = (f"Produktregel auf ${body}$: {formula}. "
                  "Die Ableitung wandert von Faktor zu Faktor.")
        defs = [(f"{names[i]}(x)", factors[i]) for i in range(n)]
        subs = [(i, f"{names[i]}'(x)", factors[i], f"Faktor {names[i]}: ") for i in range(n)]

        def compose(vals):
            terms, total = [], sp.S.Zero
            for i in range(n):
                row = [vals[i] if j == i else factors[j] for j in range(n)]
                terms.append((False, _M(*row)))
                total += sp.Mul(*row)
            return _S(terms), total

        label = _P(body) if whole_tex else _P(whole)
        return self._block(ctx, "Produktregel", "Produktregel", detail, defs, subs, compose, label)

    def _quotient(self, f, g, whole, ctx, whole_tex=None):
        body = whole_tex or _T(whole)
        detail = (f"Quotientenregel auf ${body}$: "
                  r"$\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}$.")
        defs = [("f(x)", f), ("g(x)", g)]
        subs = [("f", "f'(x)", f, "Zähler f: "), ("g", "g'(x)", g, "Nenner g: ")]

        def compose(v):
            top = _S([(False, _M(v["f"], g)), (True, _M(f, v["g"]))])
            uneval = r"\frac{" + top + "}{" + _pow_tex(g, 2) + "}"
            return uneval, (v["f"] * g - f * v["g"]) / g ** 2

        label = _P(body) if whole_tex else _P(whole)
        return self._block(ctx, "Quotientenregel", "Quotientenregel", detail, defs, subs,
                           compose, label)

    # -- Potenzen und Funktionen mit innerer Funktion -----------------------

    def _power_or_function(self, e, ctx):
        var = self.var
        if e.is_Pow:
            b, n = e.as_base_exp()
            if b.has(var) and n.has(var):
                return self._log_diff(e, b, n, ctx)
            if b.has(var):                           # (innere)^n
                return self._chain(e, _Y ** n, _Y, b, ctx)
            return self._chain(e, b ** _Y, _Y, n, ctx)  # a^(innere)
        return self._chain(e, e.func(_Y), _Y, e.args[0], ctx)

    def _chain(self, whole, outer, y, inner, ctx, whole_tex=None):
        """Kettenregel f(x) = g(h(x)), wie Folie 19."""
        var = self.var
        core = self._core(outer, y)
        gp_steps = None
        if core is not None:
            gp, grule = core
            rem = ""
        else:  # äussere Funktion ist selbst zusammengesetzt (nur bei diff_chain)
            sub = _Deriver(var=y)
            gp = sub.d(outer, ctx="Äussere Ableitung: ")
            gp_steps = sub.steps
            self.use(*sub.rules)
            self.warnings.extend(sub.warnings)
            grule = ""
        self.use("Kettenregel")
        if core is not None:
            self.use(grule)
        body = whole_tex or _T(whole)
        detail = (f"Zusammengesetzte Funktion ${body} = g(h(x))$: äussere Funktion g, innere "
                  r"Funktion h. Kettenregel: $f'(x) = g'(h(x)) \cdot h'(x)$ "
                  "(innere Funktion nachdifferenzieren).")
        gp_note = (_T(gp) + r" \quad \text{(" + grule + ")}") if grule else _T(gp)
        defs = [("f(x)", whole if not whole_tex else whole_tex),
                ("g(y)", outer), ("g'(y)", gp_note), ("h(x)", inner)]
        if gp_steps is not None:
            defs = [d for d in defs if d[0] != "g'(y)"]
            self.emit("Kettenregel", detail, defs, rule="Kettenregel", ctx=ctx)
            self.steps.extend(gp_steps)
            defs = []
            detail = ""
        subs = [("h", "h'(x)", inner, "Innere Ableitung: ")]

        def compose(v):
            outer_at = _subs_tex(gp, y, inner)
            uneval = outer_at + r" \cdot " + _par(v["h"])
            return uneval, gp.subs(y, inner) * v["h"]

        label = _P(body) if whole_tex else _P(whole)
        if gp_steps is not None:
            return self._block(ctx, "Kettenregel", "Kettenregel: einsetzen", "", defs, subs,
                               compose, label)
        return self._block(ctx, "Kettenregel", "Kettenregel", detail, defs, subs, compose, label)

    def _log_diff(self, e, u, v, ctx):
        """u^v mit x in Basis und Exponent: logarithmisch ableiten (nicht Teil der Vorlesung)."""
        self.use("Logarithmisches Ableiten", "Kettenregel")
        self.warnings.append(
            "Basis und Exponent hängen von x ab: logarithmisches Ableiten "
            "(u^v = e^(v·ln u)). Das gehört nicht zum Stoff der Vorlesung; gilt für u > 0.")
        w = sp.Mul(v, sp.log(u), evaluate=False)
        self.emit("Logarithmisches Ableiten",
                  rf"Basis und Exponent enthalten x. Umschreiben: $f = {_T(e)} = \exp(v \cdot \ln u)$ "
                  rf"mit $u = {_T(u)}$, $v = {_T(v)}$. Kettenregel: "
                  r"$f' = f \cdot (v \cdot \ln u)'$ (für $u > 0$).",
                  [("f(x)", e), (r"w(x) = v \cdot \ln u", w)], rule="Logarithmisches Ableiten",
                  ctx=ctx)
        wd = self.d(w, ctx="Exponent w: ")
        total = e * wd
        self.emit("Einsetzen", "", [(_P(e), _eq(_M(e, wd), total))],
                  rule="Kettenregel", ctx=ctx)
        return total

    def _fallback(self, e, ctx):
        val = sp.diff(e, self.var)
        self.use("sp.diff (Fallback)")
        self.warnings.append(
            f"Für ${_T(e)}$ gibt es keine Vorlesungsregel; die Ableitung stammt von SymPy.")
        self.emit("Keine Vorlesungsregel",
                  f"Für ${_T(e)}$ kennt anakit keine Regel; Ableitung von SymPy.",
                  [(_P(e), val)], rule="sp.diff", ctx=ctx)
        return val


def _is_one(e):
    return e == 1


# ==========================================================================
# Vereinfachen und Kontrollieren
# ==========================================================================

def _simplify(expr):
    """Wählt die kürzeste Form; Polynome bleiben ausmultipliziert."""
    expr = sp.sympify(expr)
    if expr.is_polynomial(x):
        return sp.expand(expr)
    cands = [expr]
    for fn in (sp.factor, sp.simplify, sp.together, sp.cancel, sp.expand):
        try:
            cands.append(fn(expr))
        except Exception:
            pass
    return min(cands, key=lambda c: (sp.count_ops(c), len(str(c))))


_POINTS = [-2.3, -1.7, -0.9, -0.35, 0.21, 0.6, 1.3, 1.9, 2.7, 3.4]


def _num_eval(expr, v):
    try:
        val = complex(sp.N(sp.sympify(expr).subs(x, v), 20))
    except Exception:
        return None
    if val != val or abs(val) == float("inf"):
        return None
    return val


def _agree(a, b, tol=1e-7) -> bool:
    ok = 0
    for v in _POINTS:
        va, vb = _num_eval(a, v), _num_eval(b, v)
        if va is None or vb is None:
            continue
        if abs(va - vb) > tol * (1 + abs(va) + abs(vb)):
            return False
        ok += 1
    return ok > 0


def _verify(result, f, n, warnings) -> bool:
    try:
        ref = sp.diff(f, x, n)
    except Exception:
        return True
    try:
        if sp.simplify(result - ref) == 0:
            return True
    except Exception:
        pass
    if _agree(result, ref):
        return True
    warnings.append(f"Kontrolle gegen SymPy fehlgeschlagen: sp.diff liefert {fmt(ref)}, "
                    f"hier kam {fmt(result)} heraus.")
    return False


def _simplify_step(raw, label):
    """Vereinfachen-Schritt, falls nötig; liefert (Wert, Step oder None)."""
    simp = _simplify(raw)
    if simp == raw:
        return raw, None
    return simp, _step("Vereinfachen", "Zusammenfassen, ausklammern, kürzen.",
                      [(label, _display(simp))], rule="")


# ==========================================================================
# Ergebnisobjekte
# ==========================================================================

class DerivativeResult(Result):
    """Ableitung mit Rechenweg: `.expr` (vereinfacht), `.f`, `.order`, `.rules_used`."""

    title = "Ableitung"

    def __init__(self, f, order=1):
        super().__init__()
        self.f = f
        self.order = order
        self.expr = None
        self.rules_used: list[str] = []
        self.derivatives: list = [f]
        self.verified = True
        self.title = "Ableitung" if order == 1 else f"{order}. Ableitung"

    def figure(self, xlim=None, dark=False, **kwargs):
        from ..grafik.plots import plot_derivative
        return plot_derivative(self.f, xlim=xlim, dark=dark)

    def __call__(self, value):
        return self.expr.subs(x, num(value))


def diff(f, n: int = 1) -> DerivativeResult:
    """
    f nach den Vorlesungsregeln ableiten, n-mal.

        diff("3x^2 - 5x + 7").expr        # 6*x - 5
        diff("ln(x^2+1)").rules_used      # ['Kettenregel', 'Formelsammlung', ...]
        diff(f, 2)                        # f''
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("Die Ordnung n muss eine ganze Zahl ≥ 1 sein.")
    f0 = parse(f)
    res = DerivativeResult(f0, n)
    cur = f0
    for k in range(1, n + 1):
        D = _Deriver()
        if n > 1:
            res.steps.append(_step(f"Die {k}. Ableitung bilden",
                                  f"Ableiten von {_name(k - 1)}.", [(_name(k - 1), cur)]))
        raw = D.d(cur)
        res.steps.extend(D.steps)
        for r in D.rules:
            if r not in res.rules_used:
                res.rules_used.append(r)
        res.warnings.extend(w for w in D.warnings if w not in res.warnings)
        simp, step = _simplify_step(raw, _name(k))
        if step is not None:
            res.steps.append(step)
        elif n > 1:
            res.steps.append(_step(f"Ergebnis {k}. Ableitung", "", [(_name(k), simp)]))
        cur = simp
        res.derivatives.append(cur)
    res.expr = cur
    res.verified = _verify(cur, f0, n, res.warnings)
    res.summary = "f(x) = " + fmt(f0) + ". Regeln: " + ", ".join(res.rules_used) + "."
    res.highlights = [(_name(n), cur)]
    return res


derive = diff


class DerivativeAtResult(DerivativeResult):
    title = "Ableitung an einer Stelle"

    def __init__(self, f, x0):
        super().__init__(f, 1)
        self.x0 = x0
        self.value = None


def diff_at(f, x0) -> DerivativeAtResult:
    """f'(x0): zuerst ableiten, dann die Stelle einsetzen."""
    f0 = parse(f)
    point = num(x0) if not isinstance(x0, sp.Basic) else x0
    base = diff(f0)
    res = DerivativeAtResult(f0, point)
    res.steps = list(base.steps)
    res.rules_used = list(base.rules_used)
    res.warnings = list(base.warnings)
    res.expr = base.expr
    res.derivatives = base.derivatives
    res.verified = base.verified
    value = base.expr.subs(x, point)
    res.steps.append(_step("Stelle einsetzen", _d(f"$x_0 = {_T(point)}$ in $f'(x)$ einsetzen."),
                          [(rf"f'({_T(point)})", _eq(_subs_tex(base.expr, x, point), value))]))
    simple = sp.simplify(value)
    if simple != value and not value.has(sp.zoo, sp.nan):
        res.steps.append(_step("Ausrechnen", "", [(rf"f'({_T(point)})", simple)]))
        value = simple
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        res.warnings.append("f' ist an dieser Stelle nicht definiert (Nenner 0 oder Wurzel/"
                            "Logarithmus ausserhalb des Definitionsbereichs).")
    res.value = value
    res.summary = f"f(x) = {fmt(f0)}, Stelle x₀ = {fmt(point)}."
    res.highlights = [(rf"f'({_T(point)})", value)]
    res.title = "Ableitung an einer Stelle"
    return res


# ==========================================================================
# Die einzelnen Regeln an konkreten Beispielen
# ==========================================================================

# (Name, Newton, Leibniz) wie auf Folie 24
_ROWS = [
    ("Konstanter Summand",
     r"(f(x) + c)' = f'(x)",
     r"\frac{\mathrm{d}(f+c)}{\mathrm{d}x} = \frac{\mathrm{d}f}{\mathrm{d}x}"),
    ("Konstanter Faktor",
     r"(c \cdot f(x))' = c \cdot f'(x)",
     r"\frac{\mathrm{d}(c \cdot f)}{\mathrm{d}x} = c \cdot \frac{\mathrm{d}f}{\mathrm{d}x}"),
    ("Summenregel",
     r"(f(x) \pm g(x))' = f'(x) \pm g'(x)",
     r"\frac{\mathrm{d}(f+g)}{\mathrm{d}x} = \frac{\mathrm{d}f}{\mathrm{d}x} + \frac{\mathrm{d}g}{\mathrm{d}x}"),
    ("Produktregel",
     r"(f(x) \cdot g(x))' = f'(x) \cdot g(x) + f(x) \cdot g'(x)",
     r"\frac{\mathrm{d}(f \cdot g)}{\mathrm{d}x} = \frac{\mathrm{d}f}{\mathrm{d}x} \cdot g + f \cdot \frac{\mathrm{d}g}{\mathrm{d}x}"),
    ("Quotientenregel",
     r"\left(\frac{f(x)}{g(x)}\right)' = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{g^{2}(x)}",
     r"\frac{\mathrm{d}}{\mathrm{d}x}\left(\frac{f}{g}\right) = \frac{1}{g^{2}} \cdot \left(\frac{\mathrm{d}f}{\mathrm{d}x} \cdot g - f \cdot \frac{\mathrm{d}g}{\mathrm{d}x}\right)"),
    ("Kettenregel",
     r"(f(g(x)))' = f'(g(x)) \cdot g'(x)",
     r"\frac{\mathrm{d}(f \circ g)}{\mathrm{d}x} = \frac{\mathrm{d}f}{\mathrm{d}g} \cdot \frac{\mathrm{d}g}{\mathrm{d}x}"),
    ("Umkehrfunktion",
     r"(f^{-1}(x))' = \frac{1}{f'(f^{-1}(x))}",
     r"\frac{\mathrm{d}f^{-1}}{\mathrm{d}x}\bigg|_{x=x_0} = \left(\frac{\mathrm{d}f}{\mathrm{d}x}\bigg|_{x=f^{-1}(x_0)}\right)^{-1}"),
]

# dieselbe Tabelle als Klartext (für Konsole und HTML ohne MathJax)
_PLAIN = {
    "Konstanter Summand": ("(f(x) + c)′ = f′(x)", "d(f+c)/dx = df/dx"),
    "Konstanter Faktor": ("(c·f(x))′ = c·f′(x)", "d(c·f)/dx = c·df/dx"),
    "Summenregel": ("(f(x) ± g(x))′ = f′(x) ± g′(x)", "d(f+g)/dx = df/dx + dg/dx"),
    "Produktregel": ("(f(x)·g(x))′ = f′(x)·g(x) + f(x)·g′(x)",
                     "d(f·g)/dx = df/dx·g + f·dg/dx"),
    "Quotientenregel": ("(f(x)/g(x))′ = (f′(x)·g(x) − f(x)·g′(x)) / g²(x)",
                        "d/dx (f/g) = 1/g² · (df/dx·g − f·dg/dx)"),
    "Kettenregel": ("(f(g(x)))′ = f′(g(x))·g′(x)", "d(f∘g)/dx = df/dg · dg/dx"),
    "Umkehrfunktion": ("(f⁻¹(x))′ = 1 / f′(f⁻¹(x))",
                       "df⁻¹/dx |x=x₀ = ( df/dx |x=f⁻¹(x₀) )⁻¹"),
}


class RuleTable(Result):
    """Die Regeltabelle von Folie 24. `.rows` = [(Name, Newton-LaTeX, Leibniz-LaTeX), ...]."""

    title = "Ableitungsregeln zusammengefasst"
    summary = "Regel | Newton | Leibniz (Folie 24, Semesterwoche 3)."

    def __init__(self):
        super().__init__()
        self.rows = list(_ROWS)
        self.steps = [_step(name, "", [("Newton", n), ("Leibniz", l)], rule=name)
                      for name, n, l in self.rows]

    def __repr__(self) -> str:
        w0 = max(len(r[0]) for r in self.rows)
        w1 = max(len(_PLAIN[r[0]][0]) for r in self.rows)
        lines = [self.title, "",
                 f"{'Regel':<{w0}} | {'Newton':<{w1}} | Leibniz",
                 "-" * w0 + "-+-" + "-" * w1 + "-+-" + "-" * 40]
        for name, _, _ in self.rows:
            newton, leibniz = _PLAIN[name]
            lines.append(f"{name:<{w0}} | {newton:<{w1}} | {leibniz}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        cells = "".join(
            "<tr><td><b>{}</b></td><td>{}</td><td>{}</td></tr>".format(
                html.escape(name), html.escape(_PLAIN[name][0]), html.escape(_PLAIN[name][1]))
            for name, _, _ in self.rows)
        return (self.CSS + "<div class='ak'><p class='ak-h'>" + html.escape(self.title) + "</p>"
                "<table style='border-collapse:collapse;font-size:13px'>"
                "<tr><th style='text-align:left;padding:4px 12px'>Regel</th>"
                "<th style='text-align:left;padding:4px 12px'>Newton</th>"
                "<th style='text-align:left;padding:4px 12px'>Leibniz</th></tr>"
                + cells.replace("<td>", "<td style='padding:4px 12px;border-top:1px solid #8884'>")
                + "</table></div>")

    def latex(self, with_steps: bool = True) -> str:
        body = r" \\[4pt] ".join(f"\\text{{{n}}} & {a} & {b}" for n, a, b in self.rows)
        return (r"\begin{array}{l|l|l}" r"\text{Regel} & \text{Newton} & \text{Leibniz} \\ \hline "
                + body + r"\end{array}")


def rule_table() -> RuleTable:
    """Regel | Newton | Leibniz, genau wie auf Folie 24."""
    return RuleTable()


class RuleResult(Result):
    """Eine einzelne Regel an konkreten Funktionen vorgeführt."""

    def __init__(self, title, whole):
        super().__init__()
        self.title = title
        self.whole = whole
        self.expr = None
        self.rules_used: list[str] = []
        self.verified = True

    def figure(self, xlim=None, dark=False, **kwargs):
        from ..grafik.plots import plot_derivative
        return plot_derivative(self.whole, xlim=xlim, dark=dark)


def _rule_header(name):
    row = next(r for r in _ROWS if r[0] == name)
    return _step("Regel", "Die allgemeine Regel (Newton und Leibniz):",
                [("Newton", row[1]), ("Leibniz", row[2].replace(r"\bigg|", r"\vert"))], rule=name)


def _finish(res, D, raw, whole, header=None):
    if header is not None:
        res.steps.insert(0, header)
    res.steps.extend(D.steps)
    res.rules_used = list(D.rules)
    res.warnings.extend(D.warnings)
    simp, step = _simplify_step(raw, "f'(x)")
    if step is not None:
        res.steps.append(step)
    res.expr = simp
    res.verified = _verify(simp, whole, 1, res.warnings)
    res.highlights = [("f'(x)", simp)]
    return res


def diff_sum(f, g, minus: bool = False) -> RuleResult:
    """Summenregel (f ± g)' = f' ± g' an konkreten f und g."""
    f, g = parse(f), parse(g)
    whole = f - g if minus else f + g
    sgn = "-" if minus else "+"
    res = RuleResult("Summenregel" + (" (Differenz)" if minus else ""), whole)
    res.summary = f"f(x) = {fmt(f)}, g(x) = {fmt(g)}"
    D = _Deriver()
    detail = (rf"Summenregel: $(f {sgn} g)' = f' {sgn} g'$. Die Ableitung einer "
              f"{'Differenz' if minus else 'Summe'} ist die {'Differenz' if minus else 'Summe'} "
              "der Ableitungen.")
    defs = [("f(x)", f), ("g(x)", g)]
    subs = [("f", "f'(x)", f, "Summand f: "), ("g", "g'(x)", g, "Summand g: ")]

    def compose(v):
        return _S([(False, _par(v["f"])), (minus, _par(v["g"]))]), v["f"] - v["g"] if minus \
            else v["f"] + v["g"]

    raw = D._block("", "Summenregel", "Einsetzen in die Summenregel", detail, defs, subs,
                   compose, _P(_M(f) + (" - " if minus else " + ") + _M(g)))
    return _finish(res, D, raw, whole, _rule_header("Summenregel"))


def diff_product(f, g) -> RuleResult:
    """Produktregel (f·g)' = f'·g + f·g' an konkreten f und g."""
    f, g = parse(f), parse(g)
    whole = f * g
    res = RuleResult("Produktregel", whole)
    res.summary = f"f(x) = {fmt(f)}, g(x) = {fmt(g)}"
    D = _Deriver()
    raw = D._product([f, g], whole, "", whole_tex=_M(f, g))
    return _finish(res, D, raw, whole, _rule_header("Produktregel"))


def diff_quotient(f, g) -> RuleResult:
    """Quotientenregel (f/g)' = (f'·g − f·g')/g² an konkreten f und g."""
    f, g = parse(f), parse(g)
    whole = f / g
    res = RuleResult("Quotientenregel", whole)
    res.summary = f"f(x) = {fmt(f)}, g(x) = {fmt(g)}"
    D = _Deriver()
    raw = D._quotient(f, g, whole, "", whole_tex=r"\frac{" + _T(f) + "}{" + _T(g) + "}")
    return _finish(res, D, raw, whole, _rule_header("Quotientenregel"))


def diff_chain(outer, inner) -> RuleResult:
    """
    Kettenregel an f(x) = g(h(x)). `outer` ist g in der Variablen x
    (diff_chain("ln(x)", "x^2+1") heisst f(x) = ln(x²+1)), `inner` ist h.
    """
    g_expr, h_expr = parse(outer), parse(inner)
    whole = g_expr.subs(x, h_expr)
    res = RuleResult("Kettenregel", whole)
    res.summary = f"g(y) = {fmt(g_expr.subs(x, _Y))}, h(x) = {fmt(h_expr)}"
    D = _Deriver()
    raw = D._chain(whole, g_expr.subs(x, _Y), _Y, h_expr, "")
    return _finish(res, D, raw, whole, _rule_header("Kettenregel"))


def diff_constant_factor(f, c) -> RuleResult:
    """(c·f)' = c·f' an konkretem f und konstantem c."""
    f, c = parse(f), parse(c)
    whole = c * f
    res = RuleResult("Konstanter Faktor", whole)
    res.summary = f"c = {fmt(c)}, f(x) = {fmt(f)}"
    D = _Deriver()
    D.use("Konstanter Faktor")
    D.emit("Konstanter Faktor", r"Konstanten Faktor vorziehen: $(c \cdot f)' = c \cdot f'$.",
           [("c", c), ("f(x)", f), (_P(_M(c, f)), _M(c, _P(f)))], rule="Konstanter Faktor")
    v = D.d(f, ctx="Ableitung von f: ")
    raw = c * v
    D.emit("Einsetzen", "", [(_P(_M(c, f)), _eq(_M(c, v), raw))], rule="Konstanter Faktor")
    return _finish(res, D, raw, whole, _rule_header("Konstanter Faktor"))


def diff_constant_summand(f, c) -> RuleResult:
    """(f + c)' = f' an konkretem f und konstantem c."""
    f, c = parse(f), parse(c)
    whole = f + c
    res = RuleResult("Konstanter Summand", whole)
    res.summary = f"f(x) = {fmt(f)}, c = {fmt(c)}"
    D = _Deriver()
    D.use("Konstanter Summand")
    D.emit("Konstanter Summand", r"Ein konstanter Summand entfällt: $(f + c)' = f' + (c)' = f'$.",
           [("f(x)", f), ("c", c), (_P(_T(f) + " + " + _T(c)), _P(f) + " + " + _P(c))],
           rule="Konstanter Summand")
    v = D.d(f, ctx="Ableitung von f: ")
    D.emit("Einsetzen", "", [(_P(_T(f) + " + " + _T(c)), _eq(_T(v) + " + 0", v))],
           rule="Konstanter Summand")
    return _finish(res, D, v, whole, _rule_header("Konstanter Summand"))


_INVERSE = {
    sp.log: lambda: sp.exp(x), sp.exp: lambda: sp.log(x),
    sp.asin: lambda: sp.sin(x), sp.acos: lambda: sp.cos(x), sp.atan: lambda: sp.tan(x),
    sp.sin: lambda: sp.asin(x), sp.cos: lambda: sp.acos(x), sp.tan: lambda: sp.atan(x),
    sp.sinh: lambda: sp.asinh(x), sp.tanh: lambda: sp.atanh(x), sp.asinh: lambda: sp.sinh(x),
}


def _find_inverse(f):
    """Umkehrfunktion g mit g(f(x)) = x suchen: Tabelle, Potenzen, sonst solve()."""
    if f.func in _INVERSE and len(f.args) == 1 and f.args[0] == x:
        return _INVERSE[f.func](), None
    if f.is_Pow:
        b, n = f.as_base_exp()
        if b == x and n.is_number and n != 0:
            return x ** (1 / n), "Umkehrfunktion nur für x > 0 (Zweig der Wurzel)." \
                if n.is_Integer or n.is_Rational else None
        if not b.has(x) and n == x:
            return sp.log(x) / sp.log(b), None
    y = sp.Symbol("y_", real=True)
    try:
        sols = sp.solve(sp.Eq(y, f), x)
    except Exception:
        sols = []
    if not sols:
        raise ValueError(f"Keine Umkehrfunktion von {fmt(f)} gefunden; gib sie mit "
                         "diff_inverse(f, inverse=...) an.")
    g = sols[0].subs(y, x)
    warn = None
    if len(sols) > 1:
        warn = ("f ist nicht injektiv; die Umkehrfunktion gilt nur auf einem Teilbereich "
                f"(gewählt: {fmt(g)}).")
    return g, warn


def diff_inverse(f, inverse=None) -> RuleResult:
    """
    Ableitung der Umkehrfunktion: f = g⁻¹, also f'(x) = 1 / g'(g⁻¹(x)) = 1 / g'(f(x)).

        diff_inverse("ln(x)")      # via e^x
        diff_inverse("sqrt(x)")    # via x^2
        diff_inverse("arcsin(x)")  # via sin x
    """
    f = parse(f)
    warn = None
    if inverse is None:
        g, warn = _find_inverse(f)
    else:
        g = parse(inverse)
    res = RuleResult("Ableitung der Umkehrfunktion", f)
    res.summary = f"f(x) = {fmt(f)} ist die Umkehrfunktion von g(x) = {fmt(g)}"
    if warn:
        res.warnings.append(warn)
    D = _Deriver()
    D.use("Umkehrfunktion")
    gd = None
    lf = D.leaf(g)
    dom = ""
    if f.func in _DOMAIN and f.args and f.args[0] == x:
        dom = f" Gültig für ${_DOMAIN[f.func].replace('<', r'<')}$."
    detail = (r"Aus $g(g^{-1}(x)) = x$ folgt mit der Kettenregel $g'(g^{-1}(x)) \cdot "
              r"(g^{-1})'(x) = 1$, also $f'(x) = (g^{-1})'(x) = \dfrac{1}{g'(g^{-1}(x))}$ "
              r"mit $f = g^{-1}$." + dom)
    defs = [("f(x)", f), (r"g(x) = f^{-1}\!(x)", g), (r"g^{-1}\!(x)", f)]
    subs = [("g", "g'(x)", g, "Ableitung der Umkehrfunktion g: ")]

    def compose(v):
        at = _subs_tex(v["g"], x, f)
        uneval = r"\frac{1}{" + at + "}"
        value = 1 / v["g"].subs(x, f)
        return uneval, value

    raw = D._block("", "Umkehrfunktion", "Umkehrfunktion", detail, defs, subs, compose,
                   r"f'(x) = \left(g^{-1}\right)'(x)")
    # alte Hilfsvariablen vermeiden
    res = _finish(res, D, raw, f, _rule_header("Umkehrfunktion"))
    return res


_RULE_NAMES = {
    "summe": "summe", "summenregel": "summe", "sum": "summe", "differenz": "differenz",
    "produkt": "produkt", "produktregel": "produkt", "product": "produkt",
    "quotient": "quotient", "quotientenregel": "quotient",
    "kette": "kette", "kettenregel": "kette", "chain": "kette",
    "umkehr": "umkehr", "umkehrfunktion": "umkehr", "inverse": "umkehr",
    "faktor": "faktor", "konstanter faktor": "faktor", "konstantenfaktor": "faktor",
    "summand": "summand", "konstanter summand": "summand",
    "potenz": "potenz", "potenzregel": "potenz",
}


def rule(name: str, f=None, g=None) -> Result:
    """
    Eine Regel an Beispielen vorführen: rule("produkt", "x", "e^x").

    Namen: summe, differenz, produkt, quotient, kette (f = äussere, g = innere
    Funktion), umkehr, faktor (f, c), summand (f, c), potenz.
    Ohne f, g wird das Beispiel aus der Vorlesung genommen.
    """
    key = _RULE_NAMES.get(name.strip().lower())
    if key is None:
        raise ValueError(f"Unbekannte Regel '{name}'. Möglich: summe, differenz, produkt, "
                         "quotient, kette, umkehr, faktor, summand, potenz.")
    if key == "summe":
        return diff_sum("x^2" if f is None else f, "sin(x)" if g is None else g)
    if key == "differenz":
        return diff_sum("x^2" if f is None else f, "sin(x)" if g is None else g, minus=True)
    if key == "produkt":
        return diff_product("x" if f is None else f, "e^x" if g is None else g)
    if key == "quotient":
        return diff_quotient("x" if f is None else f, "e^x" if g is None else g)
    if key == "kette":
        return diff_chain("ln(x)" if f is None else f, "x^2+1" if g is None else g)
    if key == "umkehr":
        return diff_inverse("ln(x)" if f is None else f, g)
    if key == "faktor":
        return diff_constant_factor("x^2" if f is None else f, 3 if g is None else g)
    if key == "summand":
        return diff_constant_summand("x^2" if f is None else f, 7 if g is None else g)
    return diff("x^3" if f is None else f)


# ==========================================================================
# Eigene Rechnung prüfen
# ==========================================================================

class CheckResult(Result):
    """Ergebnis von check_derivative: `.correct`, `.expected`, `.claimed`, `.deviations`."""

    title = "Ableitung prüfen"

    def __init__(self, f, claimed, expected):
        super().__init__()
        self.f, self.claimed, self.expected = f, claimed, expected
        self.correct = False
        self.points: list = []
        self.deviations: list = []

    def __bool__(self) -> bool:
        return self.correct


def check_derivative(f, claimed) -> CheckResult:
    """
    Eine selbst gerechnete Ableitung prüfen — symbolisch und an mehreren Stellen.

        check_derivative("x*e^x", "e^x + x*e^x")     # richtig
        check_derivative("x*e^x", "e^x * 1")         # falsch: x·e^x fehlt
    """
    f0, c0 = parse(f), parse(claimed)
    ref = diff(f0).expr
    res = CheckResult(f0, c0, ref)
    res.summary = f"f(x) = {fmt(f0)}, deine Ableitung: {fmt(c0)}"

    # 1. symbolisch
    try:
        delta = sp.simplify(c0 - ref)
    except Exception:
        delta = c0 - ref
    sym_ok = delta == 0
    res.steps.append(_step(
        "Symbolisch vergleichen", "Differenz deiner Ableitung und der richtigen bilden; "
        "sie muss 0 ergeben.",
        [(r"f'(x) \text{ (richtig)}", ref), (r"\text{deine Ableitung}", c0),
         (r"\text{Differenz}", delta)]))

    # 2. numerisch: gegen die richtige Ableitung und gegen den Differenzenquotienten
    eps = 1e-6
    ok_all = True
    for v in _POINTS:
        a, b = _num_eval(c0, v), _num_eval(ref, v)
        fp, fm = _num_eval(f0, v + eps), _num_eval(f0, v - eps)
        if a is None or b is None:
            continue
        quot = (fp - fm) / (2 * eps) if fp is not None and fm is not None else None
        ok = abs(a - b) <= 1e-7 * (1 + abs(a) + abs(b))
        res.points.append((v, b, a, ok))
        if not ok:
            ok_all = False
            res.deviations.append((v, b, a))
    pts = []
    for v, b, a, ok in res.points:
        pts.append(("", rf"\text{{x = {v}:   richtig {_fl(b)}   |   deine {_fl(a)}   "
                    + ("✓" if ok else "✗") + "}"))
    res.steps.append(_step("Numerische Probe",
                          "An mehreren Stellen die Werte vergleichen (nur Stellen, an denen beide "
                          "definiert sind).", pts))
    if not res.points:
        res.warnings.append("Keine Stelle gefunden, an der beide Terme definiert sind.")
    res.correct = bool(sym_ok or (ok_all and res.points and _agree(c0, ref)))

    # 3. Hinweise bei Fehlern
    if not res.correct:
        hints = _hints(f0, c0, ref, delta)
        for hnt in hints:
            res.warnings.append(hnt)
        if res.deviations:
            v, b, a = res.deviations[0]
            res.warnings.append(f"Erste Abweichung bei x = {v}: richtig {_fl(b, False)}, "
                                f"deine Ableitung {_fl(a, False)}.")
    res.highlights = [("Ergebnis", "RICHTIG" if res.correct else "FALSCH"),
                      (r"f'(x) \text{ (richtig)}", ref)]
    return res


def _fl(v, latex=True) -> str:
    if isinstance(v, complex):
        v = v.real if abs(v.imag) < 1e-9 else v
    s = f"{v:.6g}"
    return s


def _hints(f, claimed, ref, delta):
    hints = []
    try:
        if delta.is_number:
            hints.append(f"Die Differenz ist konstant ({fmt(delta)}): vermutlich ein konstanter "
                         "Summand zu viel oder zu wenig.")
            return hints
        if sp.simplify(claimed - f) == 0:
            hints.append("Du hast die Funktion abgeschrieben, nicht abgeleitet.")
            return hints
        ratio = sp.simplify(claimed / ref)
        if ratio != 0 and not ratio.has(sp.zoo, sp.nan):
            if ratio.is_number:
                hints.append(f"Deine Ableitung ist {fmt(ratio)}-mal die richtige: "
                             "ein konstanter Faktor stimmt nicht.")
            else:
                hints.append(f"Verhältnis (deine / richtige) = {fmt(ratio)}: vermutlich fehlt "
                             "oder ist zu viel ein Faktor, z. B. die innere Ableitung "
                             "(Kettenregel) oder ein Term der Produktregel.")
        if f.is_Mul:
            c, rest = f.as_independent(x, as_Add=False)
            parts = rest.as_ordered_factors()
            if len(parts) == 2 and not any(p.is_Pow and p.exp.is_negative for p in parts):
                a, b = parts
                wrong = sp.diff(a, x) * sp.diff(b, x) * c
                if sp.simplify(claimed - wrong) == 0:
                    hints.append("Das ist f'·g'. Die Produktregel lautet aber "
                                 "(f·g)' = f'·g + f·g'.")
    except Exception:
        pass
    return hints


__all__ = ["diff", "derive", "diff_at", "rule", "rule_table", "diff_sum", "diff_product",
           "diff_quotient", "diff_chain", "diff_inverse", "diff_constant_factor",
           "diff_constant_summand", "check_derivative", "DerivativeResult",
           "DerivativeAtResult", "RuleResult", "RuleTable", "CheckResult"]
