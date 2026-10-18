#!/usr/bin/env python3
"""
Grundlagen von anakit: Funktionen einlesen, Zahlen und Terme formatieren.

Gerechnet wird symbolisch mit SymPy, also exakt: 1/3 bleibt 1/3, sqrt(2)
bleibt sqrt(2). Erst `.as_decimal()` schaltet auf Dezimalzahlen um.
"""

from __future__ import annotations

import re
from typing import Callable

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

#: Die Variable, in der alle Funktionen stehen.
x = sp.Symbol("x", real=True)
#: Abstand der zweiten Stelle beim Differenzenquotienten.
h = sp.Symbol("h", real=True)
#: Zweite Stelle für Sekanten.
x0 = sp.Symbol("x0", real=True)
x1 = sp.Symbol("x1", real=True)
x2 = sp.Symbol("x2", real=True)

MINUS = "−"
INF = sp.oo

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,
                                          convert_xor)

_NAMESPACE = {
    "x": x, "h": h, "e": sp.E, "E": sp.E, "pi": sp.pi, "oo": sp.oo,
    "inf": sp.oo, "infty": sp.oo,
    "ln": sp.log, "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "abs": sp.Abs, "Abs": sp.Abs, "sgn": sp.sign, "sign": sp.sign,
    "H": sp.Heaviside, "heaviside": sp.Heaviside,
    "Piecewise": sp.Piecewise,
}


def _parse_piecewise(text: str):
    """'x^2 if x<1 else 2-x' -> Piecewise((x**2, x<1), (2-x, True))."""
    parts = re.split(r"\s+else\s+", text.strip())
    cases = []
    for part in parts:
        m = re.match(r"(.+?)\s+if\s+(.+)", part)
        if m:
            cases.append((parse(m.group(1)), _parse_cond(m.group(2))))
        else:
            cases.append((parse(part), True))
    return sp.Piecewise(*cases)


def _parse_cond(text: str):
    return parse_expr(text.replace("^", "**"), local_dict=_NAMESPACE,
                      transformations=_TRANSFORMS, evaluate=True)


def parse(expr) -> sp.Expr:
    """
    Macht aus Text, Zahl oder SymPy-Ausdruck einen Term in x.

        parse("x^2 - 3x + 1")
        parse("ln(x^2 + 1)")
        parse("x^2 if x<1 else 2-x")      # abschnittsweise
        parse("1/x")

    `^` bedeutet Potenz, `3x` bedeutet 3*x.
    """
    if isinstance(expr, sp.Basic):
        return expr
    if isinstance(expr, (int, float)):
        return sp.nsimplify(expr) if isinstance(expr, float) else sp.Integer(expr)
    if callable(expr) and not isinstance(expr, str):
        return sp.sympify(expr(x))
    text = str(expr).strip().replace("−", "-").replace("·", "*").replace("²", "^2") \
        .replace("³", "^3").replace("√", "sqrt")
    if re.search(r"\s+if\s+", text):
        return _parse_piecewise(text)
    return parse_expr(text, local_dict=_NAMESPACE, transformations=_TRANSFORMS,
                      evaluate=True)


def pw(*cases) -> sp.Piecewise:
    """Abschnittsweise Funktion: pw(("x^2", "x<1"), ("2-x", True))."""
    out = []
    for value, cond in cases:
        out.append((parse(value), _parse_cond(cond) if isinstance(cond, str) else cond))
    return sp.Piecewise(*out)


def num(value) -> sp.Expr:
    """Zahl exakt einlesen: num('1/3'), num(0.5) -> 1/2."""
    if isinstance(value, str):
        return parse(value)
    if isinstance(value, float):
        return sp.nsimplify(value, rational=True)
    return sp.sympify(value)


def to_callable(f) -> Callable[[float], float]:
    """Python-Funktion, die floats nimmt — für Zeichnungen."""
    import numpy as np
    expr = parse(f)
    fn = sp.lambdify(x, expr, modules=["numpy", {"Heaviside": lambda t, h0=0.5: np.heaviside(t, h0)}])
    return fn


def is_number(e) -> bool:
    return isinstance(e, sp.Basic) and e.is_number


# --------------------------------------------------------------------------
# Formatierung
# --------------------------------------------------------------------------

def _dec(e, digits: int):
    if isinstance(e, sp.Basic):
        try:
            if e.is_number and e.is_finite is not False and not e.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                v = sp.N(e, digits + 6)
                return sp.Float(round(float(v), digits)) if v.is_real else v
        except Exception:
            return e
        return e.replace(lambda t: t.is_Rational and not t.is_Integer,
                         lambda t: sp.Float(round(float(t), digits)))
    return e


def tex(e, decimal: bool = False, digits: int = 4) -> str:
    """LaTeX für einen Term, eine Zahl, eine Gleichung oder einen Text."""
    if isinstance(e, str):
        return e if "\\" in e or "$" in e else r"\text{" + e + "}"
    if isinstance(e, (list, tuple)):
        return r",\; ".join(tex(v, decimal, digits) for v in e)
    if isinstance(e, bool):
        return r"\text{" + ("wahr" if e else "falsch") + "}"
    if isinstance(e, (int, float)):
        e = num(e)
    if decimal:
        e = _dec(e, digits)
    return sp.latex(e, inv_trig_style="power", mul_symbol="dot", ln_notation=True)


def fmt(e, decimal: bool = False, digits: int = 4) -> str:
    """Lesbarer Text für die Konsole: x^2 - 3·x + 1, 1/3, sqrt(2)."""
    if isinstance(e, str):
        return e
    if isinstance(e, (int, float)):
        e = num(e)
    if decimal:
        e = _dec(e, digits)
    if isinstance(e, (list, tuple)):
        return ", ".join(fmt(v, decimal, digits) for v in e)
    s = sp.sstr(e)
    s = s.replace("**", "^").replace("*", "·").replace("oo", "∞")
    s = s.replace("log(", "ln(")
    return s
