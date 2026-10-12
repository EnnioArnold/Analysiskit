#!/usr/bin/env python3
"""Tests für anakit.derivative — laufen mit:  python3 tests/tests_derivative.py"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "..", "pkg"), os.path.join(HERE, "..", "..")):
    if os.path.isdir(os.path.join(cand, "anakit")):
        sys.path.insert(0, os.path.abspath(cand))
        break

import sympy as sp

from anakit.core import parse, x
from anakit.derivative import (check_derivative, derive, diff, diff_at, diff_chain,
                               diff_constant_factor, diff_constant_summand, diff_inverse,
                               diff_product, diff_quotient, diff_sum, rule, rule_table,
                               RuleTable)


def same(a, b) -> bool:
    """a und b sind als Terme gleich (exakt, sonst numerisch an Stichproben)."""
    d = sp.simplify(sp.sympify(a) - sp.sympify(b))
    if d == 0:
        return True
    for v in (0.31, 0.77, 1.4, 2.2, -0.45):
        try:
            if abs(complex(sp.N((sp.sympify(a) - sp.sympify(b)).subs(x, v)))) > 1e-9:
                return False
        except Exception:
            pass
    return True


def has_rule(res, name):
    assert name in res.rules_used, f"{name} fehlt in {res.rules_used}"


def clean(res):
    assert not res.warnings, res.warnings
    assert res.verified


# --------------------------------------------------------------------------
# Folien
# --------------------------------------------------------------------------

def test_polynomial():
    r = diff("3x^2 - 5x + 7")
    assert r.expr == 6 * x - 5
    for name in ("Summenregel", "Konstanter Faktor", "Potenzregel", "Konstanter Summand"):
        has_rule(r, name)
    clean(r)
    assert any(s.rule == "Summenregel" for s in r.steps)


def test_product():
    r = diff("x*e^x")
    assert same(r.expr, (1 + x) * sp.exp(x))
    has_rule(r, "Produktregel"); has_rule(r, "Formelsammlung")
    clean(r)


def test_quotient():
    r = diff("x/e^x")
    assert same(r.expr, (1 - x) / sp.exp(x))
    has_rule(r, "Quotientenregel")
    clean(r)
    assert any("Vereinfachen" in s.title for s in r.steps)


def test_chain():
    r = diff("ln(x^2+1)")
    assert same(r.expr, 2 * x / (x ** 2 + 1))
    has_rule(r, "Kettenregel")
    clean(r)
    first = r.steps[0]
    labels = [l for l, _ in first.items]
    assert "g(y)" in labels and "h(x)" in labels and "g'(y)" in labels


def test_inverse():
    assert same(diff_inverse("ln(x)").expr, 1 / x)
    assert same(diff_inverse("sqrt(x)").expr, 1 / (2 * sp.sqrt(x)))
    assert same(diff_inverse("arcsin(x)").expr, 1 / sp.sqrt(1 - x ** 2))
    assert same(diff_inverse("arccos(x)").expr, -1 / sp.sqrt(1 - x ** 2))
    assert same(diff_inverse("arctan(x)").expr, 1 / (1 + x ** 2))
    assert same(diff_inverse("e^x").expr, sp.exp(x))
    assert same(diff_inverse("2x+1").expr, 2)
    r = diff_inverse("ln(x)")
    has_rule(r, "Umkehrfunktion")
    clean(r)
    assert diff_inverse("ln(x)", inverse="e^x").expr == 1 / x


def test_example_function():
    f = "x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27"
    d1 = diff(f)
    assert sp.expand(d1.expr - (5 * x ** 4 - 28 * x ** 3 + 30 * x ** 2 + 36 * x - 27)) == 0
    d2 = diff(f, 2)
    assert sp.expand(d2.expr - (20 * x ** 3 - 84 * x ** 2 + 60 * x + 36)) == 0
    assert d2.order == 2 and len(d2.derivatives) == 3
    assert sp.expand(d2.derivatives[1] - d1.expr) == 0
    clean(d1); clean(d2)


def test_triple_product():
    r = diff("x*sin(x)*e^x")
    assert same(r.expr, sp.diff(x * sp.sin(x) * sp.exp(x), x))
    has_rule(r, "Produktregel")
    clean(r)
    assert len([s for s in r.steps if s.rule == "Produktregel"]) == 1   # ein Schritt, nicht drei


def test_nested_chain():
    r = diff("sin(e^(x^2))")
    assert same(r.expr, 2 * x * sp.exp(x ** 2) * sp.cos(sp.exp(x ** 2)))
    has_rule(r, "Kettenregel")
    assert sum(1 for s in r.steps if s.rule == "Kettenregel" and s.title.endswith("Kettenregel")) >= 2
    clean(r)


def test_quotient_rational():
    r = diff("(x^2+1)/(x-1)")
    assert same(r.expr, (x ** 2 - 2 * x - 1) / (x - 1) ** 2)
    has_rule(r, "Quotientenregel")
    clean(r)


def test_tan_and_table():
    assert same(diff("tan(x)").expr, 1 / sp.cos(x) ** 2)
    assert same(diff("cot(x)").expr, -1 / sp.sin(x) ** 2)
    assert same(diff("arctan(x)").expr, 1 / (1 + x ** 2))
    assert same(diff("arcsin(x)").expr, 1 / sp.sqrt(1 - x ** 2))
    assert same(diff("arccos(x)").expr, -1 / sp.sqrt(1 - x ** 2))
    assert same(diff("sinh(x)").expr, sp.cosh(x))
    assert same(diff("cosh(x)").expr, sp.sinh(x))
    assert same(diff("tanh(x)").expr, 1 - sp.tanh(x) ** 2)
    assert same(diff("2^x").expr, 2 ** x * sp.log(2))
    assert same(diff("e^x").expr, sp.exp(x))
    assert same(diff("ln(x)").expr, 1 / x)
    assert same(diff("abs(x)").expr, sp.sign(x))
    for f in ("tan(x)", "2^x", "sinh(x)", "arcsin(x)"):
        has_rule(diff(f), "Formelsammlung")


def test_powers():
    assert same(diff("sqrt(x)").expr, 1 / (2 * sp.sqrt(x)))
    assert same(diff("1/x").expr, -1 / x ** 2)
    assert same(diff("1/x^3").expr, -3 / x ** 4)
    assert same(diff("x^(2/3)").expr, sp.Rational(2, 3) * x ** sp.Rational(-1, 3))
    assert same(diff("x^pi").expr, sp.pi * x ** (sp.pi - 1))
    assert same(diff("7").expr, 0)
    assert same(diff("5x^3").expr, 15 * x ** 2)
    assert same(diff("-x").expr, -1)
    assert same(diff("sqrt(x^2+1)").expr, x / sp.sqrt(x ** 2 + 1))
    assert same(diff("1/(x^2+1)").expr, -2 * x / (x ** 2 + 1) ** 2)
    assert same(diff("(3x+1)^5").expr, 15 * (3 * x + 1) ** 4)
    assert same(diff("e^(-x^2/2)").expr, -x * sp.exp(-x ** 2 / 2))
    assert same(diff("a*x^2 + b*x + c").expr, 2 * parse("a") * x + parse("b"))


def test_x_to_x():
    r = diff("x^x")
    assert same(r.expr, x ** x * (sp.log(x) + 1))
    assert r.warnings            # nicht Stoff der Vorlesung
    has_rule(r, "Logarithmisches Ableiten")
    r2 = diff("x^sin(x)")
    assert same(r2.expr, sp.diff(x ** sp.sin(x), x))


def test_fallback():
    r = diff("sign(x)*x")           # sign hat keine Vorlesungsregel
    assert same(r.expr, sp.diff(sp.sign(x) * x, x))
    assert r.warnings


def test_derive_alias_and_orders():
    assert derive("x^3").expr == 3 * x ** 2
    assert diff("sin(x)", 4).expr == sp.sin(x)
    assert diff("e^(2x)", 3).expr == 8 * sp.exp(2 * x)
    assert same(diff("x*e^x", 2).expr, (x + 2) * sp.exp(x))
    for bad in (0, -1, 1.5):
        try:
            diff("x", bad)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError erwartet")


def test_diff_at():
    r = diff_at("x^2*e^x", 1)
    assert sp.simplify(r.value - 3 * sp.E) == 0
    assert diff_at("3x^2-5x+7", 2).value == 7
    assert diff_at("ln(x)", "1/2").value == 2
    assert diff_at("sin(x)", "pi").value == -1
    r = diff_at("1/x", 0)
    assert r.warnings


def test_single_rules():
    r = diff_sum("x^2", "sin(x)"); assert same(r.expr, 2 * x + sp.cos(x)); has_rule(r, "Summenregel")
    r = diff_sum("x^2", "sin(x)", minus=True); assert same(r.expr, 2 * x - sp.cos(x))
    r = diff_product("x", "e^x"); assert same(r.expr, (1 + x) * sp.exp(x)); has_rule(r, "Produktregel")
    r = diff_product("x^2+1", "ln(x)"); assert same(r.expr, sp.diff((x ** 2 + 1) * sp.log(x), x))
    r = diff_product("x", "x"); assert same(r.expr, 2 * x)
    r = diff_quotient("x", "e^x"); assert same(r.expr, (1 - x) / sp.exp(x)); has_rule(r, "Quotientenregel")
    r = diff_quotient("sin(x)", "x^2+1"); assert same(r.expr, sp.diff(sp.sin(x) / (x ** 2 + 1), x))
    r = diff_chain("ln(x)", "x^2+1"); assert same(r.expr, 2 * x / (x ** 2 + 1)); has_rule(r, "Kettenregel")
    r = diff_chain("sin(x)", "x^3"); assert same(r.expr, 3 * x ** 2 * sp.cos(x ** 3))
    r = diff_chain("x^5", "3x+1"); assert same(r.expr, 15 * (3 * x + 1) ** 4)
    r = diff_chain("x*sin(x)", "2x"); assert same(r.expr, sp.diff(2 * x * sp.sin(2 * x), x))
    r = diff_constant_factor("sin(x)", 3); assert same(r.expr, 3 * sp.cos(x)); has_rule(r, "Konstanter Faktor")
    r = diff_constant_summand("x^2", 7); assert same(r.expr, 2 * x); has_rule(r, "Konstanter Summand")
    for name, f, g in [("summe", None, None), ("differenz", None, None), ("produkt", "x^2", "cos(x)"),
                       ("quotient", None, None), ("kette", None, None), ("umkehr", "arctan(x)", None),
                       ("faktor", None, None), ("summand", None, None), ("potenz", None, None),
                       ("Kettenregel", "e^x", "sin(x)")]:
        res = rule(name, f, g)
        assert res.steps and res.expr is not None, name
        assert not res.warnings, (name, res.warnings)
    try:
        rule("unsinn")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_rule_table():
    t = rule_table()
    assert isinstance(t, RuleTable)
    names = [r[0] for r in t.rows]
    assert names == ["Konstanter Summand", "Konstanter Faktor", "Summenregel", "Produktregel",
                     "Quotientenregel", "Kettenregel", "Umkehrfunktion"]
    assert all(len(r) == 3 and all(isinstance(c, str) for c in r) for r in t.rows)
    text = repr(t)
    assert "Produktregel" in text and "Leibniz" in text and "Newton" in text
    html = t._repr_html_()
    assert "<table" in html and "Quotientenregel" in html
    assert "\\begin{array}" in t.latex()


def test_check_derivative():
    ok = check_derivative("x*e^x", "e^x + x*e^x")
    assert ok and ok.correct
    ok2 = check_derivative("x*e^x", "(1+x)e^x")
    assert ok2.correct
    bad = check_derivative("x*e^x", "e^x")
    assert not bad and not bad.correct and bad.deviations
    chain = check_derivative("sin(x^2)", "cos(x^2)")           # innere Ableitung vergessen
    assert not chain.correct and chain.warnings
    const = check_derivative("x^2", "2x + 1")
    assert not const.correct and any("konstant" in w for w in const.warnings)
    assert check_derivative("x^2", "2x").correct
    assert check_derivative("ln(x)", "1/x").correct
    assert "FALSCH" in repr(bad) and "RICHTIG" in repr(ok)


def test_rendering():
    for res in (diff("sin(e^(x^2))"), diff("x^x"), diff_at("x^2", 3), diff_inverse("ln(x)"),
                diff_quotient("x", "e^x"), check_derivative("x^2", "2x"), rule_table(),
                diff("x^3", 2)):
        assert isinstance(repr(res), str) and len(repr(res)) > 20
        assert "<div" in res._repr_html_() or "<table" in res._repr_html_()
        assert res.latex()
        assert "$" not in repr(res), repr(res)      # LaTeX-Dollars müssen aufgelöst sein


def test_figure_lazy():
    r = diff("x^2")
    try:
        import anakit.plots  # noqa: F401
    except Exception:
        return
    fig = r.figure()
    assert fig is not None


# --------------------------------------------------------------------------
# Zufallstest gegen sympy.diff
# --------------------------------------------------------------------------

def random_expr(rng, depth=3):
    atoms = ["x", "x", "x", "2", "3", "5", "1/2"]
    funcs = ["sin({})", "cos({})", "e^({})", "ln(({})^2+1)", "sqrt(({})^2+1)", "atan({})",
             "tanh({})", "sinh({})", "cosh({})", "({})^2", "({})^3", "1/(({})^2+1)", "-({})",
             "2*({})", "({})^(1/3+1)"]
    if depth == 0:
        return rng.choice(atoms)
    kind = rng.random()
    if kind < 0.35:
        return rng.choice(funcs).format(random_expr(rng, depth - 1))
    a, b = random_expr(rng, depth - 1), random_expr(rng, depth - 1)
    op = rng.choice(["+", "-", "*", "/"])
    if op == "/":
        return f"({a})/(({b})^2+1)"
    return f"({a}){op}({b})"


def test_random_against_sympy():
    rng = random.Random(20261012)
    count = 0
    fixed = ["x^2*sin(x)", "x*e^x*cos(x)", "sin(x)/x", "x^3/e^x", "ln(x)/x^2", "e^x*sin(x)^2",
             "sin(x)*cos(x)", "x*atan(x)", "(x+1)*(x+2)*(x+3)", "(x^2+1)^3*e^x",
             "x^2*e^(-x)", "ln(sin(x)^2+1)", "tan(x)*x", "sqrt(x)*sin(x)", "e^x/(e^x+1)",
             "(2x+1)/(x^2+x+1)", "cos(x)^3", "sin(2x+1)", "x^2*sqrt(x^2+1)", "1/(e^x+1)",
             "sinh(x)*cosh(x)", "3^x*x", "x^(1/3)*e^x", "ln(x^2+1)^2", "atan(e^x)", "x/ln(x^2+2)"]
    for s in fixed:
        r = diff(s)
        assert same(r.expr, sp.diff(parse(s), x)), s
        assert "sp.diff (Fallback)" not in r.rules_used, s
        assert not r.warnings, (s, r.warnings)
        count += 1
    while count < 90:
        s = random_expr(rng, 3)
        try:
            e = parse(s)
        except Exception:
            continue
        if not e.has(x):
            continue
        r = diff(s)
        ref = sp.diff(e, x)
        assert same(r.expr, ref), (s, r.expr, ref)
        assert "sp.diff (Fallback)" not in r.rules_used, s
        assert not r.warnings, (s, r.warnings)
        assert r.verified
        count += 1
    # zweite Ableitungen
    for s in ["x^3*e^x", "ln(x^2+1)", "sin(x)*cos(x)", "x/(x^2+1)"]:
        assert same(diff(s, 2).expr, sp.diff(parse(s), x, 2)), s
    print(f"  {count} Zufalls-/Festterme gegen sympy.diff geprüft")


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
