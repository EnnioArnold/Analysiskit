#!/usr/bin/env python3
"""Tests für anakit.steigung (Semesterwoche 1). Start: python3 tests/tests_steigung.py"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(os.path.abspath(_here)))
sys.path.insert(0, os.path.join(_root, "pkg"))
sys.path.insert(0, _root)

import sympy as sp  # noqa: E402

try:
    import anakit  # noqa: E402,F401
    from anakit.steigung import *  # noqa: E402,F401,F403
except ImportError:
    sys.path.insert(0, os.path.dirname(_here))  # anakit selbst als Paketordner
    raise

from anakit.core import x, h  # noqa: E402

R = sp.Rational


def _render(res):
    """Jede Ausgabe muss ohne Fehler erzeugbar sein."""
    assert repr(res)
    assert res.latex()
    assert res._repr_html_()
    assert res.steps


def test_slope():
    r = slope((-1, 2), (2, 4))          # Folie 5
    assert r.m == R(2, 3) and r.dx == 3 and r.dy == 2
    _render(r)
    v = slope((1, 2), (1, 5))
    assert v.m is None and v.vertical and v.warnings
    _render(v)
    assert slope((0, 1), (4, 1)).m == 0


def test_lines():
    r = line_through((-1, 2), (2, 4))
    assert r.m == R(2, 3) and r.t == R(8, 3)
    assert sp.simplify(r.line - (R(2, 3) * x + R(8, 3))) == 0
    _render(r)
    v = line_through((2, 0), (2, 7))
    assert v.vertical and v.line is None and v.warnings
    _render(v)
    p = line_point_slope((1, 2), 3)
    assert sp.expand(p.line - (3 * x - 1)) == 0 and p.t == -1
    _render(p)
    s = line_slope_intercept(2, -3)
    assert s.line == 2 * x - 3
    _render(s)
    assert line_slope_intercept(0, 4).line == 4


def test_secant():
    r = secant("x^2", -1, 3)             # Folie 9
    assert r.m == 2 and sp.expand(r.line - (2 * x + 3)) == 0
    _render(r)
    r2 = secant_h("x^2", 1, "1/2")       # Folie 10
    assert r2.m == R(5, 2) and r2.x2 == R(3, 2)  # (9/4 - 1)/(1/2); h + 2
    _render(r2)
    # sw2: f(x) = x^3/5 durch -1 und 2
    r3 = secant("x^3/5", -1, 2)
    assert r3.m == R(3, 5) and sp.expand(r3.line - (R(3, 5) * x + R(2, 5))) == 0
    for bad in (lambda: secant("x^2", 1, 1), lambda: secant_h("x^2", 1, 0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError erwartet")


def test_difference_quotient_and_table():
    dq = difference_quotient("x^2", 1)
    assert sp.expand(dq.expr - (h + 2)) == 0 and dq(R(1, 2)) == R(5, 2)
    assert dq.limit == 2
    _render(dq)
    assert sp.simplify(sp.sympify(dq) - (h + 2)) == 0
    t = secant_table("x^2", 1)
    assert [m for _, m in t.rows] == [3, R(5, 2), R(21, 10), R(201, 100), R(2001, 1000)]
    assert t.limit == 2
    _render(t)


def test_tangent_slope():
    for var in ("x", "h"):
        r = tangent_slope("x^2", 2, var)  # Folien 12/13
        assert r.value == 4, var
        _render(r)
    assert tangent_slope("x^3", 1).value == 3
    assert tangent_slope("1/x", 1, "h").value == -1
    assert tangent_slope("sin(x)", 0, "h").value == 1
    assert tangent_slope("sin(x)", 0, "x").value == 1
    assert tangent_slope("sqrt(x)", 4, "h").value == R(1, 4)
    k = tangent_slope("abs(x)", 0, "h")   # Knick
    assert k.value is None and k.warnings and not k.differentiable
    _render(k)


def test_tangent():
    r = tangent("1/x", 1)                 # sw2
    assert r.value == -1 and sp.expand(r.line - (2 - x)) == 0
    _render(r)
    r = tangent("x^2", 2, "h")
    assert sp.expand(r.line - (4 * x - 4)) == 0 and r.point == (2, 4)
    k = tangent("abs(x)", 0)
    assert k.line is None and k.warnings
    _render(k)


def test_derivative_by_definition():
    cases = {"x^3": 3 * x ** 2, "x^5": 5 * x ** 4, "1/x": -1 / x ** 2,
             "sin(x)": sp.cos(x), "cos(x)": -sp.sin(x), "x^2": 2 * x, "x": sp.Integer(1),
             "3x^2 - 2x + 1": 6 * x - 2, "exp(x)": sp.exp(x), "sqrt(x)": 1 / (2 * sp.sqrt(x)),
             "x^7": 7 * x ** 6, "1/x^2": -2 / x ** 3, "x^2*sin(x)": 2 * x * sp.sin(x) + x ** 2 * sp.cos(x)}
    for f, want in cases.items():
        r = derivative_by_definition(f)
        assert sp.simplify(r.expr - want) == 0, (f, r.expr)
        _render(r)
    r = derivative_by_definition("x^5")
    assert r.method == "binomisch"
    assert any("Binomischer Satz" in s.rule for s in r.steps)
    assert not r.warnings
    n = sp.Symbol("n")
    r = derivative_by_definition(x ** n)
    assert sp.simplify(r.expr - n * x ** (n - 1)) == 0
    _render(r)
    r = derivative_by_definition("sin(x)")
    assert any("Einschnürungssatz" in s.rule for s in r.steps)


def test_figures():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from anakit import plots  # noqa: F401
    except Exception:
        return  # plots.py noch nicht vorhanden
    for fig in (secant("x^2", -1, 3).figure(), tangent("x^2", 1).figure(),
                derivative_by_definition("x^3").figure(), line_through((0, 0), (1, 2)).figure(),
                secant_table("x^2", 1).figure()):
        assert fig is not None


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print("FAIL", name, repr(e))
                traceback.print_exc()
    print("alle Tests bestanden" if not fails else f"{fails} Test(s) fehlgeschlagen")
    sys.exit(1 if fails else 0)
