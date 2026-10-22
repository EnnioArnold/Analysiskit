#!/usr/bin/env python3
"""Tests für anakit.reihen und anakit.kurve (Semesterwoche 6). Aufruf: python3 tests/tests_reihen.py"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "..", "pkg"), os.path.join(HERE, "..", "..")):
    if os.path.isdir(os.path.join(cand, "anakit")):
        sys.path.insert(0, os.path.abspath(cand))
        break

import matplotlib
matplotlib.use("agg")
import sympy as sp

from anakit.core import x
from anakit.kurve import curve_basics, higher_order_test
from anakit.reihen import maclaurin, taylor

FAILS = []
R = sp.Rational


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def gleich(a, b):
    return sp.simplify(a - b) == 0


# Taylor/MacLaurin (Folien 7/8): a_k = f^(k)(x0)/k!
print("Taylor")
check(gleich(maclaurin("exp(x)", 4).polynomial, 1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24), "e^x")
check(gleich(maclaurin("sin(x)", 5).polynomial, x - x**3 / 6 + x**5 / 120), "sin")
check(gleich(maclaurin("cos(x)", 4).polynomial, 1 - x**2 / 2 + x**4 / 24), "cos")
check(gleich(maclaurin("1/(1-x)", 3).polynomial, 1 + x + x**2 + x**3), "geometrische Reihe")
check(gleich(taylor("ln(x)", 1, 3).polynomial, (x - 1) - (x - 1)**2 / 2 + (x - 1)**3 / 3), "ln um 1")
check(gleich(taylor("sqrt(x)", 4, 2).polynomial, 2 + (x - 4) / 4 - (x - 4)**2 / 64), "sqrt um 4")
check(taylor("1/x", 1, 3).coefficients == [1, -1, 1, -1], "1/x: Koeffizienten (−1)^k")
check(taylor("exp(x)", 0, 3).derivative_values == [1, 1, 1, 1], "Ableitungswerte e^x")
check(gleich(taylor("x^3 - 2x", 1, 3).polynomial, x**3 - 2 * x), "Polynom bleibt sich gleich")
check(taylor("sin(x)", 0, 0).polynomial == 0, "T0 von sin")
t5 = taylor("sin(x)", 0, 5)
check(t5.evaluate(1) == R(101, 120), "T5(1) = 101/120")
check(abs(t5.error_at(1) - abs(math.sin(1) - 101 / 120)) < 1e-12, "Fehler von T5 bei 1")
check(t5.error_at(0.1) < t5.error_at(1), "Fehler wächst mit dem Abstand zu x0")
for bad in (("ln(x)", 0, 2), ("x", 0, -1)):
    try:
        taylor(*bad)
        check(False, f"taylor{bad} hätte Fehler geben müssen")
    except ValueError:
        pass
check(t5.figure() is not None and t5.figure(orders=(1, 3)) is not None, "taylor.figure")
repr(t5), t5.latex(), t5._repr_html_()

# Sonderfälle höhere Ableitungen (Folie 5)
print("Höhere Ableitungen")
h = higher_order_test("x^4", 0)
check(h.k == 4 and h.value_k == 24 and h.kind == "relatives Minimum", "x⁴ bei 0: k = 4, Minimum")
check(higher_order_test("-x^6", 0).kind == "relatives Maximum", "−x⁶: Maximum")
for f in ("x^3", "x^5"):
    h = higher_order_test(f, 0)
    check(h.kind == "Sattelpunkt" and h.inflection, f"{f}: Sattelpunkt, Wendepunkt")
check(higher_order_test("x^3-3x", 1).kind == "relatives Minimum", "x³−3x bei 1: f'' = 6 > 0")
check(higher_order_test("x^3-3x", -1).kind == "relatives Maximum", "x³−3x bei −1: Maximum")
check(higher_order_test("x^3-3x", 0).kind == "kein kritischer Punkt" and higher_order_test("x^3-3x", 0).inflection,
      "x³−3x bei 0: f' ≠ 0, aber Wendepunkt (f'' = 0, f''' = 6)")
check(not higher_order_test("x^4", 0).inflection, "x⁴: kein Wendepunkt")
check(higher_order_test("5", 0).kind == "unbestimmt", "konstante Funktion")
check(higher_order_test("x^4", 0).figure() is not None, "higher_order.figure")
repr(h), h.latex()

# Kurvendiskussion: Basisangaben (Folie 3)
print("Funktionsüberblick")
c = curve_basics("x^2/(x-1)")
check(c.symmetry == "keine" and c.zeros == [0], "x²/(x−1): Symmetrie, Nullstelle 0")
check([p for p, k in c.poles] == [1] and c.poles[0][1] == "Pol mit Vorzeichenwechsel", "Pol bei 1")
check(c.lines == [x + 1], f"schräge Asymptote y = x + 1: {c.lines}")
check(c.range == sp.Union(sp.Interval(-sp.oo, 0), sp.Interval(4, sp.oo)), "Wertebereich (−∞,0] ∪ [4,∞)")
check(curve_basics("x^3-3x").symmetry == "ungerade", "x³−3x ungerade")
check(curve_basics("x^4-2x^2").symmetry == "gerade", "x⁴−2x² gerade")
c = curve_basics("(2x^2+1)/(x^2-3)")
check(c.lines == [2] and len(c.poles) == 2, "waagrechte Asymptote y = 2, zwei Pole ±√3")
check(curve_basics("1/x^2").poles[0][1] == "Pol ohne Vorzeichenwechsel", "1/x²: Pol ohne Vorzeichenwechsel")
check(curve_basics("(x^2-1)/(x-1)").poles[0][1].startswith("Lücke"), "(x²−1)/(x−1): Lücke bei 1")
check(curve_basics("x^2/(x-1)").figure() is not None, "curve_basics.figure")
repr(c), c.latex()

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
