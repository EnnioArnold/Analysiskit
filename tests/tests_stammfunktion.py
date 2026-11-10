#!/usr/bin/env python3
"""Tests für anakit.stammfunktion (Semesterwoche 9). Aufruf: python3 tests/tests_stammfunktion.py"""
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

from anakit.stammfunktion import (antiderivative, area_function, fundamental_theorem,
                                  integral_properties)
from anakit.plots import plot_area_function
from anakit.core import parse, x

FAILS = []
R = sp.Rational


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def raises(fn, text):
    try:
        fn()
        check(False, f"{text}: hätte ValueError geben müssen")
    except ValueError:
        pass


# Folie 11: ∫ₐᵇ f = F(b) − F(a)
print("Hauptsatz")
check(fundamental_theorem("x^2", 1, 2).value == R(7, 3), "∫₁² x² = 7/3")
check(fundamental_theorem("x^3-2x", 0, 2).value == 0, "∫₀² (x³ − 2x) = 0")
check(fundamental_theorem("sin(x)", 0, "pi").value == 2, "∫₀^π sin = 2")
check(fundamental_theorem("x^2", 1, 2).F == x ** 3 / 3, "F = x³/3")
check(fundamental_theorem("1/x", 1, 2).value == sp.log(2), "∫₁² 1/x = ln 2")
raises(lambda: fundamental_theorem("1/x", -1, 1), "Pol im Intervall")
raises(lambda: fundamental_theorem("x", 2, 1), "a ≥ b")
raises(lambda: fundamental_theorem("x^x", 0, 1), "keine Stammfunktion")

# Folie 7 und 8: Flächenfunktion, Differenz = Konstante
print("Flächenfunktion")
r = area_function("2x", 0, 1)
check(r.F == x ** 2 and r.F2 == x ** 2 - 1 and r.difference == 1, "F₀ = x², F₁ = x² − 1, Differenz 1")
check(area_function("2x", 3).F == x ** 2 - 9, "untere Grenze 3")
check(all(sp.simplify(sp.diff(area_function(f, 1).F, x) - parse(f)) == 0 for f in ("x^2", "cos(x)")),
      "F' = f")
d = area_function("x^2+1", 1, 4)
check(d.difference == sp.integrate(x ** 2 + 1, (x, 1, 4)), "Differenz = ∫ zwischen den Grenzen")

# Folie 7: Stammfunktion
print("Stammfunktion")
check(antiderivative("x^2").F == x ** 3 / 3, "∫ x² = x³/3")
check(sp.simplify(sp.diff(antiderivative("cos(2x)").F, x) - sp.cos(2 * x)) == 0, "Probe cos(2x)")

# Folie 5: Eigenschaften
print("Eigenschaften")
p = integral_properties("x^2", 0, 3, 1)
v = p.values
check(v["a→b"] == 9 and v["a→c"] == R(1, 3) and v["b→c"] == R(1, 3) - 9, "Werte x² auf [0, 3], c = 1")
check(v["a→c"] == v["a→b"] + v["b→c"], "Additivität")
check(v["a→a"] == 0 and v["b→a"] == -v["a→b"], "gleiche und vertauschte Grenzen")
check(p.sign == 1, "x² ≥ 0")
check(integral_properties("-x^2", 0, 1).sign == -1, "−x² ≤ 0")
s = integral_properties("sin(x)", 0, "2pi")
check(s.sign == 0 and s.values["a→b"] == 0, "sin wechselt das Vorzeichen, Integral 0")
check(integral_properties("x", 0, 2).c == 1, "c = Mitte")
raises(lambda: integral_properties("x", 2, 1), "a ≥ b")

# Ausgabe und Zeichnung
print("Ausgabe")
for res in (p, r, antiderivative("x^2"), fundamental_theorem("x^2", 1, 2)):
    repr(res), res.latex(), res._repr_html_()
    check(res.steps and res.summary, f"{res.title}: Schritte und Zusammenfassung")
    check(res.figure() is not None and res.figure(dark=True) is not None, f"{res.title}: Zeichnung")
check(plot_area_function("x^2", 0) is not None, "plot_area_function ohne b")
check(plot_area_function("sin(x)", 0, "pi", xlim=(-1, 4)) is not None, "plot_area_function mit xlim")
raises(lambda: plot_area_function("x^x", 0, 1), "Zeichnung ohne Stammfunktion")

if FAILS:
    print(f"\n{len(FAILS)} Fehler")
    sys.exit(1)
print("\nAlle Tests bestanden.")
