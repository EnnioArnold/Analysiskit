#!/usr/bin/env python3
"""Tests für anakit.integration (Semesterwoche 8). Aufruf: python3 tests/tests_integration.py"""
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

from anakit.integration import definite_integral, faulhaber, integral_by_limit, rectangle_sums
from anakit.plots import plot_riemann

FAILS = []
R = sp.Rational
n = sp.Symbol("n", positive=True, integer=True)


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


# Folie 10: Faulhabersche Formeln
print("Faulhaber")
f = [faulhaber(p).formula for p in range(4)]
check(f[0] == n, "Σ 1 = n")
check(f[1] == n ** 2 / 2 + n / 2, "Σ k = n²/2 + n/2 = n(n+1)/2")
check(f[2] == n ** 3 / 3 + n ** 2 / 2 + n / 6, "Σ k² = n³/3 + n²/2 + n/6")
check(f[3] == n ** 4 / 4 + n ** 3 / 2 + n ** 2 / 4, "Σ k³ = n⁴/4 + n³/2 + n²/4")
check(sp.simplify(faulhaber(2).factored - n * (n + 1) * (2 * n + 1) / 6) == 0, "Σ k² = n(n+1)(2n+1)/6")
check(sp.simplify(faulhaber(3).factored - n ** 2 * (n + 1) ** 2 / 4) == 0, "Σ k³ = n²(n+1)²/4")
check(all(faulhaber(p).formula.subs(n, 7) == sum(k ** p for k in range(1, 8)) for p in range(8)),
      "Formeln gegen direkte Summe für n = 7 (p = 0 … 7)")
check(dict(faulhaber(3, 10).highlights)["n = 10"] == 3025, "1³ + … + 10³ = 3025")
for bad in (-1, 1.5, "2"):
    try:
        faulhaber(bad)
        check(False, f"faulhaber({bad!r}) hätte Fehler geben müssen")
    except ValueError:
        pass
repr(faulhaber(2)), faulhaber(2).latex(), faulhaber(2)._repr_html_()

# Folie 8: ∫ x² dx von 1 bis 2, Ō(n) = 7/3 + 3/(2n) + 1/(6n²)
print("Grenzwert der Obersumme")
r = integral_by_limit("x^2", 1, 2)
check(sp.simplify(r.upper - (R(7, 3) + R(3, 2) / n + R(1, 6) / n ** 2)) == 0, f"Ō(n): {r.upper}")
check(sp.simplify(r.lower - (R(7, 3) - R(3, 2) / n + R(1, 6) / n ** 2)) == 0, f"Ū(n): {r.lower}")
check(r.limit == R(7, 3), "Grenzwert 7/3")
check(sp.limit(r.upper, n, sp.oo) == sp.limit(r.lower, n, sp.oo), "Ober- und Untersumme: gleicher Grenzwert")
check(sp.integrate(sp.Symbol("x") ** 2, (sp.Symbol("x"), 1, 2)) == r.limit, "Kontrolle mit sympy")
check(integral_by_limit("x^3", 0, 2).limit == 4 and integral_by_limit("x", 0, 3).limit == R(9, 2), "x³ auf [0,2], x auf [0,3]")
check(integral_by_limit("-x+3", 0, 3).limit == R(9, 2), "fallende Gerade: Obersumme = linke Randwerte")
check(integral_by_limit("x^2", 1, 2).figure() is not None, "figure")
for bad in (("sin(x)", 0, 1), ("x^2", -1, 1), ("x^2", 2, 1)):
    try:
        integral_by_limit(*bad)
        check(False, f"integral_by_limit{bad} hätte Fehler geben müssen")
    except ValueError:
        pass
repr(r), r.latex(), r._repr_html_()

# Folie 5/7: Rechtecksummen
print("Rechtecksummen")
s = rectangle_sums("x^2", 1, 2, 4)
check(s.upper == R(87, 32) and s.lower == R(63, 32), f"n = 4: Unter {s.lower}, Ober {s.upper}")   # (1.5²+1.75²+2²+... )/4
check(s.error == R(3, 4), "Differenz = Δx·(f(b) − f(a)) = 1/4 · 3")
check(s.lower < R(7, 3) < s.upper, "7/3 liegt zwischen Unter- und Obersumme")
t = rectangle_sums("x^2", 1, 2, [4, 10, 100, 1000])
check([row[0] for row in t.rows] == [4, 10, 100, 1000], "Tabelle für mehrere n")
check(all(a[3] > b[3] for a, b in zip(t.rows, t.rows[1:])), "Differenz wird mit n kleiner")
check(all(row[1] < R(7, 3) < row[2] for row in t.rows), "Schranken für alle n")
check(t.rows[3][2] == R(7, 3) + R(3, 2) / 1000 + R(1, 6) / 1000 ** 2, "n = 1000 entspricht Ō(n) der Folie 8")
d = rectangle_sums("1/x", R(1, 2), 2, 5)                       # Folie 5: fallend, 5 Rechtecke
check(d.upper > d.lower and abs(float(d.upper) - 1.6385) < 5e-4, f"1/x auf [1/2, 2], n = 5: Obersumme {float(d.upper):.4f}")
check(abs(float(d.lower) - 1.1885) < 5e-4, f"Untersumme {float(d.lower):.4f}")
check(s.figure() is not None and plot_riemann("x^2", 1, 2, 8, dark=True) is not None, "plot_riemann")
for bad in (("x^2", -1, 2, 4), ("x^2", 1, 2, 0), ("x^2", 1, 2, 2.5), ("x^2", 2, 1, 3)):
    try:
        rectangle_sums(*bad)
        check(False, f"rectangle_sums{bad} hätte Fehler geben müssen")
    except ValueError:
        pass
repr(s), s.latex(), s._repr_html_()

# Folie 11: Hauptsatz (Ausblick)
print("Hauptsatz")
h = definite_integral("x^2", 1, 2)
check(h.value == R(7, 3) and sp.simplify(h.antiderivative - sp.Symbol("x", real=True) ** 3 / 3) == 0,
      "∫ x² dx = x³/3, F(2) − F(1) = 8/3 − 1/3 = 7/3")
check(definite_integral("sin(x)", 0, sp.pi).value == 2, "∫ sin = 2")
repr(h), h.latex(), h._repr_html_()

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
