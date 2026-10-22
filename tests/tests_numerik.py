#!/usr/bin/env python3
"""Tests für anakit.numerik (Semesterwoche 5). Aufruf: python3 tests/tests_numerik.py"""
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

from anakit.numerik import (EPS_MACH, bisection, diff_error_table, newton,
                            numeric_diff, optimal_step)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


# Bisektion (Folie 6; Beispiel x² − 1.5 auf [1, 4] aus newton.ipynb)
print("Bisektion")
r = bisection("x^2 - 1.5", 1, 4, 1e-6)
check(abs(r.root - math.sqrt(1.5)) < 1e-6, f"Bisektion x²−1.5: {r.root}")
check(r.last_interval[1] - r.last_interval[0] <= 1e-6, "Intervall kleiner als tol")
check(r.iterations == 22, f"Schritte nach Länge 3/2^k ≤ 1e-6: {r.iterations}")
check(abs(bisection("x^3 - 5", 1, 2, 1e-9).root - 5 ** (1 / 3)) < 1e-9, "Bisektion x³−5")
check(abs(bisection("cos(x) - x", 0, 1, 1e-10).root - 0.7390851332151607) < 1e-9, "Bisektion cos x = x")
check(bisection("x", -1, 1).root == 0.0, "Mittelpunkt ist Nullstelle: Abbruch")
for bad in (("x^2 + 1", 0, 1), ("x", 2, 1)):
    try:
        bisection(*bad)
        check(False, f"bisection{bad} hätte Fehler geben müssen")
    except ValueError:
        pass
check(bisection("x^2-1.5", 1, 4).figure() is not None, "bisection.figure")
repr(r), r.latex(), r._repr_html_()

# Newton (Folie 9; Notebook: x0 = 3, f'(x) = 2x)
print("Newton")
n = newton("x^2 - 1.5", 3)
check(close(n.root, math.sqrt(1.5), 1e-14), f"Newton x²−1.5: {n.root}")
check([round(row[4], 6) for row in n.rows[:3]] == [1.75, 1.303571, 1.227128],
      "Newton: erste Näherungen 1.75, 1.303571, 1.227128")
check(close(newton("x^3 - 5", 3).root, 5 ** (1 / 3), 1e-14), "Newton x³−5")
check(close(newton("x^3 - 5", 3, numerisch=True).root, 5 ** (1 / 3), 1e-9), "Newton mit numerischer f'")
check(newton("x^2-1.5", 3).converged, "converged")
try:
    newton("x^2 - 1", 0)
    check(False, "f'(x0) = 0 hätte Fehler geben müssen")
except ValueError:
    pass
check(newton("x^2-1.5", 3).figure() is not None, "newton.figure")
repr(n), n.latex(), n._repr_html_()

# Numerisch differenzieren (Folie 11)
print("Differenzen")
check(close(optimal_step("vorwaerts"), math.sqrt(EPS_MACH)), "optimal vorwärts = √ε")
check(close(optimal_step("rueckwaerts"), math.sqrt(EPS_MACH)), "optimal rückwärts = √ε")
check(close(optimal_step("zentral"), EPS_MACH ** (1 / 3)), "optimal zentral = ∛ε")
check(abs(numeric_diff("x^3 - 5", 1, 1e-6).approx - 3) < 1e-9, "zentral x³−5 bei 1 (Notebook)")
check(numeric_diff("x^2", 1, 0.5, "vorwaerts").approx == 2.5, "vorwärts x² h=1/2: (2.25−1)/0.5 = 2.5")
check(numeric_diff("x^2", 1, 0.5, "rueckwaerts").approx == 1.5, "rückwärts x² h=1/2: (1−0.25)/0.5 = 1.5")
check(numeric_diff("x^2", 1, 0.5, "zentral").approx == 2.0, "zentral x² ist exakt 2")
check(numeric_diff("sin(x)", 0, kind="central").kind == "zentral", "Alias central")
# Fehlerordnung: Vorwärts ~h, zentral ~h²
e1, e2 = (numeric_diff("exp(x)", 1, h, "vorwaerts").error for h in (1e-3, 1e-4))
check(8 < e1 / e2 < 12, f"Vorwärts: Fehler ∝ h ({e1 / e2:.2f})")
e1, e2 = (numeric_diff("exp(x)", 1, h, "zentral").error for h in (1e-2, 1e-3))
check(80 < e1 / e2 < 120, f"Zentral: Fehler ∝ h² ({e1 / e2:.1f})")
# Optimale Schrittweite: Minimum der Fehlertabelle bei etwa 10⁻⁸ bzw. 10⁻⁵
t = diff_error_table("exp(x)", 1, "vorwaerts")
check(1e-9 <= t.best_h <= 1e-7, f"Tabelle vorwärts: beste h {t.best_h}")
t = diff_error_table("exp(x)", 1, "zentral")
check(1e-6 <= t.best_h <= 1e-4, f"Tabelle zentral: beste h {t.best_h}")
check(t.best_error < 1e-9, "Tabelle zentral: Fehler klein")
for bad in (dict(h=0), dict(h=-1), dict(kind="quer")):
    try:
        numeric_diff("x", 1, **bad)
        check(False, f"numeric_diff({bad}) hätte Fehler geben müssen")
    except ValueError:
        pass
repr(t), t.latex()

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
