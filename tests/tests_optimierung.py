#!/usr/bin/env python3
"""Tests für anakit.optimierung (Semesterwoche 7). Aufruf: python3 tests/tests_optimierung.py"""
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

from anakit.optimierung import optimize

FAILS = []
R = sp.Rational


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def gleich(a, b):
    return sp.simplify(a - b) == 0


Rr = sp.Symbol("R", real=True, positive=True)
L = sp.Symbol("L", real=True, positive=True)

# Notizen SW7: Zylinder mit maximalem Volumen in Kugel (R² = r² + (h/2)²)
print("Zylinder in Kugel")
z = optimize("r^2*pi*h", "R^2 = r^2 + (h/2)^2", eliminate="r^2", var="h", domain=(0, "2R"))
check(gleich(z.objective, Rr ** 2 * sp.pi * sp.Symbol("h", real=True) - sp.pi * sp.Symbol("h", real=True) ** 3 / 4),
      f"V(h) = R²πh − h³π/4: {z.objective}")
check(gleich(z.best, 2 * Rr / sp.sqrt(3)), f"h = 2R/√3: {z.best}")
h = sp.Symbol("h", real=True)
check(gleich(z.second, -sp.Rational(3, 2) * sp.pi * h), "V''(h) = −3/2·h·π")
c = [r for r in z.critical if r["inside"]]
check(len(z.critical) == 2 and len(c) == 1 and c[0]["kind"] == "Maximum", "negatives h verworfen, Maximum")
check(gleich(c[0]["z2"], -3 / sp.sqrt(3) * sp.pi * Rr), "V''(2R/√3) = −3/√3·πR < 0")
check(gleich(z.others["r^2"], 2 * Rr ** 2 / 3) and gleich(z.others["r"], sp.sqrt(6) * Rr / 3), "r² = 2R²/3")
check(gleich(z.best_value, 4 * sp.sqrt(3) * sp.pi * Rr ** 3 / 9), "V_max = 4√3πR³/9")
check(z.figure() is not None, "figure")
repr(z), z.latex(), z._repr_html_()

# Folien 9/10: Holzstück, A(x) = (4 − x)(4 + 2x − x²) = x³ − 6x² + 4x + 16
print("Holzstück")
x = sp.Symbol("x", real=True)
w = optimize("(4-x)*(4+2x-x^2)", var="x")
check(sp.expand(w.objective - (x ** 3 - 6 * x ** 2 + 4 * x + 16)) == 0, f"Zielfunktion x³−6x²+4x+16: {w.objective}")
check(gleich(w.derivative, 3 * x ** 2 - 12 * x + 4), "A'(x) = 3x² − 12x + 4")
x1, x2 = 2 - 2 * sp.sqrt(6) / 3, 2 + 2 * sp.sqrt(6) / 3
check([gleich(r["x0"], v) for r, v in zip(w.critical, (x1, x2))] == [True, True], "x₁, x₂ = 2 ∓ (2/3)√6")
# Auf der Folie stehen x₂ und B(x₁) als 3.3630; richtig ist 2 + (2/3)√6 = 3.6330 (Zahlendreher)
check(abs(float(x1) - 0.3670) < 5e-5 and abs(float(x2) - 3.6330) < 5e-5, "x₁ ≈ 0.3670, x₂ ≈ 3.6330 (Folie: 3.3630)")
check([r["kind"] for r in w.critical] == ["Maximum", "Minimum"], "x₁ Maximum, x₂ Minimum")
check(gleich(w.critical[0]["z2"], -4 * sp.sqrt(6)) and gleich(w.critical[1]["z2"], 4 * sp.sqrt(6)),
      "A''(x₁) = −4√6, A''(x₂) = 4√6")
B = 4 - w.best
H = 4 + 2 * w.best - w.best ** 2
check(abs(float(B) - 3.6330) < 5e-5, f"B(x₁) = 2 + (2/3)√6 ≈ 3.6330 (auf der Folie steht 3.3630): {float(B)}")
check(gleich(H, R(4, 3) * (1 + sp.sqrt(6))) and abs(float(H) - 4.5993) < 5e-5, "H(x₁) = 4/3 (1 + √6) ≈ 4.5993")
check(gleich(B * H, w.best_value), "A(x₁) = B·H")
# mit Bruchkante nur für 0 ≤ x ≤ 1 (siehe Zeichnung der Folie): Randwert x = 0 vergleichen
w2 = optimize("(4-x)*(4+2x-x^2)", var="x", domain=(0, 1), closed=True)
check(gleich(w2.best, x1) and len(w2.critical) == 2 and not w2.critical[1]["inside"], "Bereich [0, 1]: x₂ verworfen")
check(float(w2.best_value) > 16, "A(x₁) ≈ 16.709 > A(0) = 16")

# Notizen SW7: Zaun an der Wand, L = 2·L₁ + L₂, F = L₁·L₂
print("Zaun")
zz = optimize("L1*L2", "L = 2*L1 + L2", eliminate="L1", var="L2", domain=(0, "L"))
check(gleich(zz.best, L / 2) and gleich(zz.others["L1"], L / 4), "L₂ = L/2, L₁ = L/4")
check(gleich(zz.second, -1) and zz.critical[0]["kind"] == "Maximum", "F'' = −1 < 0: Maximum")
check(gleich(zz.best_value, L ** 2 / 8), "F_max = L²/8")

# weitere Fälle
print("Randfälle")
mn = optimize("x^2 - 4x + 1", var="x", goal="min")
check(mn.best == 2 and mn.best_value == -3, "Minimum von x²−4x+1")
check(optimize("x^2", var="x", domain=(1, 2), goal="max").summary.startswith("Keine Lösung"),
      "offenes Intervall: kein Maximum im Innern")
check(optimize("x^2", var="x", domain=(1, 2), goal="max", closed=True).best == 2, "geschlossen: Rand x = 2")
for bad in (dict(target="r*h", constraint="R = r + h"),                      # eliminate fehlt
            dict(target="x^2", var="y"),                                     # var fehlt im Term
            dict(target="a*b", constraint="a < b", eliminate="a", var="b"),  # keine Gleichung
            dict(target="x^2", goal="mittel"),
            dict(target="x.__class__")):
    try:
        optimize(**bad)
        check(False, f"{bad} hätte Fehler geben müssen")
    except ValueError:
        pass

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
