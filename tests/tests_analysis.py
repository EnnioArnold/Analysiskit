#!/usr/bin/env python3
"""Tests für anakit.analysis (Semesterwoche 4). Aufruf: python3 tests/tests_analysis.py"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pkg"))

import numpy as np
import sympy as sp

from anakit.core import parse, x
from anakit.analysis import (monotony, critical_points, curvature, extrema,
                             inflection_points, curve_discussion, MIN, MAX, SAD)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def fl(e):
    return float(sp.N(e, 30))


def grid_check(f, lo, hi, res, name, n=200001):
    """Vergleicht das Ergebnis mit einem dichten Raster (numerisch, unabhängig von SymPy-Logik)."""
    fe = parse(f)
    F = sp.lambdify(x, fe, "numpy")
    D1 = sp.lambdify(x, sp.diff(fe, x), "numpy")
    D2 = sp.lambdify(x, sp.diff(fe, x, 2), "numpy")
    a = lo if math.isfinite(lo) else -6.0
    b = hi if math.isfinite(hi) else 6.0
    eps = 1e-6 * (b - a)
    xs = np.linspace(a + eps, b - eps, n)
    with np.errstate(all="ignore"):
        d1 = np.asarray(D1(xs), float) * np.ones_like(xs)
        d2 = np.asarray(D2(xs), float) * np.ones_like(xs)
        ys = np.asarray(F(xs), float) * np.ones_like(xs)

    def inside(ivs, xv):
        return any(fl(i.start) <= xv <= fl(i.end) for i in ivs)

    # Monotonie / Krümmung: Raster-Vorzeichen gegen Intervalle
    tol = 1e-7
    for xv, v1, v2 in zip(xs[::97], d1[::97], d2[::97]):
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        if v1 > tol:
            check(inside(res.monotony.increasing, xv), f"{name}: f'>0 bei {xv:.3f} nicht in 'increasing'")
        if v1 < -tol:
            check(inside(res.monotony.decreasing, xv), f"{name}: f'<0 bei {xv:.3f} nicht in 'decreasing'")
        if v2 > tol:
            check(inside(res.curvature.left, xv), f"{name}: f''>0 bei {xv:.3f} nicht in Linkskrümmung")
        if v2 < -tol:
            check(inside(res.curvature.right, xv), f"{name}: f''<0 bei {xv:.3f} nicht in Rechtskrümmung")
    # Extrema: lokale Raster-Extrema (strenge, über f-Werte) müssen gemeldet sein
    for i in range(1, len(xs) - 1):
        pass
    mn_grid = [xs[i] for i in range(1, len(xs) - 1)
               if ys[i] < ys[i - 1] and ys[i] <= ys[i + 1] and d1[i - 1] < 0 < d1[i + 1]]
    mx_grid = [xs[i] for i in range(1, len(xs) - 1)
               if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1] and d1[i - 1] > 0 > d1[i + 1]]
    step = (b - a) / n * 3
    for g in mn_grid:
        check(any(abs(fl(p[0]) - g) < step + 1e-9 for p in res.minima), f"{name}: Raster-Minimum bei {g:.4f} fehlt")
    for g in mx_grid:
        check(any(abs(fl(p[0]) - g) < step + 1e-9 for p in res.maxima), f"{name}: Raster-Maximum bei {g:.4f} fehlt")
    # umgekehrt: jedes gemeldete Extremum ist ein Raster-Extremum (Funktionswert in Umgebung)
    for (x0, y0) in res.minima:
        v = fl(x0)
        loc = np.linspace(max(a, v - 1e-3), min(b, v + 1e-3), 201)
        with np.errstate(all="ignore"):
            check(np.nanmin(np.asarray(F(loc), float)) >= fl(y0) - 1e-9, f"{name}: Minimum {v:.4f} nicht lokal minimal")
    for (x0, y0) in res.maxima:
        v = fl(x0)
        loc = np.linspace(max(a, v - 1e-3), min(b, v + 1e-3), 201)
        with np.errstate(all="ignore"):
            check(np.nanmax(np.asarray(F(loc), float)) <= fl(y0) + 1e-9, f"{name}: Maximum {v:.4f} nicht lokal maximal")
    for (x0, y0) in res.saddles:
        v = fl(x0)
        left = float(np.asarray(D1(v - 1e-2))) ; right = float(np.asarray(D1(v + 1e-2)))
        check(left * right > 0, f"{name}: Sattelpunkt {v:.4f} hat Vorzeichenwechsel von f'")
    # Wendestellen: f'' wechselt dort wirklich das Vorzeichen
    for w in res.curvature.inflection_x:
        v = fl(w)
        check(float(D2(v - 1e-3)) * float(D2(v + 1e-3)) < 0, f"{name}: Wendestelle {v:.4f} ohne Vorzeichenwechsel")


def run(name, f, domain=None, minima=None, maxima=None, saddles=None, inflections=None, grid=True):
    print(f"- {name}")
    res = curve_discussion(f, domain)
    repr(res)
    res._repr_html_()
    res.latex()
    res.as_decimal()
    repr(res)
    lo, hi = (-math.inf, math.inf)
    if domain:
        lo = -math.inf if domain[0] is None else fl(parse(domain[0]))
        hi = math.inf if domain[1] is None else fl(parse(domain[1]))
    for label, got, want in (("Minima", res.minima, minima), ("Maxima", res.maxima, maxima),
                             ("Sattelpunkte", res.saddles, saddles)):
        if want is not None:
            check(len(got) == len(want), f"{name}: {label} {got} != {want}")
            for g, w in zip(got, want):
                check(abs(fl(g[0]) - fl(parse(w[0]))) < 1e-9 and abs(fl(g[1]) - fl(parse(w[1]))) < 1e-9,
                      f"{name}: {label} {g} != {w}")
    if inflections is not None:
        got = [fl(p[0]) for p in res.inflection_points]
        want = [fl(parse(w)) for w in inflections]
        check(len(got) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(got, want)),
              f"{name}: Wendestellen {got} != {want}")
    if grid:
        grid_check(f, lo, hi, res, name)
    return res


# --------------------------------------------------------------------------
# Pflichtfall der Folie
# --------------------------------------------------------------------------
F = "x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27"
r = run("Folie SW4", F, minima=[("3/5", "-110592/3125")], maxima=[(-1, 0)], saddles=[(3, 0)],
        inflections=[sp.Rational(3, 5) - 2 * sp.sqrt(6) / 5, sp.Rational(3, 5) + 2 * sp.sqrt(6) / 5, 3])
check(sp.expand(r.d1 - parse("5x^4-28x^3+30x^2+36x-27")) == 0, "f' stimmt nicht")
check(sp.expand(r.d2 - parse("20x^3-84x^2+60x+36")) == 0, "f'' stimmt nicht")
check([str(i) for i in r.increasing] == ["Union(Interval.open(-oo, -1), Interval.open(3/5, oo))"][:0] or len(r.increasing) == 2,
      "increasing sollte aus zwei Intervallen bestehen")
check(len(r.decreasing) == 1 and r.decreasing[0] == sp.Interval.open(-1, sp.Rational(3, 5)), "decreasing falsch")
titles = [s.title for s in r.steps]
order = ["Beispielfunktion", "Erste Ableitung und Monotonie", "Kritische Punkte",
         "Zweite Ableitung und Krümmungsverhalten", "Lokale Extrema"]
idx = [titles.index(t) for t in order]
check(idx == sorted(idx), "Reihenfolge der Folien-Schritte falsch")
ex = extrema(F)
check(len(ex.no_statement) == 1 and ex.no_statement[0][0] == 3, "x0=3 sollte 'keine Aussage' haben")
check(len(ex.supplemented) == 1, "x0=3 sollte über Zusatzprüfung entschieden sein")
check(not ex.undetermined, "undetermined sollte leer sein")

# nicht-rationale kritische Stellen -> CRootOf
G = "x^6/6 + x^2/2 - x"        # f' = x^5 + x - 1 (nicht lösbar durch Radikale)
c = critical_points(G)
check(len(c.critical_x) == 1 and isinstance(c.critical_x[0], sp.CRootOf), "CRootOf erwartet")
x0 = fl(c.critical_x[0])
check(abs(x0 ** 5 + x0 - 1) < 1e-12, "CRootOf-Wert falsch")
check(len(c.minima) == 1 and not c.maxima and not c.saddles, "x^5+x-1: ein Minimum erwartet")
check("CRootOf" in c.steps[1].note, "CRootOf-Hinweis fehlt in note")
repr(c); c.as_decimal(); repr(c); c.latex()
run("CRootOf-Fall", G)
run("5. Grad mit 2 irrationalen Nullstellen von f'", "x^5 - 5x^3 + x + 1")

# --------------------------------------------------------------------------
# Einfache Fälle
# --------------------------------------------------------------------------
run("x^3", "x^3", minima=[], maxima=[], saddles=[(0, 0)], inflections=[0])
run("x^2", "x^2", minima=[(0, 0)], maxima=[], saddles=[], inflections=[])
r = run("-x^4", "-x^4", minima=[], maxima=[(0, 0)], saddles=[], inflections=[])
check(len(r.extrema.no_statement) == 1, "-x^4: f''(0)=0 erwartet")
r = run("x^4", "x^4", minima=[(0, 0)], maxima=[], saddles=[], inflections=[])
check(r.extrema.supplemented and r.extrema.supplemented[0].kind == MIN, "x^4 Zusatzprüfung")
run("x^3-3x", "x^3 - 3x", minima=[(1, -2)], maxima=[(-1, 2)], saddles=[], inflections=[0])
s = run("sin(x) auf (0, 2pi)", "sin(x)", domain=(0, "2*pi"), minima=[("3*pi/2", -1)], maxima=[("pi/2", 1)],
        saddles=[], inflections=["pi"])
e = extrema("sin(x)", (0, "2*pi"))
check(len(e.boundary) == 2 and e.global_max[0] == sp.pi / 2 and e.global_min[0] == 3 * sp.pi / 2,
      "sin: Randpunkte/globale Kandidaten falsch")
run("x*exp(-x)", "x*exp(-x)", minima=[], maxima=[(1, "exp(-1)")], saddles=[], inflections=[2])
run("1/(1+x^2)", "1/(1+x^2)", minima=[], maxima=[(0, 1)], saddles=[],
    inflections=["-sqrt(3)/3", "sqrt(3)/3"])
run("sin(x) unbeschränkt (numerisch)", "sin(x)")
# Polstelle
r = run("1/x (Pol)", "1/x", minima=[], maxima=[], saddles=[], grid=False)
check(not r.increasing and len(r.decreasing) == 2, "1/x: zwei fallende Intervalle")
# Konstante Funktion
r = curve_discussion("5"); repr(r)

# Einzelfunktionen
m = monotony("x^3")
check(len(m.increasing) == 1 and m.increasing[0] == sp.Interval.open(-sp.oo, sp.oo), "x^3 streng wachsend auf R")
check(len(m.table) == 2 and m.critical_x == [0], "x^3 Tabelle")
i = inflection_points("x^4")
check(not i.points and len(i.no_inflection) == 1, "x^4: kein Wendepunkt")
i = inflection_points("x^3-3x")
check(i.points == [(0, 0)], "x^3-3x: Wendepunkt (0,0)")
cv = curvature("x^2")
check(len(cv.left) == 1 and not cv.right and not cv.inflection_x, "x^2 Linkskrümmung")
check(critical_points("x^3").points[0].kind == SAD, "x^3 Sattelpunkt")

# Zeichnung (nur wenn plots.py schon vorhanden)
try:
    from anakit import plots  # noqa
    fig = curve_discussion(F).figure()
    print("  Figure ok:", type(fig).__name__)
except ImportError:
    print("  (plots.py noch nicht vorhanden - figure() übersprungen)")

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
