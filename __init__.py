#!/usr/bin/env python3
"""
anakit — Grundlagen der Analysis mit nachvollziehbarem Rechenweg.

Gebaut zum Modul ANA-G der HSLU (Semesterwochen 1 bis 9: Sekanten und
Tangenten, Grenzwert und Stetigkeit, Ableitungsregeln, Monotonie und
Extrema, Numerik, höhere Ableitungen und Taylor, Optimierung, bestimmtes
Integral, Stammfunktion und Hauptsatz). Alles rechnet exakt (SymPy) und zeigt jeden Schritt so, wie er in
der Prüfung auf dem Papier stünde.

Kurzübersicht
-------------

    from anakit import *

    secant("x^2", -1, 3)                 # SW1: Sekantensteigung
    tangent("1/x", 1)                    # SW1: Tangente
    derivative_by_definition("x^3")      # SW1: Ableitung über h -> 0

    limit("(x^2-1)/(x-1)", 1)            # SW2: Grenzwert
    continuity("x^2 if x<1 else x-1", 1) # SW2: Stetigkeit

    diff("ln(x^2+1)")                    # SW3: Ableitung mit Regeln
    rule_table()                         # SW3: Regeltabelle Newton/Leibniz

    curve_discussion("x^5-7x^4+10x^3+18x^2-27x-27")   # SW4

    bisection("x^2 - 1.5", 1, 4)         # SW5: Nullstelle durch Halbieren
    newton("x^2 - 1.5", 3)               # SW5: Newton-Verfahren
    numeric_diff("x^3 - 5", 1)           # SW5: numerisch differenzieren

    curve_basics("x^2/(x-1)")            # SW6: Pole, Asymptoten, Symmetrie
    higher_order_test("x^4", 0)          # SW6: erste nichtverschwindende Ableitung
    maclaurin("exp(x)", 4)               # SW6: MacLaurin-Polynom
    curve_sketch("-1/6 x^4 + 2/3 x^3 + 2")   # SW6/7: Kurvendiskussion wie im Unterricht

    optimize("r^2*pi*h", "R^2 = r^2 + (h/2)^2", eliminate="r^2", var="h",
             domain=(0, "2R"))           # SW7: Zylinder in Kugel
    integral_by_limit("x^2", 1, 2)       # SW8: Grenzwert der Obersumme = 7/3
    rectangle_sums("x^2", 1, 2, 4)       # SW8: Unter- und Obersumme
    faulhaber(2)                         # SW8: Σ k² = n³/3 + n²/2 + n/6

    fundamental_theorem("x^2", 1, 2)     # SW9: F' = f, F(b) − F(a) = 7/3
    area_function("2x", 0, 1)            # SW9: Fₐ(x) = x², F₁ − F₂ = const.
    antiderivative("x^2")                # SW9: ∫ f dx = x³/3 + C

Jedes Ergebnis zeigt sich im Notebook von selbst. `.as_decimal()` schaltet
auf Dezimalzahlen um, `.latex()` liefert den Rechenweg zum Einfügen in eine
Abgabe, `.figure()` die passende Zeichnung.
"""

from __future__ import annotations

from .rechnen.core import (INF, MINUS, fmt, h, num, parse, pw, tex, to_callable, x)
from .rechnen.render import Result, Step, show_tex, tex_to_text
from .rechnen.steigung import (DerivativeDefResult, DifferenceQuotientResult, LineResult,
                       SecantResult, SecantTableResult, SlopeResult, TangentResult,
                       derivative_by_definition, difference_quotient, line_point_slope,
                       line_slope_intercept, line_through, secant, secant_h,
                       secant_table, slope, tangent, tangent_slope)
from .rechnen.limits import (ContinuityResult, LimitLawsResult, LimitResult, TableResult,
                     continuity, is_continuous, limit, limit_infinity, limit_laws,
                     limit_table)
from .rechnen.derivative import (CheckResult, DerivativeAtResult, DerivativeResult,
                         RuleResult, RuleTable, check_derivative, derive, diff,
                         diff_at, diff_chain, diff_constant_factor,
                         diff_constant_summand, diff_inverse, diff_product,
                         diff_quotient, diff_sum, rule, rule_table)
from .rechnen.analysis import (CriticalResult, CurvatureResult, DiscussionResult,
                       ExtremaResult, InflectionResult, MonotonyResult,
                       critical_points, curvature, curve_discussion, extrema,
                       inflection_points, monotony)
from .rechnen.numerik import (BisectionResult, DiffErrorTableResult, NewtonResult,
                       NumDiffResult, bisection, diff_error_table, newton,
                       numeric_diff, optimal_step)
from .rechnen.kurve import (CurveBasicsResult, CurveSketchResult, HigherOrderResult,
                    curve_basics, curve_sketch, higher_order_test)
from .rechnen.reihen import TaylorResult, maclaurin, taylor
from .rechnen.optimierung import OptimizeResult, optimize
from .rechnen.integration import (DefiniteIntegralResult, FaulhaberResult,
                    IntegralLimitResult, RectangleSumsResult, definite_integral,
                    faulhaber, integral_by_limit, rectangle_sums)
from .rechnen.stammfunktion import (AntiderivativeResult, AreaFunctionResult,
                    FundamentalTheoremResult, IntegralPropertiesResult, antiderivative,
                    area_function, fundamental_theorem, integral_properties)
from .grafik.plots import (has_plotly, plot_area_function, plot_bisection, plot_curve, plot_derivative,
                    plot_function, plot_limit, plot_newton, plot_riemann, plot_secant,
                    plot_secants_to_tangent, plot_tangent, plot_taylor,
                    plotly_function)

# Die Module liegen in Unterordnern; die alten Namen (anakit.core, anakit.anim,
# anakit.exercises, …) bleiben als Alias erhalten, damit Notebooks weiter laufen.
import importlib as _il, sys as _sys
for _alias, _real in (("core", "rechnen.core"), ("render", "rechnen.render"),
                      ("steigung", "rechnen.steigung"), ("limits", "rechnen.limits"),
                      ("derivative", "rechnen.derivative"), ("analysis", "rechnen.analysis"),
                      ("numerik", "rechnen.numerik"), ("kurve", "rechnen.kurve"),
                      ("reihen", "rechnen.reihen"), ("optimierung", "rechnen.optimierung"),
                      ("integration", "rechnen.integration"),
                      ("stammfunktion", "rechnen.stammfunktion"),
                      ("plots", "grafik.plots"), ("anim", "grafik.anim"), ("design", "grafik.design"),
                      ("exercises", "aufgaben.exercises"), ("uebung", "aufgaben.uebung")):
    _m = _il.import_module(f"{__name__}.{_real}")
    _sys.modules[f"{__name__}.{_alias}"] = _m
    globals()[_alias] = _m
del _il, _sys, _alias, _real, _m

__version__ = "0.1.0"

__all__ = [
    # Grundlagen
    "x", "h", "parse", "pw", "num", "fmt", "tex", "to_callable", "INF", "MINUS",
    "Result", "Step", "show_tex", "tex_to_text",
    # SW1 Sekanten und Tangenten
    "slope", "line_through", "line_point_slope", "line_slope_intercept",
    "secant", "secant_h", "difference_quotient", "secant_table",
    "tangent_slope", "tangent", "derivative_by_definition",
    # SW2 Grenzwert und Stetigkeit
    "limit", "limit_infinity", "limit_laws", "limit_table",
    "continuity", "is_continuous",
    # SW3 Ableitungsregeln
    "diff", "derive", "diff_at", "rule", "rule_table", "check_derivative",
    "diff_sum", "diff_product", "diff_quotient", "diff_chain", "diff_inverse",
    "diff_constant_factor", "diff_constant_summand",
    # SW4 Monotonie und Extrema
    "monotony", "critical_points", "curvature", "extrema", "inflection_points",
    "curve_discussion",
    # SW5 Numerische Mathematik I
    "bisection", "newton", "numeric_diff", "optimal_step", "diff_error_table",
    # SW6 Höhere Ableitungen, Kurvendiskussion, Taylor
    "curve_basics", "curve_sketch", "higher_order_test", "taylor", "maclaurin",
    # SW7 Optimierung
    "optimize",
    # SW8 Bestimmtes Integral
    "rectangle_sums", "integral_by_limit", "faulhaber", "definite_integral",
    # SW9 Unbestimmtes Integral und Hauptsatz
    "integral_properties", "area_function", "antiderivative", "fundamental_theorem",
    # Zeichnen
    "plot_function", "plot_secant", "plot_secants_to_tangent", "plot_tangent",
    "plot_derivative", "plot_curve", "plot_limit", "plot_bisection", "plot_newton",
    "plot_taylor", "plot_riemann", "plot_area_function", "plotly_function", "has_plotly",
]


def info() -> str:
    """Was dieses Paket kann und was es braucht."""
    lines = [
        f"anakit {__version__} — Analysis mit Rechenweg",
        "",
        "SW1 Sekanten und Tangenten:",
        "  slope, line_through, line_point_slope, secant, secant_h,",
        "  secant_table, difference_quotient, tangent_slope, tangent,",
        "  derivative_by_definition",
        "SW2 Grenzwert und Stetigkeit:",
        "  limit, limit_infinity, limit_laws, limit_table, continuity, is_continuous",
        "SW3 Ableitungsregeln:",
        "  diff, diff_at, rule, rule_table, check_derivative,",
        "  diff_sum/product/quotient/chain/inverse",
        "SW4 Monotonie und Extrema:",
        "  monotony, critical_points, curvature, extrema, inflection_points,",
        "  curve_discussion",
        "SW5 Numerische Mathematik I:",
        "  bisection, newton, numeric_diff, optimal_step, diff_error_table",
        "SW6 Höhere Ableitungen und Taylor:",
        "  curve_basics, curve_sketch, higher_order_test, taylor, maclaurin",
        "SW7 Optimierung:",
        "  optimize (Zielfunktion, Nebenbedingung, Z' = 0, Test mit Z'')",
        "SW8 Bestimmtes Integral:",
        "  rectangle_sums, integral_by_limit, faulhaber, definite_integral",
        "SW9 Unbestimmtes Integral und Hauptsatz:",
        "  integral_properties, area_function, antiderivative, fundamental_theorem",
        "",
        "Zeichnen:",
        "  plot_function, plot_secant, plot_secants_to_tangent, plot_tangent,",
        "  plot_derivative, plot_curve, plot_limit,",
        "  plot_bisection, plot_newton, plot_taylor, plot_riemann, plot_area_function",
        f"  Plotly für Interaktives: {'vorhanden' if has_plotly() else 'fehlt'}",
        "",
        "Jedes Ergebnis kennt .as_decimal(), .as_exact(), .latex(), .show(), .figure().",
    ]
    return "\n".join(lines)
