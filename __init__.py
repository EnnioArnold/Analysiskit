#!/usr/bin/env python3
"""
anakit — Grundlagen der Analysis mit nachvollziehbarem Rechenweg.

Gebaut zum Modul ANA-G der HSLU (Semesterwochen 1 bis 4: Sekanten und
Tangenten, Grenzwert und Stetigkeit, Ableitungsregeln, Monotonie und
Extrema). Alles rechnet exakt (SymPy) und zeigt jeden Schritt so, wie er in
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

Jedes Ergebnis zeigt sich im Notebook von selbst. `.as_decimal()` schaltet
auf Dezimalzahlen um, `.latex()` liefert den Rechenweg zum Einfügen in eine
Abgabe, `.figure()` die passende Zeichnung.
"""

from __future__ import annotations

from .core import (INF, MINUS, fmt, h, num, parse, pw, tex, to_callable, x)
from .render import Result, Step, show_tex, tex_to_text
from .steigung import (DerivativeDefResult, DifferenceQuotientResult, LineResult,
                       SecantResult, SecantTableResult, SlopeResult, TangentResult,
                       derivative_by_definition, difference_quotient, line_point_slope,
                       line_slope_intercept, line_through, secant, secant_h,
                       secant_table, slope, tangent, tangent_slope)
from .limits import (ContinuityResult, LimitLawsResult, LimitResult, TableResult,
                     continuity, is_continuous, limit, limit_infinity, limit_laws,
                     limit_table)
from .derivative import (CheckResult, DerivativeAtResult, DerivativeResult,
                         RuleResult, RuleTable, check_derivative, derive, diff,
                         diff_at, diff_chain, diff_constant_factor,
                         diff_constant_summand, diff_inverse, diff_product,
                         diff_quotient, diff_sum, rule, rule_table)
from .analysis import (CriticalResult, CurvatureResult, DiscussionResult,
                       ExtremaResult, InflectionResult, MonotonyResult,
                       critical_points, curvature, curve_discussion, extrema,
                       inflection_points, monotony)
from .plots import (has_plotly, plot_curve, plot_derivative, plot_function,
                    plot_limit, plot_secant, plot_secants_to_tangent,
                    plot_tangent, plotly_function)

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
    # Zeichnen
    "plot_function", "plot_secant", "plot_secants_to_tangent", "plot_tangent",
    "plot_derivative", "plot_curve", "plot_limit", "plotly_function", "has_plotly",
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
        "",
        "Zeichnen:",
        "  plot_function, plot_secant, plot_secants_to_tangent, plot_tangent,",
        "  plot_derivative, plot_curve, plot_limit",
        f"  Plotly für Interaktives: {'vorhanden' if has_plotly() else 'fehlt'}",
        "",
        "Jedes Ergebnis kennt .as_decimal(), .as_exact(), .latex(), .show(), .figure().",
    ]
    return "\n".join(lines)
