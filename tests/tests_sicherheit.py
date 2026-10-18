#!/usr/bin/env python3
"""Tests für die Eingabeprüfung in anakit.core.parse: python3 tests/tests_sicherheit.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pkg'))
import sympy as sp
from anakit.core import parse, x

# Gewöhnliche Terme bleiben lesbar
for ok in ("x^2 - 3x + 1", "0.5x", ".5x", "ln(x^2+1)", "sqrt(x)", "e^x", "x^2 if x<1 else 2-x",
           "x if x<0 else 1 if x<2 else 3-x", "x!", "log(x, 2)", "²x²", "√x", "sinx", "2cos(x)"):
    assert isinstance(parse(ok), sp.Basic), ok

# Attribute, Strings, Listen und Unterstriche werden abgewiesen, bevor eval() sie sieht
for bad in ("x.__class__", "x.is_real", "(1).real", "x[0]", "{x}", "'a'", '"a"',
            "__import__", "x;1", "lambda: x", "x if x.is_real else 1", "x # c", "x\\1"):
    try:
        parse(bad)
    except ValueError:
        continue
    raise AssertionError(f"nicht abgewiesen: {bad!r}")

print("alle Tests bestanden")
