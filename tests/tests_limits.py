#!/usr/bin/env python3
"""Tests für anakit.limits: python3 tests/tests_limits.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pkg'))
import sympy as sp
from anakit.limits import (limit, limit_infinity, limit_laws, continuity,
                           is_continuous, limit_table)

oo = sp.oo
def eq(a, b): assert sp.simplify(a - b) == 0, (a, b)

# Folie: Beispiele für Grenzwerte
r = limit("(x^2-1)/(x-1)", 1); eq(r.value, 2); assert r.exists and r.kind == 'eigentlich'
assert any('Kürzen' in s.title for s in r.steps)
r = limit("x^2 if x<1 else 2-x", 1); eq(r.left, 1); eq(r.right, 1); eq(r.value, 1)
c = continuity("x^2 if x<1 else 2-x", 1); assert c.continuous and c.kind == 'stetig'
r = limit("x^2 if x<1 else x-1", 1); eq(r.left, 1); eq(r.right, 0)
assert not r.exists and r.kind == 'existiert nicht' and r.value is None
c = continuity("x^2 if x<1 else x-1", 1); assert not c.continuous and c.kind == 'Sprungstelle'; eq(c.jump, -1)
r = limit("abs(x)", 0); eq(r.value, 0); assert continuity("abs(x)", 0).continuous
r = limit("H(x)", 0); eq(r.left, 0); eq(r.right, 1); assert not r.exists
c = continuity("H(x)", 0); assert not c.continuous and c.kind == 'Sprungstelle'; eq(c.f_x0, 1)
r = limit("sgn(x)", 0); eq(r.left, -1); eq(r.right, 1); assert not r.exists
assert not continuity("sgn(x)", 0).continuous
r = limit("1/x^2", oo); eq(r.value, 0); eq(limit_infinity("1/x^2").value, 0)
eq(limit_infinity("1/x", '-').value, 0); eq(limit("1/x", "-oo").value, 0)
eq(limit_infinity("(2x^2+1)/(x^2-3)").value, 2); assert limit_infinity("x^3/(x^2+1)").value == oo
assert limit_infinity("x^3/(x^2+1)", '-').value == -oo
eq(limit_infinity("sin(x)/x").value, 0)
r = limit("1/x^2", 0, '+'); assert r.value == oo and r.kind == 'uneigentlich'
r = limit("1/x^2", 0, '-'); assert r.value == oo
r = limit("1/x^2", 0); assert r.value == oo and r.kind == 'uneigentlich'
r = limit("1/x", 0); assert not r.exists and r.left == -oo and r.right == oo
r = limit("sin(1/x)", 0); assert not r.exists and r.kind == 'existiert nicht'
assert continuity("sin(1/x)", 0).kind == 'Oszillation'
eq(limit("sin(x)/x", 0).value, 1); eq(limit("(1-cos(x))/x", 0).value, 0)
assert any('Einschnürungssatz' in s.rule for s in limit("sin(x)/x", 0).steps)
r = limit("sgn(x)^2", 0); eq(r.value, 1)
c = continuity("sgn(x)^2", 0); eq(c.f_x0, 0); eq(c.limit, 1)
assert not c.continuous and c.kind == 'hebbare Lücke'
# tan: korrekte Richtung (Folie vertauscht)
r = limit("tan(x)", "pi/2"); assert r.left == oo and r.right == -oo and not r.exists
assert limit("tan(x)", sp.pi/2, '-').value == oo and limit("tan(x)", sp.pi/2, '+').value == -oo
assert 'Folie' in r.steps[-1].note
# Weiteres
c = continuity("(x^2-1)/(x-1)", 1); assert c.kind == 'hebbare Lücke' and c.f_x0 is None
c = continuity("1/x", 0); assert c.kind == 'Polstelle' and not c.continuous
assert continuity("x^2+1", 3).continuous
assert limit("sqrt(x)", 0, '-').kind == 'existiert nicht'
# ganze Funktion
r = is_continuous("x^2 if x<1 else 2-x"); assert r.continuous and r.kind == 'stetig'
r = is_continuous("x^2 if x<1 else x-1"); assert not r.continuous and r.discontinuities[0][0] == 1
r = is_continuous("H(x)"); assert not r.continuous
r = is_continuous("abs(x)"); assert r.continuous
r = is_continuous("1/(x-2)"); assert r.continuous and r.gaps and r.kind == 'stetig auf dem Definitionsbereich'
r = is_continuous("sgn(x)^2"); assert not r.continuous
r = is_continuous("x^2+x"); assert r.continuous
r = is_continuous("tan(x)", domain=(0, 3)); assert r.gaps
# Grenzwertsätze
L = limit_laws("x^2+1", "2x-1", 2); assert L.all_ok; eq(L.a, 5); eq(L.b, 3)
assert set(L.checks) == {'Summe', 'Differenz', 'Produkt', 'Quotient', 'Verkettung'}
L = limit_laws("sin(x)/x", "cos(x)", 0); assert L.all_ok
L = limit_laws("x", "x-1", 1); assert L.all_ok and 'Quotient' not in L.checks
# Tabelle
T = limit_table("(x^2-1)/(x-1)", 1); assert len(T.rows_left) == 6 and len(T.rows_right) == 6
assert abs(float(T.rows_left[5][1]) - 2) < 1e-5
T = limit_table("1/x", "oo"); assert len(T.rows_right) == 6
# Darstellung
for obj in (limit("(x^2-1)/(x-1)", 1), limit("tan(x)", "pi/2"), continuity("H(x)", 0),
            is_continuous("1/x"), limit_laws("x", "x+1", 1), limit_table("sin(x)/x", 0),
            limit_infinity("1/x^2")):
    repr(obj); obj._repr_html_(); obj.latex()
print("alle Tests bestanden")
