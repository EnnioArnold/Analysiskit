#!/usr/bin/env python3
"""Tests für anakit.exercises und anakit.uebung. Aufruf: python3 tests/tests_exercises.py"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "..", "pkg"), os.path.join(HERE, "..", "..")):
    if os.path.isdir(os.path.join(cand, "anakit")):
        sys.path.insert(0, os.path.abspath(cand))
        break

import sympy as sp

from anakit.core import parse, x
from anakit.exercises import (EXERCISES, Exercise, Reading, aufgabe, liste, loese,
                              loese_alle)
from anakit.uebung import (pruefe_ableitung, pruefe_extrema, vorlage_ableitung,
                           vorlage_extrema)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FEHLER:", msg)


def gleich(a, b):
    """Zwei Terme sind als Funktionen gleich."""
    return sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0


R = sp.Rational

# --------------------------------------------------------------------------
# 1. Aufgaben: Aufbau
# --------------------------------------------------------------------------
print("Aufgaben: Aufbau")
check(len(EXERCISES) >= 55, f"zu wenige Aufgaben: {len(EXERCISES)}")
for key, ex in EXERCISES.items():
    check(isinstance(ex, Exercise) and ex.key == key, f"{key}: Schlüssel stimmt nicht")
    check(ex.week == key.split(".")[0], f"{key}: Woche passt nicht")
    check(ex.title and ex.task, f"{key}: Titel oder Aufgabentext fehlt")
    repr(ex)
    check(isinstance(ex._repr_html_(), str), f"{key}: _repr_html_")
for w in ("SW1", "SW2", "SW3", "SW4"):
    l = liste(w)
    check(len(l) >= 5, f"{w}: zu wenige Einträge")
    check(all(k.startswith(w + ".") for k in l), f"{w}: fremde Schlüssel")
    check(l[-1] == f"{w}.DIY", f"{w}: Leseliste fehlt am Ende")
    repr(l); l._repr_html_()
check(liste("sw2") == liste("SW2"), "liste() ist nicht gross-/kleinschreibungsfest")
check(len(liste()) == len(EXERCISES), "liste() ohne Woche")
check(aufgabe("sw1.4") is aufgabe("SW1.4"), "aufgabe() klein geschrieben")
try:
    liste("SW9")
    check(False, "liste('SW9') hätte scheitern müssen")
except KeyError:
    pass

for key in ("SW9.9", "SW2.3", "xyz", "SW2"):
    try:
        aufgabe(key)
        check(False, f"aufgabe('{key}') hätte scheitern müssen")
    except KeyError as e:
        msg = str(e)
        check("Gemeint" in msg or "Woche" in msg, f"Fehlermeldung ohne Vorschlag für {key}: {msg}")
try:
    aufgabe("SW2.3")
except KeyError as e:
    check("SW2.3a" in str(e) and "SW2.3b" in str(e), "Vorschläge für SW2.3 sollten SW2.3a/b nennen")

for k in ("SW1.1", "SW1.3", "SW1.4", "SW1.5a", "SW1.5b", "SW1.6a", "SW1.6b", "SW1.6c",
          "SW1.6d", "SW1.7a", "SW1.7b", "SW2.2", "SW2.3a", "SW2.3b", "SW2.4a", "SW2.4b",
          "SW2.4c", "SW2.5a", "SW2.6", "SW2.7a", "SW2.7b", "SW2.8a", "SW2.8b", "SW2.11",
          "SW3.1", "SW3.2", "SW3.3a", "SW3.3b", "SW3.4", "SW3.5", "SW3.6a", "SW3.6b",
          "SW3.7", "SW4.1", "SW4.2", "SW4.3", "SW4.4", "SW4.5", "SW4.6"):
    check(k in EXERCISES, f"Pflichtaufgabe {k} fehlt")

# Leseliste
r = aufgabe("SW1.DIY")
check(isinstance(r, Reading), "SW1.DIY ist keine Reading")
check("Hughes-Hallet" in repr(r) and "778" in repr(r) and "1146" in repr(r),
      "SW1.DIY: Seitenzahlen fehlen")
check("Seiten 76 – 80" in repr(r), "SW1.DIY: Hughes-Hallet 2.1 fehlt")
r._repr_html_()
try:
    loese("SW1.DIY")
    check(False, "loese(Leseliste) hätte scheitern müssen")
except ValueError as e:
    check("Leseliste" in str(e), "Meldung für Leseliste")
try:
    loese("SW9.9")
    check(False, "loese(unbekannt)")
except KeyError:
    pass

# --------------------------------------------------------------------------
# 2. Aufgaben: alles löst sich, mit Schritten
# --------------------------------------------------------------------------
print("Aufgaben: loese_alle")
alle = loese_alle()
rechen = [k for k, e in EXERCISES.items() if not isinstance(e, Reading)]
check(set(alle) == set(rechen), "loese_alle() deckt nicht alle Rechenaufgaben ab")
for key, res in alle.items():
    check(getattr(res, "steps", None), f"{key}: Ergebnis ohne Schritte")
    repr(res)
    check(isinstance(res._repr_html_(), str), f"{key}: Ergebnis _repr_html_")
for w in ("SW1", "SW2", "SW3", "SW4"):
    check(set(loese_alle(w)) == {k for k in rechen if k.startswith(w + ".")}, f"loese_alle({w})")

# --------------------------------------------------------------------------
# 3. Aufgaben: richtige Werte
# --------------------------------------------------------------------------
print("Aufgaben: Werte")
L = loese


def val(key, **kw):
    return L(key, **kw)


check(val("SW1.1").m == R(2, 3), "SW1.1: m = 2/3")
check(gleich(val("SW1.2").line, R(2, 3) * x + R(8, 3)), "SW1.2: y = 2/3 x + 8/3")
check(val("SW1.3").m == 2 and gleich(val("SW1.3").line, 2 * x + 3), "SW1.3: m = 2, s = 2x+3")
check(val("SW1.4").m == R(5, 2), "SW1.4: m = 5/2")
for k in ("SW1.5a", "SW1.5b"):
    t = val(k)
    check(t.value == 4 and gleich(t.line, 4 * x - 4), f"{k}: f'(2)=4, t = 4x-4")
check(val("SW1.5a", variant="h").value == 4, "loese: variant überschreibbar")
check(gleich(val("SW1.6a").expr, 3 * x**2), "SW1.6a")
check(gleich(val("SW1.6b").expr, 5 * x**4), "SW1.6b")
check(gleich(val("SW1.6c").expr, -1 / x**2), "SW1.6c")
n = sp.Symbol("n")
check(gleich(val("SW1.6d").expr, n * x**(n - 1)), "SW1.6d: n x^(n-1)")
check(gleich(val("SW1.7a").expr, sp.cos(x)), "SW1.7a")
check(gleich(val("SW1.7b").expr, -sp.sin(x)), "SW1.7b")

check(val("SW2.1a").m == R(3, 5) and gleich(val("SW2.1a").line, R(3, 5) * x + R(2, 5)),
      "SW2.1a: Sekante x^3/5")
check(val("SW2.1b").value == -1 and gleich(val("SW2.1b").line, 2 - x), "SW2.1b: Tangente 1/x")
check(val("SW2.2").value == 2, "SW2.2: Grenzwert 2")
check(val("SW2.3a").value == 1, "SW2.3a: Grenzwert 1")
check(not val("SW2.3b").exists, "SW2.3b: kein Grenzwert")
check((val("SW2.3b").left, val("SW2.3b").right) == (1, 0), "SW2.3b: 1 und 0")
check(val("SW2.4a").value == 0, "SW2.4a: |x| -> 0")
check(not val("SW2.4b").exists, "SW2.4b: Heaviside")
check(not val("SW2.4c").exists, "SW2.4c: sgn")
check(val("SW2.5a").value == 0, "SW2.5a")
check(val("SW2.5b").value == sp.oo and val("SW2.5c").value == sp.oo, "SW2.5b/c: +oo")
check(not val("SW2.6").exists, "SW2.6: sin(1/x)")
check(val("SW2.7a").value == 1 and val("SW2.7b").value == 0, "SW2.7")
check(gleich(val("SW2.8a").expr, sp.cos(x)) and gleich(val("SW2.8b").expr, -sp.sin(x)), "SW2.8")
check(val("SW2.9a").value == -sp.oo and val("SW2.9b").value == sp.oo, "SW2.9: tan bei pi/2")

erwartet_stetig = {"a": False, "b": True, "c": False, "d": True, "e": False, "f": False,
                   "g": False, "h": False, "i": False, "j": False}
for s, flag in erwartet_stetig.items():
    check(val(f"SW2.10{s}").continuous is flag, f"SW2.10{s}: stetig = {flag}")
check(val("SW2.10a").kind == "hebbare Lücke", "SW2.10a: hebbare Lücke")
check(val("SW2.10c").kind == "Sprungstelle", "SW2.10c: Sprungstelle")
check(val("SW2.11").continuous is False, "SW2.11: sgn^2 unstetig bei 0")
check(val("SW2.11").kind == "hebbare Lücke", "SW2.11: hebbare Lücke")

check(gleich(val("SW3.1").expr, 6 * x - 5), "SW3.1")
check(gleich(val("SW3.2").expr, (1 + x) * sp.exp(x)), "SW3.2")
check(gleich(val("SW3.3b").expr, (x * sp.sin(x) + x * sp.cos(x) + sp.sin(x)) * sp.exp(x)), "SW3.3b")
tp = val("SW3.3a")
check(any("f'" in s.detail or any("f'" in str(o) for _, o in s.items) for s in tp.steps),
      "SW3.3a: Ableitungsregel fehlt")
check(gleich(val("SW3.4").expr, (1 - x) * sp.exp(-x)), "SW3.4")
check(gleich(val("SW3.5").expr, 2 * x / (x**2 + 1)), "SW3.5")
check(gleich(val("SW3.6a").expr, 1 / x), "SW3.6a")
check(gleich(val("SW3.6b").expr, 1 / (2 * sp.sqrt(x))), "SW3.6b")
check(len(val("SW3.7").rows) >= 6, "SW3.7: Regeltabelle")

F = parse("x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27")
d2 = val("SW4.1")
check(gleich(d2.derivatives[1], 5 * x**4 - 28 * x**3 + 30 * x**2 + 36 * x - 27), "SW4.1: f'")
check(gleich(d2.derivatives[2], 20 * x**3 - 84 * x**2 + 60 * x + 36), "SW4.1: f''")
e = val("SW4.5")
check([p[0] for p in e.minima] == [R(3, 5)], "SW4.5: Minimum bei 3/5")
check([p[0] for p in e.maxima] == [-1], "SW4.5: Maximum bei -1")
check([p[0] for p in e.saddles] == [3], "SW4.5: Sattelpunkt bei 3")
check(val("SW1.4", decimal=True).decimal, "decimal=True")

# --------------------------------------------------------------------------
# 4. Prüfstand: Beispiellösungen (nur hier, nicht im Paket)
# --------------------------------------------------------------------------
print("Prüfstand Ableitung")


def richtige_ableitung(f, x):
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2 * h)


def vorwaerts_klein(f, x):
    h = 1e-8
    return (f(x + h) - f(x)) / h


def zu_gross(f, x):
    h = 0.1
    return (f(x + h) - f(x - h)) / (2 * h)


def zu_klein(f, x):
    h = 1e-14
    return (f(x + h) - f(x - h)) / (2 * h)


def einseitig_h5(f, x):
    h = 1e-5
    return (f(x + h) - f(x)) / h


def faktor_zwei(f, x):
    h = 1e-5
    return (f(x + h) - f(x - h)) / h


def ohne_return(f, x):
    h = 1e-5
    (f(x + h) - f(x - h)) / (2 * h)


def vorzeichen(f, x):
    h = 1e-5
    return (f(x - h) - f(x + h)) / (2 * h)


p = pruefe_ableitung(richtige_ableitung, ausgabe=False)
check(bool(p) and p.ok and p.bestanden == p.gesamt == 7, "symmetrische Lösung wird nicht akzeptiert")
check(any(s.name.startswith("|x|") for s in p.stolpersteine), "Stolperstein |x| fehlt")
check("✓" in p.bericht() and isinstance(p.html(), str), "Bericht/HTML")
check(isinstance(repr(p), str) and p._repr_html_() is not None, "repr/html vor dem Anzeigen")
check(bool(pruefe_ableitung(vorwaerts_klein, ausgabe=False)), "einseitig mit h=1e-8 sollte reichen")

for name, fn, muster in [
        ("zu grosses h", zu_gross, "zu gross"),
        ("zu kleines h", zu_klein, "zu klein"),
        ("einseitig", einseitig_h5, "symmetrische"),
        ("Faktor 2", faktor_zwei, "doppelt so gross"),
        ("ohne return", ohne_return, "return"),
        ("Vorzeichen", vorzeichen, "Vorzeichen")]:
    q = pruefe_ableitung(fn, ausgabe=False)
    check(not q and not q.ok, f"falsche Lösung '{name}' wurde akzeptiert")
    text = q.bericht()
    check(muster in text, f"'{name}': Hinweis mit '{muster}' fehlt")
    check("✗" in text, f"'{name}': kein ✗")

q = pruefe_ableitung(lambda f, x: (_ for _ in ()).throw(RuntimeError("kaputt")), ausgabe=False)
check(not q and "kaputt" in q.bericht(), "Ausnahme in der Studierendenfunktion")
q = pruefe_ableitung(lambda f, x: float("nan"), ausgabe=False)
check(not q and "endlich" in q.bericht(), "nan wird gemeldet")
q = pruefe_ableitung(lambda f, x: (f(x + 0.3) - f(x - 0.3)) / 0.6, ausgabe=False)
check(not q and "zu gross" in q.bericht(), "sqrt-Randfall / grosses h")
check("TODO" in vorlage_ableitung() and "h" in vorlage_ableitung(), "vorlage_ableitung")
exec(compile(vorlage_ableitung(), "vorlage", "exec"), {})   # muss syntaktisch laufen
q = pruefe_ableitung(lambda f, x: None, ausgabe=False)
check(not q, "Vorlage-ähnlich (None) darf nicht bestehen")

print("Prüfstand Extrema")


def werte_ableitung(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def richtige_extrema(f, a, b, n):
    dx = (b - a) / n
    xs = [a + i * dx for i in range(n + 1)]
    d = [werte_ableitung(f, t, dx * 1e-3) for t in xs]
    skala = max(1.0, max(abs(v) for v in d))
    schranke = 1e-6 * skala
    zeichen = [0 if abs(v) <= schranke else (1 if v > 0 else -1) for v in d]
    minima, maxima = [], []
    letzter, letzter_i = 0, None
    for i, s in enumerate(zeichen):
        if s == 0:
            continue
        if letzter != 0 and s != letzter:
            mitte = (xs[letzter_i] + xs[i]) / 2
            (minima if letzter < 0 else maxima).append(mitte)
        letzter, letzter_i = s, i
    return minima, maxima


def streng_extrema(f, a, b, n):
    dx = (b - a) / n
    xs = [a + i * dx for i in range(n + 1)]
    d = [werte_ableitung(f, t, dx * 1e-3) for t in xs]
    minima, maxima = [], []
    for i in range(n):
        if d[i] * d[i + 1] < 0:
            (minima if d[i] < 0 else maxima).append((xs[i] + xs[i + 1]) / 2)
    return minima, maxima


def vertauscht_extrema(f, a, b, n):
    mi, ma = richtige_extrema(f, a, b, n)
    return ma, mi


def nullstellen_extrema(f, a, b, n):
    """Meldet jede Nullstelle der Steigung als Extremum (auch Sattel, auch Rand)."""
    dx = (b - a) / n
    xs = [a + i * dx for i in range(n + 1)]
    d = [werte_ableitung(f, t, dx * 1e-3) for t in xs]
    minima, maxima = [], []
    for i in range(n):
        if d[i] * d[i + 1] <= 0 or abs(d[i]) < 1e-6:
            (minima if d[i + 1] > d[i] else maxima).append(xs[i])
    return minima, maxima


def ohne_listen(f, a, b, n):
    return None


p = pruefe_extrema(richtige_extrema, ausgabe=False)
check(bool(p) and p.bestanden == p.gesamt >= 8, "richtige Extrema-Lösung wird nicht akzeptiert:\n" + p.bericht())

q = pruefe_extrema(streng_extrema, ausgabe=False)
check(not q, "strenger Vorzeichentest wurde akzeptiert")
check("Gitterpunkt" in q.bericht() and "nicht gefunden" in q.bericht(),
      "Hinweis auf Gitterpunkt fehlt:\n" + q.bericht())

q = pruefe_extrema(vertauscht_extrema, ausgabe=False)
check(not q and "vertauscht" in q.bericht(), "vertauschte Listen wurden nicht erkannt")

q = pruefe_extrema(nullstellen_extrema, ausgabe=False)
check(not q, "Nullstellen-Lösung wurde akzeptiert")
check("Sattelpunkt" in q.bericht(), "Sattelpunkt-Hinweis fehlt")

q = pruefe_extrema(ohne_listen, ausgabe=False)
check(not q and "None" in q.bericht() or "return" in q.bericht(), "None-Rückgabe")
q = pruefe_extrema(lambda f, a, b, n: ([], []), ausgabe=False)
check(not q and 0 < q.bestanden < q.gesamt, "leere Listen sollten nur bei den Fällen ohne Extrema bestehen")
check(any(f.name.startswith("konstante") and f.ok for f in q.faelle), "konstante Funktion: leere Listen sind richtig")
check("TODO" in vorlage_extrema(), "vorlage_extrema")
exec(compile(vorlage_extrema(), "vorlage", "exec"), {})
q = pruefe_extrema(lambda f, a, b, n: (_ for _ in ()).throw(RuntimeError("bumm")), ausgabe=False)
check(not q and "bumm" in q.bericht(), "Ausnahme im Extrema-Prüfstand")
# Vorlage unverändert: liefert zwei leere Listen und besteht daher nicht
ns = {}
exec(compile(vorlage_extrema(), "vorlage", "exec"), ns)
check(not pruefe_extrema(ns["meine_extrema"], ausgabe=False), "unausgefüllte Vorlage darf nicht bestehen")

# Anzeige: show() druckt im Skript
import io
import contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = pruefe_ableitung(richtige_ableitung)
check("von 7 Fällen bestanden" in buf.getvalue() and "Alles bestanden" in buf.getvalue(), "Ausgabe im Skript")
check(repr(r).startswith("<"), "kompakte repr nach der Anzeige")

print()
if FAILS:
    print(f"{len(FAILS)} FEHLER")
    sys.exit(1)
print("Alle Tests bestanden.")
