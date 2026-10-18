#!/usr/bin/env python3
"""
Die Rechenaufgaben aus den Folien der Semesterwochen 1 bis 4.

Die Aufgaben stehen hier als Daten, nicht als fertige Antworten. Du holst
dir eine Aufgabe, rechnest selbst und vergleichst — das ist der Zweck.
`loese(...)` rechnet sie durch, wenn du kontrollieren willst.

    from anakit.exercises import aufgabe, liste, loese

    liste("SW2")              # was es gibt
    a = aufgabe("SW2.3a")     # die Daten
    loese("SW2.3a")           # mit Rechenweg

Aufgabenschlüssel folgen der Reihenfolge der Folien: `SW1.3` ist das dritte
Beispiel von Semesterwoche 1, `SW2.5b` die zweite Teilaufgabe des fünften
Beispiels in Semesterwoche 2. Die Aufgaben aus Papula und Hughes-Hallet
kennt dieses Paket nicht — sie stehen nur als Leseliste (`SW1.DIY` usw.)
mit den Seitenzahlen von den Folien.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from ..rechnen.render import Result, Step

__all__ = ["Exercise", "Reading", "EXERCISES", "aufgabe", "liste", "loese",
           "loese_alle", "TripleProductResult"]


# ==========================================================================
# Datenstruktur
# ==========================================================================

def _zeige(value) -> str:
    """Eine Angabe lesbar: Punkte als (x/y), Optionen als name=wert."""
    if isinstance(value, tuple) and len(value) == 2:
        return f"({value[0]}/{value[1]})".replace("-", "−")
    if isinstance(value, dict):
        return ", ".join(f"{k} = {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)


@dataclass
class Exercise:
    """Eine Aufgabe mit ihren Daten und dem passenden Verfahren."""
    key: str
    week: str
    title: str
    task: str
    data: dict = field(default_factory=dict)
    method: str = ""
    note: str = ""

    def __repr__(self) -> str:
        lines = [f"{self.key} — {self.title}", f"  {self.task}"]
        for name, value in self.data.items():
            lines.append(f"  {name} = {_zeige(value)}")
        if self.note:
            lines.append(f"  Hinweis: {self.note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        rows = "".join(
            f"<div style='font-family:SF Mono,Menlo,Consolas,monospace;font-size:13px'>"
            f"{html.escape(str(name))} = {html.escape(_zeige(value))}</div>"
            for name, value in self.data.items())
        note = (f"<p style='color:#86868b;font-size:12px'>"
                f"{html.escape(self.note)}</p>" if self.note else "")
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,"
            "'Segoe UI',Helvetica,sans-serif\">"
            f"<p style='font-weight:600;margin:0'>{html.escape(self.key)} — "
            f"{html.escape(self.title)}</p>"
            f"<p style='color:#6e6e73;font-size:13px;margin:2px 0 10px'>"
            f"{html.escape(self.task)}</p>" + rows + note + "</div>")


@dataclass
class Reading(Exercise):
    """
    Verweisliste „Do It Yourself“ einer Woche: Theorie und Übungen aus
    Papula und Hughes-Hallet. Ohne Lösung — die Aufgaben stehen im Buch.
    `data` hat die Einträge „Theorie“ und „Übungen“ (Listen von Texten).
    """
    method: str = "lektuere"

    def __repr__(self) -> str:
        lines = [f"{self.key} — {self.title}", f"  {self.task}"]
        for name, items in self.data.items():
            lines.append(f"  {name}:")
            lines += [f"    • {item}" for item in items]
        if self.note:
            lines.append(f"  Hinweis: {self.note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        blocks = []
        for name, items in self.data.items():
            lis = "".join(f"<li>{html.escape(i)}</li>" for i in items)
            blocks.append(f"<p style='font-weight:600;font-size:13px;margin:10px 0 2px'>"
                          f"{html.escape(name)}</p><ul style='margin:0;font-size:13px'>"
                          f"{lis}</ul>")
        note = (f"<p style='color:#86868b;font-size:12px'>{html.escape(self.note)}</p>"
                if self.note else "")
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,"
            "'Segoe UI',Helvetica,sans-serif\">"
            f"<p style='font-weight:600;margin:0'>{html.escape(self.key)} — "
            f"{html.escape(self.title)}</p>"
            f"<p style='color:#6e6e73;font-size:13px;margin:2px 0 0'>"
            f"{html.escape(self.task)}</p>" + "".join(blocks) + note + "</div>")


class Liste(list):
    """Liste von Aufgabenschlüsseln, die sich mit Titel anzeigt."""

    def __repr__(self) -> str:
        if not self:
            return "(keine Aufgaben)"
        w = max(len(k) for k in self)
        return "\n".join(f"{k:<{w}}  {EXERCISES[k].title}" for k in self)

    def _repr_html_(self) -> str:
        import html
        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0;font-family:SF Mono,Menlo,"
            f"Consolas,monospace;font-size:13px'>{html.escape(k)}</td>"
            f"<td style='font-size:13px'>{html.escape(EXERCISES[k].title)}</td></tr>"
            for k in self)
        return (
            "<table style=\"font-family:-apple-system,BlinkMacSystemFont,"
            f"'Segoe UI',Helvetica,sans-serif;border-collapse:collapse\">{rows}</table>")


EXERCISES: dict[str, Exercise] = {}


def _add(ex: Exercise):
    if ex.key in EXERCISES:
        raise ValueError(f"Doppelter Aufgabenschlüssel {ex.key}")
    EXERCISES[ex.key] = ex


def _ex(key, title, task, data, method, note=""):
    _add(Exercise(key, key.split(".")[0], title, task, data, method, note))


# ==========================================================================
# Die Sammlung
# ==========================================================================

# -- Semesterwoche 1: Sekanten und Tangenten --------------------------------

_ex("SW1.1", "Steigung einer Geraden",
    "Steigung der Geraden durch die Punkte P1(−1/2) und P2(2/4) bestimmen.",
    {"P1": ("-1", "2"), "P2": ("2", "4")}, "slope",
    "Folie 5. Schreibweise P(x/y): P1 hat x = −1 und y = 2.")

_ex("SW1.2", "Gerade durch zwei Punkte",
    "Die Gleichung der Geraden durch zwei Punkte mit verschiedenen "
    "x-Koordinaten berechnen.",
    {"P1": ("-1", "2"), "P2": ("2", "4")}, "line_through",
    "Folie 7 lässt die Punkte frei. Hier dieselben wie in SW1.1; "
    "probier auch eigene aus.")

_ex("SW1.3", "Sekante an x² (zwei Stellen)",
    "Sekante zu f(x) = x² mit x1 = −1 und x2 = 3.",
    {"f": "x^2", "x1": "-1", "x2": "3"}, "secant", "Folie 9.")

_ex("SW1.4", "Sekante an x² (Stelle und Abstand)",
    "Sekante zu f(x) = x² mit x0 = 1 und h = 1/2.",
    {"f": "x^2", "x0": "1", "h": "1/2"}, "secant_h", "Folie 10.")

_ex("SW1.5a", "Tangente an x² bei 2 (Grenzwert x → x0)",
    "Tangente zu f(x) = x² an der Stelle x0 = 2 über "
    "lim_{x→x0} (f(x) − f(x0))/(x − x0).",
    {"f": "x^2", "x0": "2", "kwargs": {"variant": "x"}}, "tangent", "Folie 12.")

_ex("SW1.5b", "Tangente an x² bei 2 (Grenzwert h → 0)",
    "Tangente zu f(x) = x² an der Stelle x0 = 2 über "
    "lim_{h→0} (f(x0 + h) − f(x0))/h.",
    {"f": "x^2", "x0": "2", "kwargs": {"variant": "h"}}, "tangent", "Folie 13.")

_ex("SW1.6a", "Ableitungsfunktion von x³",
    "Die Formel für die Ableitung auf p1(x) = x³ anwenden.",
    {"f": "x^3"}, "derivative_by_definition", "Folie 15.")

_ex("SW1.6b", "Ableitungsfunktion von x⁵",
    "Die Formel für die Ableitung auf p5(x) = x⁵ anwenden.",
    {"f": "x^5"}, "derivative_by_definition",
    "Folie 15. Hinweis der Folie: binomischer Satz für (x + h)⁵.")

_ex("SW1.6c", "Ableitungsfunktion von 1/x",
    "Die Formel für die Ableitung auf r(x) = 1/x anwenden.",
    {"f": "1/x"}, "derivative_by_definition", "Folie 15.")

_ex("SW1.6d", "Vermutung für xⁿ",
    "Vermutung, wie die Ableitung von xⁿ lautet — und die Herleitung dazu.",
    {"f": "x^n"}, "derivative_by_definition",
    "Folie 15. Vergleiche mit den Ergebnissen für x³, x⁵ und 1/x = x⁻¹.")

_ex("SW1.7a", "Ableitung von sin x",
    "Wie könnte die Ableitungsfunktion zu t1(x) = sin(x) lauten?",
    {"f": "sin(x)"}, "derivative_by_definition",
    "Folie 15. Die Herleitung braucht zwei Grenzwerte aus SW2.")

_ex("SW1.7b", "Ableitung von cos x",
    "Wie könnte die Ableitungsfunktion zu t2(x) = cos(x) lauten?",
    {"f": "cos(x)"}, "derivative_by_definition", "Folie 15.")

# -- Semesterwoche 2: Grenzwert und Stetigkeit --------------------------------

_ex("SW2.1a", "Wiederholung: Sekante an x³/5",
    "Gleichung der Sekante zu f(x) = x³/5 durch die Punkte an den Stellen "
    "−1 und 2.",
    {"f": "x^3/5", "x1": "-1", "x2": "2"}, "secant", "Folie 5.")

_ex("SW2.1b", "Wiederholung: Tangente an 1/x",
    "Gleichung der Tangente zu g(x) = 1/x an der Stelle x0 = 1.",
    {"f": "1/x", "x0": "1"}, "tangent", "Folie 5.")

_ex("SW2.2", "Grenzwert (x² − 1)/(x − 1)",
    "Grenzwert für x → 1.",
    {"f": "(x^2-1)/(x-1)", "x0": "1"}, "limit",
    "Folie 10. An der Stelle 1 ist die Funktion nicht definiert.")

_ex("SW2.3a", "Abschnittsweise Funktion f: x² und 2 − x",
    "Grenzwert von f für x → 1 mit f(x) = x² für x < 1 und 2 − x für x ≥ 1.",
    {"f": "x^2 if x<1 else 2-x", "x0": "1"}, "limit", "Folie 10.")

_ex("SW2.3b", "Abschnittsweise Funktion g: x² und x − 1",
    "Grenzwert von g für x → 1 mit g(x) = x² für x < 1 und x − 1 für x ≥ 1.",
    {"f": "x^2 if x<1 else x-1", "x0": "1"}, "limit",
    "Folie 10 (dort steht beim Funktionsnamen versehentlich f statt g).")

_ex("SW2.4a", "Grenzwert des Betrags",
    "Grenzwert von |x| für x → 0.",
    {"f": "abs(x)", "x0": "0"}, "limit", "Folie 10.")

_ex("SW2.4b", "Grenzwert der Heaviside-Funktion",
    "Grenzwert von H(x) für x → 0 mit H(x) = 0 für x < 0 und 1 für x ≥ 0.",
    {"f": "0 if x<0 else 1", "x0": "0"}, "limit", "Folie 10.")

_ex("SW2.4c", "Grenzwert der Signum-Funktion",
    "Grenzwert von sgn(x) für x → 0.",
    {"f": "sgn(x)", "x0": "0"}, "limit", "Folie 11.")

_ex("SW2.5a", "1/x² für x → ∞",
    "Verhalten von 1/x² im Unendlichen.",
    {"f": "1/x^2", "kwargs": {"direction": "+"}}, "limit_infinity", "Folie 11.")

_ex("SW2.5b", "1/x² für x → 0 von rechts",
    "Rechtsseitiger Grenzwert von 1/x² an der Stelle 0.",
    {"f": "1/x^2", "x0": "0", "kwargs": {"side": "+"}}, "limit", "Folie 11.")

_ex("SW2.5c", "1/x² für x → 0 von links",
    "Linksseitiger Grenzwert von 1/x² an der Stelle 0.",
    {"f": "1/x^2", "x0": "0", "kwargs": {"side": "-"}}, "limit", "Folie 11.")

_ex("SW2.6", "Grenzwert von sin(1/x)",
    "Grenzwert von sin(1/x) für x → 0.",
    {"f": "sin(1/x)", "x0": "0"}, "limit",
    "Folie 11. Überlege dir zwei Folgen gegen 0 mit verschiedenen Funktionswerten.")

_ex("SW2.7a", "Grenzwert von sin(x)/x",
    "Grenzwert von sin(x)/x für x → 0.",
    {"f": "sin(x)/x", "x0": "0"}, "limit",
    "Folie 11. Einschnürungssatz (Squeezing Theorem).")

_ex("SW2.7b", "Grenzwert von (1 − cos x)/x",
    "Grenzwert von (1 − cos(x))/x für x → 0.",
    {"f": "(1-cos(x))/x", "x0": "0"}, "limit", "Folie 11.")

_ex("SW2.8a", "Ableitung von sin über den Grenzwert",
    "(sin x)' = lim_{h→0} (sin(x + h) − sin(x))/h berechnen.",
    {"f": "sin(x)"}, "derivative_by_definition",
    "Folie 11. Additionstheorem und die beiden Grenzwerte aus SW2.7.")

_ex("SW2.8b", "Ableitung von cos über den Grenzwert",
    "(cos x)' über den Grenzwert berechnen.",
    {"f": "cos(x)"}, "derivative_by_definition", "Folie 11.")

_ex("SW2.9a", "tan x für x → π/2 von rechts",
    "Uneigentlicher Grenzwert von tan(x) für x → π/2 von rechts.",
    {"f": "tan(x)", "x0": "pi/2", "kwargs": {"side": "+"}}, "limit",
    "Folie 8. Die Folie ordnet die Vorzeichen anders zu — rechne selbst nach: "
    "rechts von π/2 ist tan negativ.")

_ex("SW2.9b", "tan x für x → π/2 von links",
    "Uneigentlicher Grenzwert von tan(x) für x → π/2 von links.",
    {"f": "tan(x)", "x0": "pi/2", "kwargs": {"side": "-"}}, "limit", "Folie 8.")

for _suffix, _title, _f, _x0 in [
        ("a", "(x² − 1)/(x − 1)", "(x^2-1)/(x-1)", "1"),
        ("b", "f mit x² und 2 − x", "x^2 if x<1 else 2-x", "1"),
        ("c", "g mit x² und x − 1", "x^2 if x<1 else x-1", "1"),
        ("d", "|x|", "abs(x)", "0"),
        ("e", "Heaviside H(x)", "0 if x<0 else 1", "0"),
        ("f", "sgn(x)", "sgn(x)", "0"),
        ("g", "1/x²", "1/x^2", "0"),
        ("h", "sin(1/x)", "sin(1/x)", "0"),
        ("i", "sin(x)/x", "sin(x)/x", "0"),
        ("j", "(1 − cos x)/x", "(1-cos(x))/x", "0")]:
    _ex(f"SW2.10{_suffix}", f"Stetigkeit von {_title}",
        f"Ist die Funktion an der Stelle {_x0} stetig? Links- und rechtsseitiger "
        "Grenzwert mit dem Funktionswert vergleichen.",
        {"f": _f, "x0": _x0}, "continuity", "Folie 16.")

_ex("SW2.11", "Stetigkeit von sgn²",
    "Ist die Funktion sgn²(x) = (sgn(x))² an der Stelle 0 stetig?",
    {"f": "sgn(x)^2", "x0": "0"}, "continuity",
    "Folie 16. Achtung: sgn(0) = 0, also ist auch sgn²(0) = 0.")

# -- Semesterwoche 3: Ableitungsregeln -----------------------------------------

_ex("SW3.1", "Polynom ableiten",
    "f(x) = 3x² − 5x + 7 mit Summen-, Faktor- und Summandenregel ableiten.",
    {"f": "3x^2 - 5x + 7"}, "diff", "Folie 9.")

_ex("SW3.2", "Produktregel: x·eˣ",
    "f(x) = x·eˣ ableiten.",
    {"f": "x", "g": "e^x"}, "diff_product", "Folie 12.")

_ex("SW3.3a", "Produktregel mit drei Faktoren (allgemein)",
    "(f·g·h)' allgemein herleiten, indem man zuerst f·g als einen Faktor "
    "auffasst.",
    {"f": "f(x)", "g": "g(x)", "h": "h(x)"}, "triple_product", "Folie 13.")

_ex("SW3.3b", "Produktregel mit drei Faktoren: x·sin x·eˣ",
    "f(x) = x·sin(x)·eˣ ableiten.",
    {"f": "x*sin(x)*e^x"}, "diff",
    "Folie 13 (konkretes Beispiel, hier selbst gewählt). "
    "Auch hier wandert die Ableitung von Faktor zu Faktor.")

_ex("SW3.4", "Quotientenregel: x/eˣ",
    "f(x) = x/eˣ ableiten.",
    {"f": "x", "g": "e^x"}, "diff_quotient", "Folie 16.")

_ex("SW3.5", "Kettenregel: ln(x² + 1)",
    "f(x) = ln(x² + 1) mit äusserer Funktion g(y) = ln y und innerer "
    "Funktion h(x) = x² + 1 ableiten.",
    {"outer": "ln(x)", "inner": "x^2+1"}, "diff_chain", "Folie 19.")

_ex("SW3.6a", "Umkehrfunktion: ln x",
    "Die Ableitung von f(x) = ln x aus der Umkehrfunktion g(x) = eˣ "
    "herleiten.",
    {"f": "ln(x)"}, "diff_inverse", "Folie 22.")

_ex("SW3.6b", "Umkehrfunktion: √x",
    "Die Ableitung von f(x) = √x aus der Umkehrfunktion g(x) = x² "
    "herleiten.",
    {"f": "sqrt(x)"}, "diff_inverse",
    "Folie 23. Zur Kontrolle auch als x^(1/2) mit der Potenzregel.")

_ex("SW3.7", "Regeltabelle",
    "Alle Ableitungsregeln in Newton- und Leibniz-Schreibweise.",
    {}, "rule_table", "Folie 24.")

# -- Semesterwoche 4: Monotonie und Extrema ----------------------------------

_SW4_F = "x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27"

_ex("SW4.1", "Beispielfunktion: Ableitungen",
    "f' und f'' der Beispielfunktion bilden.",
    {"f": _SW4_F, "kwargs": {"n": 2}}, "diff", "Folie 5.")

_ex("SW4.2", "Erste Ableitung und Monotonie",
    "Wo wächst bzw. fällt f? Vorzeichentabelle von f'.",
    {"f": _SW4_F}, "monotony", "Folie 7.")

_ex("SW4.3", "Kritische Punkte",
    "Die kritischen Stellen (f'(x0) = 0) und Punkte bestimmen und einordnen.",
    {"f": _SW4_F}, "critical_points", "Folie 9.")

_ex("SW4.4", "Zweite Ableitung und Krümmung",
    "Wo ist f linksgekrümmt, wo rechtsgekrümmt?",
    {"f": _SW4_F}, "curvature", "Folie 11.")

_ex("SW4.5", "Lokale Extrema",
    "Welche kritischen Punkte sind relative Minima oder Maxima, welche "
    "Sattelpunkte?",
    {"f": _SW4_F}, "extrema",
    "Folie 13. Bei f'(x0) = 0 und f''(x0) = 0 sagt das Kriterium nichts aus.")

_ex("SW4.6", "Ganze Kurvendiskussion",
    "Monotonie, kritische Punkte, Krümmung und Extrema der Beispielfunktion "
    "zusammen.",
    {"f": _SW4_F}, "curve_discussion", "Folien 5 bis 13.")


# -- Do It Yourself: Verweise ohne Lösung --------------------------------------

def _diy(key, theorie, uebungen):
    _add(Reading(
        key, key.split(".")[0], "Do It Yourself (Literatur und Übungen)",
        "Verweise aus den Folien. Die Aufgaben stehen im Buch und sind hier "
        "nicht abgedruckt, es gibt dazu keine Lösung im Paket.",
        {"Theorie": theorie, "Übungen": uebungen}))


_diy("SW1.DIY", [
    "Video: Differenzen- und Differentialquotient",
    "Wiki: Differenzenquotient",
    "Video: Sekante",
    "Papula Band 1, IV.1.1 und IV.1.2 (Seiten 335 – 340 1/3)",
    "Hughes-Hallet: Zusammenfassung „The Derivative“ (Seite 1127, bis vor der blauen Box)",
    "Hughes-Hallet: Chapter 2.1 (Seiten 76 – 80)",
    "Hughes-Hallet: Chapter 2.2 (Seiten 83 – 87)",
], [
    "Papula Band 1, IV, zu Abschnitt 1: Aufgaben 1a) und b), Seite 426 (Lösung: Seite 778)",
    "Hughes-Hallet, Exercises and Problems for Section 2.1: 2, 4, 11 – 13, 18 – 19 "
    "(Seiten 80 – 81, Lösungen Seite 1146)",
    "Hughes-Hallet, Exercises and Problems for Section 2.2: 1, 6, 10 – 12 "
    "(Seiten 87 – 88, Lösungen Seite 1147)",
])

_diy("SW2.DIY", [
    "Papula Band 1, III.4.1 – 4.4 (Seiten 173 – 189)",
    "Papula Band 1, IV.1.3 (Seiten 340 – 342)",
    "Papula Band 1, IV.2.1 – IV.2.7 (Seiten 343 – 344)",
], [
    "Papula Band 1, III, zu Abschnitt 4: Aufgaben 4, 7, 8 und 9 (Seiten 324 – 325, "
    "Lösungen Seiten 760 und 761)",
    "Papula Band 1, IV, zu Abschnitt 1: Aufgabe 1 (Seite 426, Lösung Seite 778)",
    "Papula Band 1, IV, zu Abschnitt 2: Aufgabe 1 (Seite 426, Lösung Seite 778)",
])

_diy("SW3.DIY", [
    "Papula Band 1, Kapitel IV, Abschnitt 1.3 (Seiten 340 – 341)",
    "Papula Band 1, Kapitel IV, Abschnitte 2.1 – 2.7 (Seiten 343 – 357)",
], [
    "Papula Band 1, Kapitel IV, zu Abschnitt 2: Aufgaben 1 und 2 (Seite 426)",
    "3 selbstgewählte Teilaufgaben aus Aufgabe 3 (Seite 426)",
    "3 selbstgewählte Teilaufgaben aus Aufgabe 4 (Seite 427)",
    "3 selbstgewählte Teilaufgaben aus Aufgabe 5 (Seite 427)",
    "3 selbstgewählte Teilaufgaben aus Aufgabe 6 (Seite 427)",
    "Aufgaben 7 – 9 (Seiten 427 – 428)",
])

_diy("SW4.DIY", [
    "Papula Kapitel III: Erste Ableitung, Monotonie — Abschnitt 2.3, Seiten 154 – 157 "
    "(14. und 16. Aufl.); Abschnitt 2.5, Seiten 159 – 163 (14. und 16. Aufl.)",
    "Papula Kapitel IV: Zweite Ableitung, Krümmungsverhalten (ohne Krümmung und "
    "Krümmungskreis einer Kurve) — Abschnitt 2.11, Seiten 352 – 353 (14. Aufl.), "
    "364 – 365 (16. Aufl.); Abschnitt 3.3, Seiten 371 – 374 (14. Aufl.), "
    "383 – 386 (16. Aufl.)",
    "Papula Kapitel IV: Kritische Punkte, Minima und Maxima — Abschnitt 3.4.1 – 3.4.2, "
    "Seiten 382 – 388 (14. Aufl.), 394 – 402 (16. Aufl.)",
], [
    "Kapitel III, Abschnitt 2: Aufgaben 3 a) – 3 f); Abschnitt 3: Aufgaben 5 a) und 5 b) — "
    "Seite 311 (14. Aufl.), 323 (16. Aufl.), Lösungen Seite 745 (14. Aufl.), 757 (16. Aufl.)",
    "Kapitel IV, Abschnitt 2: Aufgaben 6 a) – f), 15 a) – f), 16 a) – 16 c) — "
    "Seiten 415 – 417 (14. Aufl.), 427 – 429 (16. Aufl.); Lösungen Seite 770 (14. Aufl.), "
    "782 – 785 (16. Aufl.)",
    "Kapitel IV, Abschnitt 3: Aufgabe 5) — Seite 418 (14. Aufl.), 430 (16. Aufl.); "
    "Lösung Seite 775 (14. Aufl.), 789 (16. Aufl.)",
    "Kapitel IV, Abschnitt 3: Aufgaben 11), 12 a) – 12 f) — Seite 419 (14. Aufl.), "
    "431 (16. Aufl.); Lösungen Seite 777 (14. Aufl.), 791 (16. Aufl.)",
])


# ==========================================================================
# Zugriff
# ==========================================================================

_LOWER = {k.lower(): k for k in EXERCISES}


def _wochen() -> list[str]:
    out: list[str] = []
    for e in EXERCISES.values():
        if e.week not in out:
            out.append(e.week)
    return out


def liste(week: str | None = None) -> Liste:
    """
    Alle Aufgabenschlüssel in der Reihenfolge der Folien, auf Wunsch nur
    einer Woche (`liste("SW2")`). Die Leseliste „Do It Yourself“ steht
    jeweils am Ende (`SW2.DIY`).
    """
    keys = list(EXERCISES)
    if week:
        w = str(week).upper().strip()
        if w.isdigit():
            w = "SW" + w
        if w not in _wochen():
            raise KeyError(f"Woche '{week}' gibt es nicht. Möglich: {', '.join(_wochen())}")
        keys = [k for k in keys if EXERCISES[k].week == w]
    return Liste(keys)


def _unbekannt(key: str) -> KeyError:
    text = str(key).strip()
    weeks = _wochen()
    if text.upper() in weeks:
        return KeyError(f"'{key}' ist eine Woche, keine Aufgabe. "
                        f"liste('{text.upper()}') zeigt die Aufgaben.")
    pool = list(EXERCISES)
    low = text.lower()
    near = [k for k in pool if k.lower().startswith(low)]
    for n in difflib.get_close_matches(low, [k.lower() for k in pool], n=5, cutoff=0.6):
        if _LOWER[n] not in near:
            near.append(_LOWER[n])
    near = near[:5]
    if not near:
        head = text.upper().split(".")[0]
        near = [k for k in pool if k.upper().startswith(head)][:6]
    if not near:
        near = pool[:6]
    return KeyError(f"Aufgabe '{key}' gibt es nicht. Gemeint war vielleicht: "
                    f"{', '.join(near)}. (liste() zeigt alle.)")


def aufgabe(key: str) -> Exercise:
    """Eine Aufgabe mit ihren Daten. Gross-/Kleinschreibung ist egal."""
    if key in EXERCISES:
        return EXERCISES[key]
    real = _LOWER.get(str(key).strip().lower())
    if real is None:
        raise _unbekannt(key)
    return EXERCISES[real]


# ==========================================================================
# Rechnen
# ==========================================================================

class TripleProductResult(Result):
    """Produktregel für drei Faktoren, allgemein hergeleitet (Folie 13)."""

    title = "Produktregel mit drei Faktoren"

    def __init__(self):
        super().__init__()
        self.summary = "(f·g·h)' = f'·g·h + f·g'·h + f·g·h'"
        self.steps = [
            Step("Zwei Faktoren zusammenfassen",
                 "Man fasst $f\\cdot g$ als einen Faktor auf und wendet die "
                 "Produktregel für zwei Faktoren an.",
                 [("(f·g·h)'", r"\left(\left(f\cdot g\right)\cdot h\right)'")],
                 rule="Produktregel"),
            Step("Äussere Produktregel",
                 "Die Ableitung wandert von $(f\\cdot g)$ zu $h$.",
                 [("", r"\left(f\cdot g\right)'\cdot h + \left(f\cdot g\right)\cdot h'")],
                 rule="Produktregel"),
            Step("Innere Produktregel",
                 "Jetzt $(f\\cdot g)'$ selbst nach der Produktregel.",
                 [("", r"\left(f'\cdot g + f\cdot g'\right)\cdot h + f\cdot g\cdot h'")],
                 rule="Produktregel"),
            Step("Ausmultiplizieren",
                 "In jedem Summanden ist genau ein Faktor abgeleitet.",
                 [("(f·g·h)'", r"f'\cdot g\cdot h + f\cdot g'\cdot h + f\cdot g\cdot h'")],
                 note="Auch hier wandert die Ableitung von einem Faktor zum anderen. "
                      "Bei vier Faktoren p·q·r·s gibt es entsprechend vier Summanden."),
        ]
        self.highlights = [("(f·g·h)'", r"f'\cdot g\cdot h + f\cdot g'\cdot h + f\cdot g\cdot h'")]
        self.expr = None


def _lookup():
    """Alle Verfahren, die als `method` vorkommen dürfen."""
    from ..rechnen import analysis, derivative, limits, steigung
    table = {}
    for mod in (steigung, limits, derivative, analysis):
        for name in dir(mod):
            if not name.startswith("_") and callable(getattr(mod, name)):
                table[name] = getattr(mod, name)
    table["triple_product"] = lambda **kw: TripleProductResult()
    return table


#: Welche Einträge aus `data` als Positionsargumente an das Verfahren gehen.
_ARGS: dict[str, list[str]] = {
    "slope": ["P1", "P2"],
    "line_through": ["P1", "P2"],
    "secant": ["f", "x1", "x2"],
    "secant_h": ["f", "x0", "h"],
    "tangent": ["f", "x0"],
    "tangent_slope": ["f", "x0"],
    "derivative_by_definition": ["f"],
    "limit": ["f", "x0"],
    "limit_infinity": ["f"],
    "continuity": ["f", "x0"],
    "diff": ["f"],
    "diff_at": ["f", "x0"],
    "diff_product": ["f", "g"],
    "diff_quotient": ["f", "g"],
    "diff_chain": ["outer", "inner"],
    "diff_inverse": ["f"],
    "rule_table": [],
    "monotony": ["f"],
    "critical_points": ["f"],
    "curvature": ["f"],
    "extrema": ["f"],
    "inflection_points": ["f"],
    "curve_discussion": ["f"],
    "triple_product": [],
}


def loese(key: str, **kw):
    """
    Rechnet eine Aufgabe durch und gibt das Ergebnis von anakit zurück.

    Gedacht zum Kontrollieren, nachdem du selbst gerechnet hast — der
    Rechenweg ist dieselbe Abfolge, die auf dem Papier stünde.

        loese("SW1.4")                       # Sekante mit x0 = 1, h = 1/2
        loese("SW1.5a", variant="h")         # andere Variante des Grenzwerts
        loese("SW2.2", decimal=True)         # Dezimalzahlen statt Brüchen

    Weitere Schlüsselwörter überschreiben die Optionen der Aufgabe und gehen
    an das Verfahren. Mit `decimal=True` wird das Ergebnis dezimal angezeigt.
    """
    ex = aufgabe(key)
    if isinstance(ex, Reading):
        raise ValueError(
            f"{ex.key} ist eine Leseliste, keine Rechenaufgabe. Die Aufgaben "
            "stehen in Papula bzw. Hughes-Hallet; sie sind hier nicht abgedruckt "
            "und haben keine Lösung im Paket. aufgabe('" + ex.key + "') zeigt die Seiten.")
    decimal = kw.pop("decimal", False)
    table = _lookup()
    if ex.method not in table or ex.method not in _ARGS:
        raise ValueError(f"Für '{ex.method}' gibt es noch keinen Rechenweg")
    args = [ex.data[name] for name in _ARGS[ex.method]]
    options = dict(ex.data.get("kwargs", {}))
    options.update(kw)
    result = table[ex.method](*args, **options)
    if decimal and hasattr(result, "as_decimal"):
        result.as_decimal()
    return result


def loese_alle(week: str | None = None) -> dict:
    """
    Rechnet alle Rechenaufgaben (einer Woche) durch — gedacht für Tests.

    Gibt {Schlüssel: Ergebnis} zurück. Wirft den ersten Fehler mit dem
    Aufgabenschlüssel in der Meldung weiter.
    """
    out = {}
    for key in liste(week):
        if isinstance(EXERCISES[key], Reading):
            continue
        try:
            out[key] = loese(key)
        except Exception as exc:  # pragma: no cover - nur für die Fehlersuche
            raise RuntimeError(f"{key}: {type(exc).__name__}: {exc}") from exc
    return out
