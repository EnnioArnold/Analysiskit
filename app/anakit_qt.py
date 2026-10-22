#!/usr/bin/env python3
"""
anakit — Analysis mit Rechenweg, als Desktop-App im Apple-Stil (Qt).

Gebaut zum Modul ANA-G der HSLU (Semesterwochen 1 bis 6). Die App rechnet
mit demselben Paket `anakit` wie das Notebook; sie ist nur eine andere
Oberfläche dafür. Sechs Seiten:

    Steigung & Tangente   Sekanten, Tangente, Ableitung aus der Definition
    Grenzwerte            Grenzwert, Stetigkeit, Wertetabelle
    Ableiten              Ableitungsregeln, Einzelregeln, Eigene Ableitung prüfen
    Kurvendiskussion      Monotonie, kritische Punkte, Krümmung, Extrema
    Numerik               Bisektion, Newton-Verfahren, numerisch Differenzieren
    Höhere Ableitungen    Taylor, MacLaurin, Sonderfälle, Funktionsüberblick

Jede Seite zeigt ihr Ergebnis als Karte mit drei Ansichten: Rechenweg
(gesetzte Formeln), Zeichnung und Animation. Die Prüfung mit Wolfram|Alpha
ist optional und braucht eine kostenlose AppID von
https://developer.wolframalpha.com/ (Umgebungsvariable WOLFRAM_APPID oder
einmalige Eingabe in der App).

Installation:
    pip install PySide6 matplotlib sympy numpy        # plotly, manimgx freiwillig

Start (aus dem Ordner heraus, egal wie er heisst):
    python3 app/anakit_qt.py
"""

from __future__ import annotations

import base64
import html
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

# Matplotlib ohne eigenes Fenster: die Figuren entstehen im Hintergrund,
# gezeigt werden sie von der Qt-Leinwand der App.
try:
    import matplotlib
    matplotlib.use("agg")
except Exception:                                            # pragma: no cover
    pass

from PySide6.QtCore import (QBuffer, QByteArray, QEasingCurve, QIODevice, QObject,
                            QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer,
                            QUrl, Signal)
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontInfo, QFontMetricsF,
                           QGuiApplication, QImage, QMovie, QPainter, QPen)
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame,
                               QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMenu, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget, QVBoxLayout, QWidget)

# Dieser Ordner ist selbst das Paket `anakit` — auch dann, wenn er nach dem
# GitHub-Download »anakit-main« heisst oder man `python3 app/anakit_qt.py`
# aufruft. Darum hier von Hand als Paket anmelden, vor allen Importen.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "anakit" not in sys.modules:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "anakit", os.path.join(_HERE, "__init__.py"),
        submodule_search_locations=[_HERE])
    _pkg = _ilu.module_from_spec(_spec)
    sys.modules["anakit"] = _pkg
    _spec.loader.exec_module(_pkg)

import sympy as sp                                            # noqa: E402

import anakit as ak                                           # noqa: E402
from anakit.core import fmt, parse as ak_parse, num as ak_num  # noqa: E402
from anakit.render import tex_any, text_any, tex_to_text      # noqa: E402

# Die Übungsaufgaben und die Animation entstehen getrennt und sind freiwillig.
try:
    from anakit.exercises import aufgabe as ex_aufgabe, liste as ex_liste, loese as ex_loese
    HAS_EXERCISES = True
except Exception:                                             # pragma: no cover
    ex_aufgabe = ex_liste = ex_loese = None
    HAS_EXERCISES = False

try:
    from anakit import anim as ak_anim
    ANIM_IMPORT_ERROR = ""
except Exception as _exc:                                     # pragma: no cover
    ak_anim = None
    ANIM_IMPORT_ERROR = str(_exc)

# QtMultimedia spielt das MP4 ab; fehlt es, fällt der Abspieler auf das GIF zurück.
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer  # noqa: F401
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_VIDEO = True
except Exception:                                             # pragma: no cover
    HAS_VIDEO = False

MINUS = "−"
x = ak.x


# ==========================================================================
# Eingaben lesen und Fehler freundlich sagen
# ==========================================================================
#
# Alles in diesem und den folgenden Abschnitten bis »Farben und Schriften«
# kommt ohne Fenster aus und lässt sich im Test direkt aufrufen.

class InputError(ValueError):
    """Eine Eingabe, die der Mensch korrigieren kann — mit fertigem Satz."""


SYNTAX_HELP = ("Erlaubt sind x^2, 3x, sin(x), cos(x), tan(x), ln(x), e^x, sqrt(x), "
               "abs(x), pi und Brüche wie 1/x. Abschnittsweise: x^2 if x<1 else 2-x.")

PIECEWISE_HELP = ("Abschnittsweise Funktionen: »x^2 if x<1 else 2-x« heisst x² für "
                  "x < 1, sonst 2 − x. Mehrere Äste: »x if x<0 else 1 if x<2 else 3-x«. "
                  "Betrag: abs(x), Vorzeichen: sgn(x).")


def read_function(text: str, what: str = "die Funktion") -> tuple[str, sp.Expr]:
    """
    Liest f(x) aus dem Textfeld. Gibt (bereinigter Text, Term) zurück.

    Fehler werden zu einem `InputError` mit einem Satz, der sagt, was
    nicht stimmt — Syntaxfehler, unbekannte Variablen, Division durch 0.
    """
    raw = (text or "").strip()
    if not raw:
        raise InputError(f"Bitte {what} eingeben, zum Beispiel x^2 - 3x + 1.")
    cleaned = re.sub(r"^\s*[fgh]\s*\(\s*x\s*\)\s*=\s*", "", raw)    # »f(x) = …« erlauben
    if "=" in cleaned and not re.search(r"[<>!]=|==", cleaned):
        raise InputError(f"»{raw}« enthält ein Gleichheitszeichen. Nur den Term "
                         "eingeben, etwa x^2 - 3x, nicht eine Gleichung.")
    try:
        expr = ak_parse(cleaned)
    except ZeroDivisionError:
        raise InputError(f"»{raw}« enthält eine Division durch 0.") from None
    except RecursionError:
        raise InputError("Der Term ist zu verschachtelt zum Einlesen.") from None
    except Exception:
        raise InputError(f"»{raw}« lässt sich nicht lesen. {SYNTAX_HELP}") from None
    if not isinstance(expr, sp.Basic):
        raise InputError(f"»{raw}« lässt sich nicht lesen. {SYNTAX_HELP}")
    if expr.has(sp.zoo, sp.nan):
        raise InputError(f"»{raw}« enthält eine Division durch 0 und ist darum "
                         "an jeder Stelle nicht definiert.")
    extra = sorted(str(s) for s in expr.free_symbols if s != x)
    if extra:
        raise InputError(f"In »{raw}« kommt {'die Variable' if len(extra) == 1 else 'kommen die Variablen'} "
                         f"{', '.join(extra)} vor. Erlaubt ist nur x; Zahlen wie "
                         "pi und e sind schon bekannt.")
    unknown = sorted({str(f.func) for f in expr.atoms(sp.core.function.AppliedUndef)})
    if unknown:
        raise InputError(f"Die Funktion {', '.join(unknown)}( … ) kenne ich nicht. {SYNTAX_HELP}")
    return cleaned, expr


def read_number(text: str, what: str = "die Zahl", infinity: bool = False) -> sp.Expr:
    """Eine Zahl lesen: 3, -1/2, 0.25, sqrt(2), pi/2 — und mit infinity auch oo."""
    raw = (text or "").strip()
    if not raw:
        raise InputError(f"Bitte {what} eingeben.")
    norm = raw.replace("∞", "oo").replace(MINUS, "-").replace(",", ".")
    if re.fullmatch(r"\+?\s*(oo|inf|infty|unendlich)", norm, re.I):
        if infinity:
            return sp.oo
        raise InputError(f"{what[0].upper() + what[1:]} muss eine endliche Zahl sein.")
    if re.fullmatch(r"-\s*(oo|inf|infty|unendlich)", norm, re.I):
        if infinity:
            return -sp.oo
        raise InputError(f"{what[0].upper() + what[1:]} muss eine endliche Zahl sein.")
    try:
        value = ak_num(norm)
    except ZeroDivisionError:
        raise InputError(f"»{raw}« ist eine Division durch 0.") from None
    except Exception:
        raise InputError(f"»{raw}« ist keine Zahl. Erlaubt sind 3, −1/2, 0.25, "
                         "sqrt(2) oder pi/2.") from None
    if not isinstance(value, sp.Basic) or not value.is_number or value.has(sp.zoo, sp.nan):
        raise InputError(f"»{raw}« ist keine Zahl. Erlaubt sind 3, −1/2, 0.25, "
                         "sqrt(2) oder pi/2.")
    if value.is_real is False:
        raise InputError(f"»{raw}« ist keine reelle Zahl.")
    if isinstance(value, sp.Float):
        value = sp.nsimplify(value, rational=True)
    return value


def friendly_error(exc: BaseException) -> str:
    """Macht aus einer Ausnahme einen Satz, den man lesen will."""
    if isinstance(exc, InputError):
        return str(exc)
    if isinstance(exc, ZeroDivisionError):
        return ("Hier wird durch 0 geteilt. Die Rechnung ist an dieser Stelle "
                "nicht definiert — eine andere Stelle wählen oder bei Grenzwerten "
                "den Grenzwert statt des Funktionswerts nehmen.")
    if isinstance(exc, RecursionError):
        return "Der Term ist zu verschachtelt: die Rechnung wurde abgebrochen."
    if isinstance(exc, (SyntaxError, sp.SympifyError)):
        return f"Der Term lässt sich nicht lesen. {SYNTAX_HELP}"
    if isinstance(exc, NotImplementedError):
        return ("Das kann anakit für diese Eingabe nicht rechnen"
                + (f": {exc}" if str(exc) else ".")
                + " Eine einfachere Funktion oder einen kleineren Bereich probieren.")
    if isinstance(exc, NameError) and "DiracDelta" in str(exc):
        return ("Diese Funktion ist an einer Stelle nicht differenzierbar (etwa ein "
                "Betrag oder Knick). Für die Kurvendiskussion bitte abschnittsweise "
                "schreiben, z. B. »x if x>=0 else -x«, oder den Bereich so wählen, "
                "dass die Knickstelle nicht darin liegt.")
    if isinstance(exc, ValueError) and str(exc):
        return str(exc)
    return (f"Das hat nicht geklappt ({type(exc).__name__}: {exc}). Bitte die "
            "Eingabe prüfen — ein einfacheres Beispiel hilft herauszufinden, was stört.")


# ==========================================================================
# Die Rechnungen der sechs Seiten
# ==========================================================================
#
# Jede Seite hat ein Wörterbuch `p` mit den Eingaben als Text; `calc(seite, p)`
# gibt das anakit-Ergebnis zurück oder wirft einen InputError.

PAGE_KEYS = ("steigung", "limits", "derivative", "curve", "numerik", "hoeher")

STEIGUNG_METHODS = [("sek12", "Sekante", "Sekante durch die Stellen x₁ und x₂"),
                    ("sekh", "Sekante (h)", "Differenzenquotient an x₀ mit Abstand h"),
                    ("tan", "Tangente", "Tangentensteigung f′(x₀) und Tangentengleichung"),
                    ("def", "Ableitung", "Ableitungsfunktion f′(x) aus der Definition"),
                    ("line", "Gerade", "Gerade durch zwei Punkte, ohne Funktion")]

LIMIT_ACTIONS = [("limit", "Grenzwert", "Grenzwert von f für x → x₀"),
                 ("cont", "Stetigkeit", "Stetigkeit an der Stelle x₀ nach Definition"),
                 ("contall", "Stetig (ganz)", "Stetigkeit der ganzen Funktion: alle Verdachtsstellen prüfen"),
                 ("table", "Wertetabelle", "Wertetabelle f(x₀ ± 10⁻ᵏ) als Beleg")]

LIMIT_SIDES = [("both", "beidseitig"), ("-", "links"), ("+", "rechts")]

DERIVATIVE_ACTIONS = [("diff", "Ableiten", "Ableitung mit den Regeln, Schritt für Schritt"),
                      ("at", "An Stelle", "Ableitung ableiten und danach die Stelle x₀ einsetzen"),
                      ("rule", "Einzelregel", "Eine Regel an einem Beispiel vorführen"),
                      ("table", "Regeltabelle", "Die Regeln von Newton und Leibniz auf einen Blick"),
                      ("check", "Prüfen", "Die selbst gerechnete Ableitung prüfen")]

RULES = [("produkt", "Produktregel", "g(x)", "x, e^x"),
         ("quotient", "Quotientenregel", "g(x)", "x, e^x"),
         ("kette", "Kettenregel", "innere Funktion u(x)", "ln(x), x^2+1"),
         ("umkehr", "Umkehrregel", "Umkehrfunktion (frei lassen = automatisch)", "ln(x)"),
         ("summe", "Summenregel", "g(x)", "x^2, sin(x)"),
         ("differenz", "Differenzregel", "g(x)", "x^2, sin(x)"),
         ("faktor", "Faktorregel", "Faktor c", "x^2, 3"),
         ("summand", "Konstanter Summand", "Konstante c", "x^2, 7"),
         ("potenz", "Potenzregel", "", "x^3")]

CURVE_ACTIONS = [("all", "Alles", "Ganze Kurvendiskussion mit Zusammenfassung"),
                 ("mono", "Monotonie", "Monotonie über das Vorzeichen von f′"),
                 ("crit", "Kritisch", "Kritische Punkte: f′(x₀) = 0"),
                 ("curv", "Krümmung", "Krümmungsverhalten über f″"),
                 ("ext", "Extrema", "Lokale Extrema mit dem Kriterium der zweiten Ableitung"),
                 ("infl", "Wendepunkte", "Wendepunkte: f″(x₀) = 0, f‴(x₀) ≠ 0")]

NUMERIK_ACTIONS = [("bisect", "Bisektion", "Nullstelle durch Halbieren des Intervalls [a, b]"),
                   ("newton", "Newton", "Nullstelle mit dem Newton-Verfahren aus einem Startwert"),
                   ("numdiff", "Differenz", "f′(x₀) numerisch: vorwärts, rückwärts oder zentral"),
                   ("errtab", "Fehler", "Fehler der Differenz für h = 10⁻¹ … 10⁻¹⁶ (optimale Schrittweite)")]

DIFF_KINDS = [("vorwaerts", "vorwärts"), ("rueckwaerts", "rückwärts"), ("zentral", "zentral")]

HOEHER_ACTIONS = [("taylor", "Taylor", "Taylor-Polynom n-ten Grades um die Stelle x₀"),
                  ("mac", "MacLaurin", "MacLaurin-Polynom n-ten Grades (x₀ = 0)"),
                  ("higher", "Sonderfall", "Erste nichtverschwindende Ableitung an x₀ und der Schluss daraus"),
                  ("basics", "Überblick", "Definitionsbereich, Symmetrie, Nullstellen, Pole, Asymptoten")]

#: Beispiele je Seite: (Name, Eingaben als Wörterbuch). Wer eine Seite
#: öffnet, kann sie durchklicken — und der Test rechnet jede einzelne.
EXAMPLES: dict[str, list[tuple[str, dict]]] = {
    "steigung": [
        ("Sekante von x² durch −1 und 3", dict(f="x^2", method="sek12", x1="-1", x2="3")),
        ("Sekante von x² an 1 mit h = 1/2", dict(f="x^2", method="sekh", x0="1", h="1/2")),
        ("Tangente von 1/x bei 1", dict(f="1/x", method="tan", x0="1")),
        ("Tangente von x³ − 3x bei 2 (h-Form)", dict(f="x^3-3x", method="tan", x0="2", variant="h")),
        ("Ableitung von x³ aus der Definition", dict(f="x^3", method="def")),
        ("Ableitung von 1/x aus der Definition", dict(f="1/x", method="def")),
        ("Ableitung von sin(x) aus der Definition", dict(f="sin(x)", method="def")),
        ("Gerade durch (−1|2) und (2|4)", dict(method="line", p1x="-1", p1y="2", p2x="2", p2y="4")),
    ],
    "limits": [
        ("(x²−1)/(x−1) für x → 1", dict(f="(x^2-1)/(x-1)", x0="1", action="limit", side="both")),
        ("1/x von rechts an 0", dict(f="1/x", x0="0", action="limit", side="+")),
        ("sin(x)/x für x → 0", dict(f="sin(x)/x", x0="0", action="limit", side="both")),
        ("(2x²+1)/(x²−3) für x → ∞", dict(f="(2x^2+1)/(x^2-3)", x0="oo", action="limit", side="both")),
        ("tan(x) von links an π/2", dict(f="tan(x)", x0="pi/2", action="limit", side="-")),
        ("Abschnittsweise f stetig bei 1?", dict(f="x^2 if x<1 else 2-x", x0="1", action="cont", side="both")),
        ("Sprung: |x|/x bei 0", dict(f="abs(x)/x", x0="0", action="cont", side="both")),
        ("Stetigkeit insgesamt: 1/(x−2)", dict(f="1/(x-2)", x0="0", action="contall", side="both")),
        ("Wertetabelle (e^x−1)/x bei 0", dict(f="(e^x-1)/x", x0="0", action="table", side="both")),
    ],
    "derivative": [
        ("Kettenregel: ln(x²+1)", dict(f="ln(x^2+1)", n="1", action="diff")),
        ("Produktregel: x·e^x", dict(f="x*e^x", n="1", action="diff")),
        ("Quotientenregel: x/(x²+1)", dict(f="x/(x^2+1)", n="1", action="diff")),
        ("Zweite Ableitung: x⁴ − 2x²", dict(f="x^4-2x^2", n="2", action="diff")),
        ("f′(2) für x³ − 3x", dict(f="x^3-3x", n="1", action="at", x0="2")),
        ("Produktregel vorführen", dict(f="x", n="1", action="rule", rule="produkt", g="e^x")),
        ("Quotientenregel vorführen", dict(f="x", n="1", action="rule", rule="quotient", g="e^x")),
        ("Kettenregel vorführen", dict(f="ln(x)", n="1", action="rule", rule="kette", g="x^2+1")),
        ("Umkehrregel vorführen", dict(f="ln(x)", n="1", action="rule", rule="umkehr", g="")),
        ("Regeltabelle (Newton/Leibniz)", dict(f="x", n="1", action="table")),
        ("Eigene Ableitung prüfen (falsch)", dict(f="x*e^x", n="1", action="check", claimed="e^x")),
        ("Eigene Ableitung prüfen (richtig)", dict(f="x*e^x", n="1", action="check", claimed="e^x + x*e^x")),
    ],
    "curve": [
        ("x⁵ − 7x⁴ + 10x³ + 18x² − 27x − 27", dict(f="x^5-7x^4+10x^3+18x^2-27x-27", a="-1.5", b="4.5", action="all")),
        ("x³ − 3x", dict(f="x^3-3x", a="", b="", action="all")),
        ("x⁴ − 2x² (Extrema)", dict(f="x^4-2x^2", a="", b="", action="ext")),
        ("x·e^x (Monotonie)", dict(f="x*e^x", a="", b="", action="mono")),
        ("x²/(x−1) (kritische Punkte)", dict(f="x^2/(x-1)", a="", b="", action="crit")),
        ("sin(x) auf [0, 2π] (Krümmung)", dict(f="sin(x)", a="0", b="2*pi", action="curv")),
        ("x³ (Sattelpunkt, Wendepunkte)", dict(f="x^3", a="", b="", action="infl")),
    ],
    # SW5: die ersten vier stammen aus den Notebooks der Vorlesung
    "numerik": [
        ("Newton: x² − 1.5 ab 3", dict(f="x^2-1.5", action="newton", x0="3")),
        ("Bisektion: x² − 1.5 auf [1, 4]", dict(f="x^2-1.5", action="bisect", a="1", b="4", tol="1e-6")),
        ("Zentrale Differenz: x³ − 5 bei 1", dict(f="x^3-5", action="numdiff", x0="1", h="1e-6", kind="zentral")),
        ("Newton mit numerischer Ableitung: x³ − 5", dict(f="x^3-5", action="newton", x0="3", numerisch="1")),
        ("Vorwärtsdifferenz: e^x bei 1 (optimales h)", dict(f="e^x", action="numdiff", x0="1", h="", kind="vorwaerts")),
        ("Fehler und Schrittweite: e^x bei 1", dict(f="e^x", action="errtab", x0="1", kind="vorwaerts")),
    ],
    # SW6: eigene Beispiele zum Stoff (die Folien rechnen keine vor)
    "hoeher": [
        ("MacLaurin von e^x, Grad 4", dict(f="e^x", action="mac", n="4")),
        ("MacLaurin von sin(x), Grad 5", dict(f="sin(x)", action="mac", n="5")),
        ("Taylor von ln(x) um 1, Grad 3", dict(f="ln(x)", action="taylor", x0="1", n="3")),
        ("Sonderfall x⁴ bei 0 (Minimum)", dict(f="x^4", action="higher", x0="0")),
        ("Sonderfall x³ bei 0 (Sattelpunkt)", dict(f="x^3", action="higher", x0="0")),
        ("Überblick x²/(x−1)", dict(f="x^2/(x-1)", action="basics")),
        ("Überblick (2x²+1)/(x²−3)", dict(f="(2x^2+1)/(x^2-3)", action="basics")),
    ],
}

DEFAULTS: dict[str, dict] = {
    "steigung": dict(f="x^2", method="sek12", x1="-1", x2="3", x0="1", h="1/2",
                     variant="x", p1x="-1", p1y="2", p2x="2", p2y="4"),
    "limits": dict(f="(x^2-1)/(x-1)", x0="1", action="limit", side="both"),
    "derivative": dict(f="ln(x^2+1)", n="1", action="diff", x0="1", rule="produkt",
                       g="e^x", claimed=""),
    "curve": dict(f="x^3-3x", a="", b="", action="all"),
    "numerik": dict(f="x^2-1.5", action="newton", a="1", b="4", tol="1e-6", x0="3", h="",
                    kind="zentral", numerisch="0"),
    "hoeher": dict(f="e^x", action="mac", x0="0", n="4"),
}


def _val(p: dict, key: str, what: str, infinity: bool = False):
    return read_number(p.get(key, ""), what, infinity=infinity)


def _domain(p: dict):
    """Der optionale Bereich [a, b] der Kurvendiskussion, oder None."""
    ta, tb = (p.get("a") or "").strip(), (p.get("b") or "").strip()
    if not ta and not tb:
        return None
    lo = _val(p, "a", "die linke Grenze a") if ta else None
    hi = _val(p, "b", "die rechte Grenze b") if tb else None
    if lo is not None and hi is not None and float(lo) >= float(hi):
        raise InputError("Für den Bereich muss a kleiner als b sein.")
    return (lo, hi)


def calc_steigung(p: dict):
    method = p.get("method", "sek12")
    if method == "line":
        P1 = (_val(p, "p1x", "x₁"), _val(p, "p1y", "y₁"))
        P2 = (_val(p, "p2x", "x₂"), _val(p, "p2y", "y₂"))
        if P1 == P2:
            raise InputError("Die beiden Punkte sind gleich — durch einen Punkt "
                             "gehen unendlich viele Geraden.")
        return ak.line_through(P1, P2)
    _, f = read_function(p.get("f", ""))
    if method == "sek12":
        a, b = _val(p, "x1", "die Stelle x₁"), _val(p, "x2", "die Stelle x₂")
        if a == b:
            raise InputError("Die Stellen x₁ und x₂ müssen verschieden sein, sonst "
                             "wird beim Differenzenquotienten durch 0 geteilt.")
        return ak.secant(f, a, b)
    if method == "sekh":
        a, hv = _val(p, "x0", "die Stelle x₀"), _val(p, "h", "den Abstand h")
        if hv == 0:
            raise InputError("h = 0 ist nicht erlaubt: der Differenzenquotient "
                             "teilt durch h.")
        return ak.secant_h(f, a, hv)
    if method == "tan":
        a = _val(p, "x0", "die Stelle x₀")
        return ak.tangent(f, a, "h" if p.get("variant") == "h" else "x")
    if method == "def":
        return ak.derivative_by_definition(f)
    raise InputError("Unbekannte Methode.")


def calc_limits(p: dict):
    action = p.get("action", "limit")
    if action == "contall":
        _, f = read_function(p.get("f", ""))
        return ak.is_continuous(f)
    _, f = read_function(p.get("f", ""))
    x0 = _val(p, "x0", "die Stelle x₀", infinity=True)
    side = p.get("side", "both")
    if action == "limit":
        return ak.limit(f, x0, side)
    if action == "cont":
        if x0 in (sp.oo, -sp.oo):
            raise InputError("Stetigkeit gibt es nur an einer endlichen Stelle; "
                             "für x → ∞ den Grenzwert nehmen.")
        return ak.continuity(f, x0)
    if action == "table":
        return ak.limit_table(f, x0, side)
    raise InputError("Unbekannte Aktion.")


def calc_derivative(p: dict):
    action = p.get("action", "diff")
    if action == "table":
        return ak.rule_table()
    if action == "rule":
        name = p.get("rule", "produkt")
        ftxt, gtxt = (p.get("f") or "").strip(), (p.get("g") or "").strip()
        f = read_function(ftxt, "die erste Funktion")[1] if ftxt else None
        if name in ("faktor", "summand"):
            g = _val(p, "g", "die Konstante c") if gtxt else None
        elif name == "umkehr":
            g = read_function(gtxt, "die Umkehrfunktion")[1] if gtxt else None
        elif name == "potenz":
            g = None
        else:
            g = read_function(gtxt, "die zweite Funktion")[1] if gtxt else None
        return ak.rule(name, f, g)
    _, f = read_function(p.get("f", ""))
    if action == "diff":
        try:
            n = int(float(p.get("n", "1")))
        except ValueError:
            raise InputError("Die Ordnung n muss eine ganze Zahl sein.") from None
        if not 1 <= n <= 6:
            raise InputError("Die Ordnung n muss zwischen 1 und 6 liegen.")
        return ak.diff(f, n)
    if action == "at":
        return ak.diff_at(f, _val(p, "x0", "die Stelle x₀"))
    if action == "check":
        ctext = (p.get("claimed") or "").strip()
        if not ctext:
            raise InputError("Bitte deine Ableitung eingeben, zum Beispiel e^x + x*e^x.")
        return ak.check_derivative(f, read_function(ctext, "deine Ableitung")[1])
    raise InputError("Unbekannte Aktion.")


def calc_curve(p: dict):
    _, f = read_function(p.get("f", ""))
    dom = _domain(p)
    fn = {"all": ak.curve_discussion, "mono": ak.monotony, "crit": ak.critical_points,
          "curv": ak.curvature, "ext": ak.extrema, "infl": ak.inflection_points}
    action = p.get("action", "all")
    if action not in fn:
        raise InputError("Unbekannte Aktion.")
    return fn[action](f, dom)


def _pos_float(p: dict, key: str, what: str, lo: float = 1e-15) -> float:
    """Eine positive Zahl als float lesen (Toleranz, Schrittweite)."""
    v = float(_val(p, key, what))
    if not v >= lo:
        raise InputError(f"{what[0].upper() + what[1:]} muss mindestens {lo:g} sein.")
    return v


def calc_numerik(p: dict):
    action = p.get("action", "newton")
    _, f = read_function(p.get("f", ""))
    if action == "bisect":
        a, b = _val(p, "a", "die linke Grenze a"), _val(p, "b", "die rechte Grenze b")
        if not float(a) < float(b):
            raise InputError("Für die Bisektion muss a kleiner als b sein.")
        return ak.bisection(f, a, b, _pos_float(p, "tol", "die Toleranz"))
    kind = p.get("kind", "zentral")
    if kind not in dict(DIFF_KINDS):
        raise InputError("Unbekannte Differenz.")
    x0 = _val(p, "x0", "die Stelle x₀" if action != "newton" else "den Startwert x₀")
    if action == "newton":
        return ak.newton(f, x0, _pos_float(p, "tol", "die Toleranz"),
                         numerisch=p.get("numerisch") == "1")
    if action == "numdiff":
        h_text = (p.get("h") or "").strip()
        hv = _pos_float(p, "h", "die Schrittweite h", 1e-300) if h_text else None
        return ak.numeric_diff(f, x0, hv, kind)
    if action == "errtab":
        return ak.diff_error_table(f, x0, kind)
    raise InputError("Unbekannte Aktion.")


def calc_hoeher(p: dict):
    action = p.get("action", "mac")
    _, f = read_function(p.get("f", ""))
    if action == "basics":
        return ak.curve_basics(f)
    if action == "higher":
        return ak.higher_order_test(f, _val(p, "x0", "die Stelle x₀"))
    try:
        n = int(float(p.get("n", "")))
    except ValueError:
        raise InputError("Der Grad n muss eine ganze Zahl sein.") from None
    if not 0 <= n <= 15:
        raise InputError("Der Grad n muss zwischen 0 und 15 liegen.")
    if action == "mac":
        return ak.maclaurin(f, n)
    if action == "taylor":
        return ak.taylor(f, _val(p, "x0", "die Stelle x₀"), n)
    raise InputError("Unbekannte Aktion.")


CALC = {"steigung": calc_steigung, "limits": calc_limits,
        "derivative": calc_derivative, "curve": calc_curve,
        "numerik": calc_numerik, "hoeher": calc_hoeher}


def calc(page: str, p: dict):
    """Rechnet eine Seite. Wirft InputError mit freundlichem Satz bei Fehlern."""
    try:
        return CALC[page](p)
    except InputError:
        raise
    except Exception as exc:                  # SymPy und anakit werfen vielerlei
        raise InputError(friendly_error(exc)) from exc


def figure_names(page: str, p: dict) -> list[str]:
    """Die Zeichnungen, zwischen denen die Seite umschalten kann."""
    if page == "steigung" and p.get("method") == "tan":
        return ["Tangente", "Sekanten → Tangente"]
    return ["Zeichnung"]


def make_figure(page: str, p: dict, result, dark: bool = False, variant: int = 0):
    """
    Die Matplotlib-Figur zur Rechnung. Wirft NotImplementedError, wenn es zu
    dieser Rechnung keine Zeichnung gibt (zum Beispiel die Regeltabelle).
    """
    if page == "steigung" and variant == 1 and p.get("method") == "tan":
        return ak.plot_secants_to_tangent(result.f, float(result.x0), dark=dark)
    if page == "derivative" and p.get("action") == "check":
        return ak.plot_derivative(result.f, dark=dark)
    if page == "derivative" and p.get("action") == "table":
        raise NotImplementedError("Die Regeltabelle hat keine Zeichnung.")
    if page == "curve":
        dom = _domain(p)
        if dom and dom[0] is not None and dom[1] is not None:
            return result.figure(xlim=(float(dom[0]), float(dom[1])), dark=dark)
    return result.figure(dark=dark)


# ==========================================================================
# Wolfram|Alpha
# ==========================================================================

def ssl_context():
    """
    Der Zertifikatsspeicher für die Anfragen.

    Ein mit pyenv gebautes Python benutzt den Speicher der OpenSSL-Version,
    gegen die es gebaut wurde — der ist auf macOS oft leer. Ist das Paket
    certifi vorhanden, nehmen wir dessen Zertifikate; das ist derselbe Satz,
    den auch Browser verwenden, und erspart jede Bastelei am System.
    """
    import ssl
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()      # bewusst gesetzt, respektieren
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


WOLFRAM_URL = "https://api.wolframalpha.com/v2/query"
WOLFRAM_SITE = "https://www.wolframalpha.com/input"
APPID_FILE = Path.home() / ".anakit_wolfram_appid"
#: Wer schon eine AppID für lialkit hinterlegt hat, braucht sie nicht noch einmal.
APPID_FALLBACK = Path.home() / ".gauss_wolfram_appid"
USER_AGENT = "anakit-app/1.0"
_CACHE: dict[str, str] = {}


def load_appid() -> str:
    key = os.environ.get("WOLFRAM_APPID", "").strip()
    if key:
        return key
    for path in (APPID_FILE, APPID_FALLBACK):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return ""


def save_appid(key: str) -> None:
    APPID_FILE.write_text(key.strip(), encoding="utf-8")
    try:
        APPID_FILE.chmod(0o600)
    except OSError:
        pass


def browser_url(query: str) -> str:
    return f"{WOLFRAM_SITE}?i={urllib.parse.quote(query)}"


def explain_network_error(reason) -> str:
    """
    Macht aus dem technischen Grund einer fehlgeschlagenen Anfrage einen
    Satz, der sagt, woran es liegt und was zu tun ist.
    """
    import ssl
    import socket

    text = str(reason)
    if isinstance(reason, ssl.SSLCertVerificationError) or \
            "CERTIFICATE_VERIFY_FAILED" in text:
        try:
            import certifi                                    # noqa: F401
            return ("Das Sicherheitszertifikat liess sich nicht prüfen, obwohl "
                    "certifi vorhanden ist. Vermutlich hängt ein Netzwerkfilter "
                    "dazwischen, der den Verkehr aufbricht — probier ein "
                    "anderes Netz, etwa den Hotspot vom Handy.")
        except ImportError:
            return ("Das Sicherheitszertifikat liess sich nicht prüfen: diesem "
                    "Python fehlen die Wurzelzertifikate. „pip install certifi“ "
                    "ausführen und die App neu starten — sie nimmt die "
                    "Zertifikate dann von dort.")
    if isinstance(reason, socket.gaierror) or "Name or service not known" in text \
            or "nodename nor servname" in text:
        return ("Der Server api.wolframalpha.com war nicht auffindbar. Das "
                "deutet auf eine fehlende Internetverbindung oder einen "
                "blockierenden DNS hin.")
    if "Tunnel connection failed" in text or "proxy" in text.lower():
        return ("Ein Proxy im Netzwerk hat die Anfrage abgewiesen. In einem "
                "Firmen- oder Schulnetz hilft meist ein anderes Netz, etwa der "
                "Hotspot vom Handy.")
    if isinstance(reason, ConnectionRefusedError) or "refused" in text.lower():
        return ("Die Verbindung wurde abgewiesen — vermutlich blockiert eine "
                "Firewall oder ein Netzwerkfilter den Zugriff.")
    if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in text:
        return ("Zeitüberschreitung beim Verbinden. Entweder ist die Leitung "
                "sehr langsam oder etwas blockiert die Anfrage.")
    return f"Keine Verbindung zu Wolfram|Alpha: {text}"


def cert_source() -> str:
    """Woher die Zertifikate kommen — gehört in die Diagnose."""
    if os.environ.get("SSL_CERT_FILE"):
        return "Zertifikat geprüft (SSL_CERT_FILE)."
    try:
        import certifi                                        # noqa: F401
        return "Zertifikat geprüft (certifi)."
    except ImportError:
        return "Zertifikat geprüft (Systemspeicher)."


def network_selftest(appid: str = "", timeout: float = 20) -> list[tuple[str, bool, str]]:
    """
    Prüft der Reihe nach: Namensauflösung, TLS-Verbindung, Antwort der API.
    Gibt je Schritt (Name, geklappt, Erläuterung) zurück.
    """
    import socket

    out: list[tuple[str, bool, str]] = []
    host = "api.wolframalpha.com"

    try:
        address = socket.gethostbyname(host)
        out.append(("Namensauflösung", True, f"{host} → {address}"))
    except OSError as exc:
        out.append(("Namensauflösung", False, explain_network_error(exc)))
        return out

    try:
        req = urllib.request.Request(f"https://{host}/",
                                     headers={"User-Agent": USER_AGENT})
        urllib.request.urlopen(req, timeout=timeout,
                               context=ssl_context()).read(64)
        out.append(("Verschlüsselte Verbindung", True, cert_source()))
    except urllib.error.HTTPError:
        # Eine Fehlerseite ist auch eine Antwort — die Leitung steht.
        out.append(("Verschlüsselte Verbindung", True, cert_source()))
    except urllib.error.URLError as exc:
        out.append(("Verschlüsselte Verbindung", False,
                    explain_network_error(exc.reason)))
        return out
    except OSError as exc:
        out.append(("Verschlüsselte Verbindung", False,
                    explain_network_error(exc)))
        return out

    if not appid:
        out.append(("AppID", False, "Noch keine AppID hinterlegt."))
        return out

    try:
        answer = wolfram_text(appid, "2+2", timeout=timeout)
        out.append(("Testabfrage 2+2", True, f"Antwort: {answer.splitlines()[0]}"))
    except RuntimeError as exc:
        out.append(("Testabfrage 2+2", False, str(exc)))
    return out


def wolfram_text(appid: str, query: str, timeout: float = 30) -> str:
    """Fragt die Full Results API und gibt den Klartext des Ergebnis-Pods."""
    if query in _CACHE:
        return _CACHE[query]
    params = urllib.parse.urlencode({
        "appid": appid, "input": query, "output": "json",
        "format": "plaintext", "podtimeout": "20",
    })
    req = urllib.request.Request(f"{WOLFRAM_URL}?{params}",
                                 headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "Die AppID wurde abgelehnt. Prüfe sie unter „AppID“ — eine "
                "frische AppID braucht manchmal ein paar Minuten.") from exc
        if exc.code == 429:
            raise RuntimeError("Zu viele Anfragen. Das kostenlose Kontingent "
                               "liegt bei 2000 Abfragen im Monat.") from exc
        raise RuntimeError(f"Wolfram antwortete mit HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(explain_network_error(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("Wolfram hat nicht rechtzeitig geantwortet. "
                           "Versuch es gleich noch einmal.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wolfram hat keine lesbare Antwort geschickt.") from exc
    except OSError as exc:                       # Verbindung abgebrochen u. ä.
        raise RuntimeError(explain_network_error(exc)) from exc

    qr = data.get("queryresult", {})
    if qr.get("error"):
        detail = qr.get("error")
        msg = detail.get("msg") if isinstance(detail, dict) else None
        raise RuntimeError(msg or "Wolfram meldet einen Fehler.")
    if not qr.get("success"):
        raise RuntimeError("Wolfram konnte die Anfrage nicht auswerten.")

    pods = qr.get("pods", [])
    preferred = [p for p in pods if p.get("id") in
                 ("Result", "Solution", "DecimalApproximation", "Limit",
                  "Derivative", "TangentLine", "LocalExtrema")]
    for pod in preferred + pods:
        if pod.get("id") in ("Input", "InputInterpretation"):
            continue
        for sub in pod.get("subpods", []):
            text = (sub.get("plaintext") or "").strip()
            if text:
                _CACHE[query] = text
                return text
    raise RuntimeError("Wolfram hat kein Ergebnis zurückgegeben.")


# -- Eingaben für Wolfram und Antworten von Wolfram lesen ------------------

def wolfram_input(expr) -> str:
    """Einen SymPy-Term so schreiben, dass Wolfram|Alpha ihn versteht."""
    if isinstance(expr, str):
        expr = ak_parse(expr)
    if isinstance(expr, sp.Basic) and expr.has(sp.Piecewise):
        return sp.mathematica_code(expr)
    s = sp.sstr(expr).replace("**", "^")
    s = s.replace("Abs(", "abs(").replace("oo", "infinity")
    return s


def wolfram_at(x0) -> str:
    if x0 == sp.oo:
        return "infinity"
    if x0 == -sp.oo:
        return "-infinity"
    return wolfram_input(x0)


_SUPERS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_FUNCS = r"(?:sin|cos|tan|cot|sinh|cosh|tanh|ln|log)"


def _clean_wolfram_line(line: str) -> str:
    s = line.strip().translate(str.maketrans({
        "−": "-", "×": "*", "·": "*", "π": "pi", "∞": "oo", "√": "sqrt",
        "‐": "-", "–": "-"}))
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^" + m.group(0).translate(_SUPERS), s)
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)                        # |x| -> abs(x)
    s = re.sub(rf"({_FUNCS})\^\(\s*-1\s*\)\s*\(", lambda m:
               {"sin": "asin(", "cos": "acos(", "tan": "atan("}.get(m.group(1), m.group(0)), s)
    s = re.sub(rf"({_FUNCS})\^(-?\d+)\(([^()]*)\)", r"(\1(\3))^\2", s)  # sin^2(x)
    s = re.sub(r"\bsec\(([^()]*)\)", r"(1/cos(\1))", s)
    s = re.sub(r"\bcsc\(([^()]*)\)", r"(1/sin(\1))", s)
    s = re.sub(r"\barctan\b", "atan", s)
    return s


def parse_wolfram_expr(text: str) -> sp.Expr:
    """
    Liest einen Term aus Wolframs Klartext: »3 x^2 - 3«, »d/dx(x^3) = 3 x^2«,
    »y = 2 - x«. Wirft ValueError, wenn sich der Text nicht auswerten lässt.
    """
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Wolframs Antwort ist leer.")
    line = lines[0]
    if "≈" in line:
        line = line.split("≈")[0]
    if "=" in line and not re.search(r"[<>!]=", line):
        line = line.rsplit("=", 1)[1]
    line = re.sub(r"\s*\((?:for|when|where|valid).*\)\s*$", "", line, flags=re.I)
    line = re.sub(r"\s+\+\s*constant\s*$", "", line, flags=re.I)
    s = _clean_wolfram_line(line)
    if not s.strip():
        raise ValueError("Wolframs Antwort ist leer.")
    try:
        value = ak_parse(s)
    except Exception as exc:
        raise ValueError(f"nicht auswertbar: {s[:60]}") from exc
    if not isinstance(value, sp.Basic):
        raise ValueError("nicht auswertbar")
    extra = value.free_symbols - {x}
    if extra:
        raise ValueError("nicht auswertbar: unbekannte Symbole "
                         + ", ".join(sorted(map(str, extra))))
    return value


def parse_wolfram_limit(text: str):
    """
    Wolframs Grenzwert: ('value', Zahl | ±oo), ('dne', None) für »existiert
    nicht« oder ('undef', None) für alles, was sich nicht deuten lässt.
    """
    low = (text or "").strip().lower()
    if not low:
        return ("undef", None)
    if any(k in low for k in ("does not exist", "no limit", "doesn't exist",
                              "limit does not")):
        return ("dne", None)
    if "complex infinity" in low or "indeterminate" in low or "undefined" in low:
        return ("undef", None)
    first = low.splitlines()[0].strip().replace("−", "-").replace("∞", "oo")
    if "=" in first:
        first = first.rsplit("=", 1)[1].strip()
    first = re.sub(r"\s+", " ", first)
    if re.fullmatch(r"\+?\s*(oo|infinity)", first):
        return ("value", sp.oo)
    if re.fullmatch(r"-\s*(oo|infinity)", first):
        return ("value", -sp.oo)
    try:
        value = parse_wolfram_expr(first)
    except ValueError:
        return ("undef", None)
    if value.has(sp.zoo, sp.nan):
        return ("undef", None)
    return ("value", value)


def _split_values(text: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b|\n)\s*", text.strip())
    return [p for p in parts if p]


def parse_wolfram_points(text: str) -> list[float]:
    """
    Liest Stellen aus Wolframs Antwort: »x = -1 and x = 1«, »x = ±sqrt(3)«,
    »x = 0«. Leere Antworten (»no critical points«) ergeben []. Periodische
    Lösungen (»n ∈ ℤ«) und alles Unlesbare werfen ValueError.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Wolframs Antwort ist leer.")
    low = t.lower()
    if re.search(r"\bno\b.*\b(point|extrem|solution|root)", low) or low in ("none", "(none)"):
        return []
    if re.search(r"\bn\b\s*(∈|element|in)\s*(ℤ|z\b)|integer", low):
        raise ValueError("nicht auswertbar: periodische Lösung (n ∈ ℤ)")
    values: list[float] = []
    for line in t.splitlines():
        line = line.replace("−", "-").replace("≈", "~")
        if "~" in line:
            line = line.split("~")[0]
        for m in re.finditer(r"\bx\s*=\s*([^,;\n]+?)(?=\s+(?:and|or)\b|,|;|\s+x\s*=|$)", line):
            raw = m.group(1).strip()
            raw = re.split(r"\s+(?:at|for)\b|\s{2,}", raw)[0]
            signs = [1]
            if raw.startswith("±") or raw.startswith("+-"):
                raw = raw[1:].lstrip("-").strip()
                signs = [1, -1]
            try:
                v = float(sp.N(parse_wolfram_expr(raw)))
            except (ValueError, TypeError):
                raise ValueError(f"nicht auswertbar: x = {raw[:40]}") from None
            values.extend(s * v for s in signs)
    if not values:
        raise ValueError("nicht auswertbar: keine Stellen gefunden")
    return sorted(set(round(v, 9) for v in values))


def parse_wolfram_extrema(text: str) -> list[tuple[str, float, float | None]]:
    """
    Liest lokale Extrema: »max{x^3 - 3 x} = 2 at x = -1« oder
    »local minimum | f(1) = -2«. Gibt [('max'|'min', x0, y0)] zurück.
    """
    out: list[tuple[str, float, float | None]] = []
    for raw in (text or "").splitlines():
        line = raw.strip().replace("−", "-")
        if not line:
            continue
        m = re.match(r"(?:local\s+)?(max|min)(?:imum)?\b.*?=\s*(.+?)\s+at\s+(?:x\s*=\s*)?(.+?)\s*$",
                     line, re.I)
        m2 = re.match(r"(?:local\s+)?(max|min)imum\s*\|?\s*f\((.+?)\)\s*=\s*(.+?)\s*$", line, re.I)
        try:
            if m:
                kind, ytxt, xtxt = m.group(1).lower(), m.group(2), m.group(3)
                xs = parse_wolfram_points("x = " + xtxt)
                y = float(sp.N(parse_wolfram_expr(ytxt)))
                out.extend((kind, xv, y) for xv in xs)
            elif m2:
                kind = m2.group(1).lower()
                xv = float(sp.N(parse_wolfram_expr(m2.group(2))))
                y = float(sp.N(parse_wolfram_expr(m2.group(3))))
                out.append((kind, xv, y))
        except (ValueError, TypeError):
            raise ValueError("nicht auswertbar: " + line[:60]) from None
    if not out:
        low = (text or "").lower()
        if re.search(r"\bno\b.*extrem|\bnone\b", low):
            return []
        raise ValueError("nicht auswertbar: keine Extrema gefunden")
    return out


# -- Vergleiche mit SymPy --------------------------------------------------

_SAMPLES = (-3.1, -2.3, -1.7, -0.9, -0.35, 0.27, 0.6, 1.3, 1.9, 2.6, 3.4)


def exprs_equal(a, b):
    """
    True, wenn a und b als Terme in x gleich sind (symbolisch oder, wenn
    SymPy nicht weiterkommt, an mehreren Stellen), False wenn nicht, None
    wenn sich nichts vergleichen lässt.
    """
    a, b = sp.sympify(a), sp.sympify(b)
    try:
        if a == b or sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    close = total = 0
    for t in _SAMPLES:
        try:
            va = complex(sp.N(a.subs(x, t), 20))
            vb = complex(sp.N(b.subs(x, t), 20))
        except Exception:
            continue
        if any(math.isnan(z) or math.isinf(z) for z in (va.real, va.imag, vb.real, vb.imag)):
            continue
        total += 1
        if abs(va - vb) <= 1e-8 * (1 + abs(va) + abs(vb)):
            close += 1
    if total < 3:
        return None
    return close == total


def values_equal(a, b, tol: float = 1e-9) -> bool:
    """Zwei Zahlen gleich? Auch ±∞ und Wurzelausdrücke."""
    a, b = sp.sympify(a), sp.sympify(b)
    try:
        if a == b:
            return True
        if a in (sp.oo, -sp.oo) or b in (sp.oo, -sp.oo):
            return False
        if sp.simplify(a - b) == 0:
            return True
        va, vb = complex(sp.N(a, 30)), complex(sp.N(b, 30))
        return abs(va - vb) <= tol * (1 + abs(va))
    except Exception:
        return False


def point_lists_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(u - v) <= tol * (1 + abs(u)) for u, v in zip(sorted(a), sorted(b)))


@dataclass
class Check:
    """Ein einzelner Prüfpunkt im Bericht."""
    name: str
    status: str          # 'ok' | 'bad' | 'info' | 'error'
    ours: str = ""
    theirs: str = ""
    note: str = ""


class WolframReport:
    def __init__(self):
        self.checks: list[Check] = []
        self.texts: list[tuple[str, str]] = []     # (Frage, Wolframs Antwort)
        self.url: str = ""

    @property
    def verdict(self) -> str:
        if any(c.status == "bad" for c in self.checks):
            return "bad"
        if any(c.status == "error" for c in self.checks):
            return "warn"
        if any(c.status == "ok" for c in self.checks):
            return "ok"
        return "warn"

    @property
    def headline(self) -> str:
        return {
            "ok": "Wolfram|Alpha bestätigt das Ergebnis.",
            "bad": "Wolfram|Alpha kommt auf etwas anderes.",
            "warn": "Wolfram konnte nicht alles prüfen.",
        }[self.verdict]


def _ask(appid: str, queries: dict[str, str], ask=None) -> dict[str, object]:
    """Stellt die Anfragen nebeneinander und sammelt Antworten wie Fehler."""
    ask = ask or wolfram_text
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(ask, appid, q) for key, q in queries.items()}
        answers: dict[str, object] = {}
        for key, fut in futures.items():
            try:
                answers[key] = fut.result()
            except (RuntimeError, ValueError) as exc:
                answers[key] = exc
    # Scheitert alles, ist meist AppID oder Netz schuld — ein Bericht mit
    # lauter gleichen Fehlern hilft nicht.
    if answers and all(isinstance(v, Exception) for v in answers.values()):
        raise RuntimeError(str(next(iter(answers.values()))))
    return answers


def _show(v) -> str:
    return fmt(v) if not isinstance(v, str) else v


def _note_text(report: WolframReport, label: str, query: str, answer):
    if isinstance(answer, str):
        report.texts.append((label, answer.splitlines()[0] if answer.count("\n") == 0
                             else answer.replace("\n", "  ·  ")))


def check_expr(report: WolframReport, name: str, ours, answer, label: str = "",
               query: str = "", at=None, ours_text=None) -> None:
    """
    Vergleicht einen Term mit Wolframs Antwort: Differenz mit SymPy
    vereinfachen. Lässt sich Wolframs Text nicht lesen, steht er im Bericht
    und der Punkt heisst »nicht auswertbar«.

    at: wenn gegeben, wird an dieser Stelle verglichen (Wert statt Term).
    """
    shown = ours_text if ours_text is not None else _show(ours)
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=shown, note=str(answer)))
        return
    _note_text(report, label or name, query, answer)
    try:
        theirs = parse_wolfram_expr(answer)
    except ValueError as exc:
        report.checks.append(Check(name, "info", ours=shown,
                                   note=f"Wolframs Antwort „{answer.splitlines()[0][:50]}“ "
                                        f"ist {str(exc) if 'nicht auswertbar' in str(exc) else 'nicht auswertbar'}."))
        return
    if at is not None:
        try:
            theirs_v = theirs.subs(x, at)
        except Exception:
            theirs_v = None
        if ours is None:
            bad = theirs_v is not None and not theirs_v.has(sp.zoo, sp.nan) and theirs_v.is_finite
            report.checks.append(Check(name, "bad" if bad else "ok", ours=shown,
                                       theirs=_show(theirs_v) if theirs_v is not None else ""))
            return
        same = theirs_v is not None and values_equal(ours, theirs_v)
        report.checks.append(Check(name, "ok" if same else "bad", ours=shown,
                                   theirs="" if same else _show(theirs_v)))
        return
    verdict = exprs_equal(ours, theirs)
    if verdict is None:
        report.checks.append(Check(name, "info", ours=shown, theirs=_show(theirs),
                                   note="Nicht auswertbar: keine Stelle gefunden, an der beide definiert sind."))
    else:
        report.checks.append(Check(name, "ok" if verdict else "bad", ours=shown,
                                   theirs="" if verdict else _show(theirs),
                                   note="Differenz mit SymPy vereinfacht: 0." if verdict else ""))


def limit_of(res, which: str):
    """Unser Grenzwert als ('value', v) oder ('dne', None); which: both | left | right."""
    if which == "left":
        v, kind = res.left, res.left_kind
    elif which == "right":
        v, kind = res.right, res.right_kind
    else:
        v, kind = (res.value, res.kind) if res.exists else (None, "existiert nicht")
    if v is None:
        return ("dne", None)
    return ("value", v)


def limit_str(lim) -> str:
    kind, v = lim
    return "existiert nicht" if kind == "dne" else fmt(v)


def check_limit(report: WolframReport, name: str, ours, answer, label: str = "",
                query: str = "") -> None:
    """Vergleicht einen Grenzwert (value | dne) mit Wolframs Antwort."""
    shown = limit_str(ours)
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=shown, note=str(answer)))
        return
    _note_text(report, label or name, query, answer)
    kind, v = parse_wolfram_limit(answer)
    if kind == "undef":
        report.checks.append(Check(name, "info", ours=shown,
                                   note=f"Wolframs Antwort „{answer.splitlines()[0][:50]}“ "
                                        "ist nicht auswertbar."))
        return
    theirs_s = "existiert nicht" if kind == "dne" else fmt(v)
    if ours[0] == "dne":
        same = kind == "dne"
    else:
        same = kind == "value" and values_equal(ours[1], v)
    report.checks.append(Check(name, "ok" if same else "bad", ours=shown,
                               theirs="" if same else theirs_s))


def check_points(report: WolframReport, name: str, ours: Sequence, answer, label: str = "",
                 query: str = "", domain=None) -> None:
    """Vergleicht Stellen (kritische Punkte, Wendestellen) als Menge."""
    shown = ", ".join(fmt(v) for v in ours) or "keine"
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=shown, note=str(answer)))
        return
    _note_text(report, label or name, query, answer)
    try:
        theirs = parse_wolfram_points(answer)
    except ValueError as exc:
        report.checks.append(Check(name, "info", ours=shown, note=_nicht_auswertbar(answer, exc)))
        return
    theirs = _in_domain(theirs, domain)
    mine = sorted(float(sp.N(v)) for v in ours)
    same = point_lists_equal(mine, theirs)
    report.checks.append(Check(
        name, "ok" if same else "bad", ours=shown,
        theirs="" if same else (", ".join(_num(v) for v in theirs) or "keine"),
        note="Die gleichen Stellen." if same else ""))


def _num(v: float) -> str:
    return fmt(sp.nsimplify(round(v, 9), rational=False) if abs(v - round(v)) < 1e-9 else sp.Float(v, 5))


def _nicht_auswertbar(answer, exc) -> str:
    first = answer.splitlines()[0][:50] if answer else ""
    return f"Wolframs Antwort „{first}“ ist nicht auswertbar ({str(exc).replace('nicht auswertbar: ', '')})."


def _in_domain(values: Sequence[float], domain) -> list[float]:
    if not domain:
        return list(values)
    lo = -math.inf if domain[0] is None else float(domain[0])
    hi = math.inf if domain[1] is None else float(domain[1])
    return [v for v in values if lo < v < hi]


def _ordinal(n: int) -> str:
    return {1: "", 2: "second ", 3: "third "}.get(n, f"{n}th ")


def wolfram_checks_steigung(appid, p, res, ask=None) -> WolframReport:
    report = WolframReport()
    method = p.get("method")
    if method == "line":
        P1 = (sp.nsimplify(res.points[0][0]), sp.nsimplify(res.points[0][1]))
        P2 = (sp.nsimplify(res.points[1][0]), sp.nsimplify(res.points[1][1]))
        m = getattr(res, "m", None)
        q = f"line through ({wolfram_input(P1[0])}, {wolfram_input(P1[1])}) and " \
            f"({wolfram_input(P2[0])}, {wolfram_input(P2[1])})"
        report.url = browser_url(q)
        ans = _ask(appid, {"line": q}, ask)["line"]
        if res.vertical:
            report.checks.append(Check("Gerade", "info", ours="senkrecht",
                                       note="Für senkrechte Geraden gibt es keine Form y = m·x + t."))
            _note_text(report, "Gerade", q, ans)
        else:
            check_expr(report, "Geradengleichung", res.line, ans, query=q)
        return report
    f = res.f
    wf = wolfram_input(f)
    if method in ("sek12", "sekh"):
        a, b = res.x1, res.x2
        fa, fb = wolfram_input(f.subs(x, a)), wolfram_input(f.subs(x, b))
        q = f"simplify (({fb}) - ({fa}))/(({wolfram_input(b)}) - ({wolfram_input(a)}))"
        report.url = browser_url(q)
        ans = _ask(appid, {"m": q}, ask)["m"]
        check_expr(report, "Sekantensteigung m", res.m, ans, label="Differenzenquotient", query=q)
        return report
    if method == "tan":
        q1, q2 = f"derivative of {wf}", f"tangent line to {wf} at x = {wolfram_at(res.x0)}"
        report.url = browser_url(q1)
        answers = _ask(appid, {"d": q1, "t": q2}, ask)
        check_expr(report, "Steigung f′(x₀)", res.value, answers["d"], label="Ableitung f′(x)",
                   query=q1, at=res.x0, ours_text=fmt(res.value) if res.value is not None
                   else "nicht differenzierbar")
        if res.line is not None:
            check_expr(report, "Tangentengleichung", res.line, answers["t"], label="Tangente", query=q2)
        return report
    q = f"derivative of {wf}"
    report.url = browser_url(q)
    answer = _ask(appid, {"d": q}, ask)["d"]
    check_expr(report, "Ableitungsfunktion f′(x)", res.expr, answer, label="Ableitung f′(x)", query=q)
    return report


def wolfram_checks_limits(appid, p, res, ask=None) -> WolframReport:
    report = WolframReport()
    action = p.get("action", "limit")
    if action == "contall":
        pts = [pt for pt in res.points if not pt.continuous][:3] or res.points[:3]
        queries: dict[str, str] = {}
        meta = {}
        wf = wolfram_input(res.expr if not res.expr.has(sp.Piecewise) else res.expr)
        for k, pt in enumerate(pts):
            for side, arrow in (("left", "^-"), ("right", "^+")):
                key = f"{side}{k}"
                queries[key] = f"limit of {wf} as x->{wolfram_at(pt.x0)}{arrow}"
                meta[key] = (pt, side)
        if not queries:
            report.checks.append(Check("Verdachtsstellen", "info",
                                       note="Es gibt keine Verdachtsstellen zu prüfen."))
            return report
        report.url = browser_url(next(iter(queries.values())))
        answers = _ask(appid, queries, ask)
        for key, (pt, side) in meta.items():
            name = f"x₀ = {fmt(pt.x0)}: {'linksseitig' if side == 'left' else 'rechtsseitig'}"
            mine = ("value", pt.left if side == "left" else pt.right) \
                if (pt.left if side == "left" else pt.right) is not None else ("dne", None)
            check_limit(report, name, mine, answers[key], label=name, query=queries[key])
        return report

    x0 = res.x0
    wf = wolfram_input(res.expr if hasattr(res, "expr") else p.get("f", ""))
    at = wolfram_at(x0)
    infinite = x0 in (sp.oo, -sp.oo)
    queries = {}
    if action == "cont":
        queries = {"left": f"limit of {wf} as x->{at}^-", "right": f"limit of {wf} as x->{at}^+"}
    elif infinite:
        queries = {"both": f"limit of {wf} as x->{at}"}
    else:
        side = getattr(res, "side", "both")
        if side == "both":
            queries = {"both": f"limit of {wf} as x->{at}",
                       "left": f"limit of {wf} as x->{at}^-",
                       "right": f"limit of {wf} as x->{at}^+"}
        elif side == "-":
            queries = {"left": f"limit of {wf} as x->{at}^-"}
        else:
            queries = {"right": f"limit of {wf} as x->{at}^+"}
    report.url = browser_url(next(iter(queries.values())))
    answers = _ask(appid, queries, ask)
    names = {"both": "Grenzwert", "left": "linksseitiger Grenzwert", "right": "rechtsseitiger Grenzwert"}
    for key, q in queries.items():
        if action == "cont":
            mine = ("value", res.left) if key == "left" and res.left is not None else \
                   ("value", res.right) if key == "right" and res.right is not None else ("dne", None)
        elif hasattr(res, "side") and res.side in ("+", "-") and key != "both":
            mine = ("value", res.value) if res.value is not None else ("dne", None)
        else:
            mine = limit_of(res, key)
        check_limit(report, names[key], mine, answers[key], label=names[key], query=q)
    return report


def wolfram_checks_derivative(appid, p, res, ask=None) -> WolframReport:
    report = WolframReport()
    action = p.get("action", "diff")
    if action == "table":
        report.checks.append(Check("Regeltabelle", "info",
                                   note="Die Regeltabelle ist reine Theorie — dazu gibt es nichts zu prüfen."))
        return report
    if action == "rule":
        f = res.whole
        q = f"derivative of {wolfram_input(f)}"
        report.url = browser_url(q)
        ans = _ask(appid, {"d": q}, ask)["d"]
        check_expr(report, f"Ableitung ({res.title})", res.expr, ans, label="Ableitung", query=q)
        return report
    f = res.f
    wf = wolfram_input(f)
    if action == "diff":
        n = res.order
        q = f"{_ordinal(n)}derivative of {wf}"
        report.url = browser_url(q)
        ans = _ask(appid, {"d": q}, ask)["d"]
        check_expr(report, "Ableitung" if n == 1 else f"{n}. Ableitung", res.expr, ans,
                   label="Ableitung", query=q)
    elif action == "at":
        q = f"derivative of {wf}"
        report.url = browser_url(q)
        ans = _ask(appid, {"d": q}, ask)["d"]
        check_expr(report, "Ableitungsfunktion f′(x)", ak.diff(f).expr, ans, label="Ableitung", query=q)
        check_expr(report, f"f′({fmt(res.x0)})", res.value, ans, label="Ableitung", query=q, at=res.x0)
    elif action == "check":
        q = f"derivative of {wf}"
        report.url = browser_url(q)
        ans = _ask(appid, {"d": q}, ask)["d"]
        if isinstance(ans, Exception):
            report.checks.append(Check("Deine Ableitung", "error", note=str(ans)))
        else:
            _note_text(report, "Ableitung", q, ans)
            try:
                theirs = parse_wolfram_expr(ans)
                verdict = exprs_equal(res.claimed, theirs)
            except ValueError as exc:
                report.checks.append(Check("Deine Ableitung", "info", ours=fmt(res.claimed),
                                           note=_nicht_auswertbar(ans, exc)))
                return report
            if verdict is None:
                report.checks.append(Check("Deine Ableitung", "info", ours=fmt(res.claimed),
                                           note="Nicht auswertbar: keine gemeinsame Stelle."))
            elif verdict == bool(res.correct):
                report.checks.append(Check(
                    "Urteil der App", "ok", ours="richtig" if res.correct else "falsch",
                    note="Wolfram kommt zum gleichen Urteil über deine Ableitung."))
            else:
                report.checks.append(Check(
                    "Urteil der App", "bad", ours="richtig" if res.correct else "falsch",
                    theirs="richtig" if verdict else "falsch"))
    return report


def wolfram_checks_curve(appid, p, res, ask=None) -> WolframReport:
    report = WolframReport()
    action = p.get("action", "all")
    f = res.f
    wf = wolfram_input(f)
    dom = _domain(p)
    queries: dict[str, str] = {}
    if action in ("all", "mono", "crit"):
        queries["crit"] = f"critical points of {wf}"
    if action in ("all", "ext"):
        queries["ext"] = f"local extrema of {wf}"
    if action in ("all", "curv", "infl"):
        queries["infl"] = f"inflection points of {wf}"
    report.url = browser_url(next(iter(queries.values())))
    answers = _ask(appid, queries, ask)
    if "crit" in answers:
        check_points(report, "Kritische Stellen f′(x₀) = 0", list(res.critical_x), answers["crit"],
                     label="Kritische Punkte", query=queries["crit"], domain=dom)
    if "ext" in answers:
        ans = answers["ext"]
        mins = [pt[0] for pt in getattr(res, "minima", [])]
        maxs = [pt[0] for pt in getattr(res, "maxima", [])]
        shown = "Max: " + (", ".join(fmt(v) for v in maxs) or "—") + " · Min: " + (", ".join(fmt(v) for v in mins) or "—")
        if isinstance(ans, Exception):
            report.checks.append(Check("Lokale Extrema", "error", ours=shown, note=str(ans)))
        else:
            _note_text(report, "Lokale Extrema", queries["ext"], ans)
            try:
                found = parse_wolfram_extrema(ans)
            except ValueError as exc:
                report.checks.append(Check("Lokale Extrema", "info", ours=shown,
                                           note=_nicht_auswertbar(ans, exc)))
            else:
                tmax = _in_domain([xv for k, xv, _ in found if k == "max"], dom)
                tmin = _in_domain([xv for k, xv, _ in found if k == "min"], dom)
                same = point_lists_equal([float(sp.N(v)) for v in maxs], tmax) and \
                    point_lists_equal([float(sp.N(v)) for v in mins], tmin)
                report.checks.append(Check(
                    "Lokale Extrema", "ok" if same else "bad", ours=shown,
                    theirs="" if same else "Max: " + (", ".join(_num(v) for v in tmax) or "—")
                    + " · Min: " + (", ".join(_num(v) for v in tmin) or "—"),
                    note="Maxima und Minima stimmen überein." if same else ""))
    if "infl" in answers:
        ours = list(getattr(res, "inflection_x", None) or
                    [pt[0] for pt in getattr(res, "inflection_points", [])])
        check_points(report, "Wendestellen f″(x₀) = 0", ours, answers["infl"],
                     label="Wendepunkte", query=queries["infl"], domain=dom)
    return report


def _wolfram_series(answer: str) -> sp.Expr:
    """Wolframs Reihe »1 + x + x^2/2 + O(x^3)« ohne das Restglied lesen."""
    first = next((ln for ln in (answer or "").splitlines() if ln.strip()), "")
    first = re.sub(r"\s*[+-]\s*O\(.*$", "", first.split("=")[-1] if "=" in first else first)
    return parse_wolfram_expr(first)


def wolfram_checks_numerik(appid, p, res, ask=None) -> WolframReport:
    report = WolframReport()
    action = p.get("action", "newton")
    wf = wolfram_input(res.f)
    if action in ("bisect", "newton"):
        q = f"solve {wf} = 0 for x"
        report.url = browser_url(q)
        ans = _ask(appid, {"r": q}, ask)["r"]
        name = "Nullstelle (Näherung)"
        if isinstance(ans, Exception):
            report.checks.append(Check(name, "error", ours=_num(res.root), note=str(ans)))
            return report
        _note_text(report, "Nullstellen", q, ans)
        try:
            theirs = parse_wolfram_points(ans)
        except ValueError as exc:
            report.checks.append(Check(name, "info", ours=_num(res.root), note=_nicht_auswertbar(ans, exc)))
            return report
        near = [t for t in theirs if abs(t - res.root) <= 1e-4 * (1 + abs(t))]
        report.checks.append(Check(
            name, "ok" if near else "bad", ours=_num(res.root),
            theirs="" if near else (", ".join(_num(t) for t in theirs) or "keine"),
            note="Liegt bei einer Nullstelle von Wolfram|Alpha." if near else ""))
        return report
    q = f"derivative of {wf}"
    report.url = browser_url(q)
    ans = _ask(appid, {"d": q}, ask)["d"]
    exact = getattr(res, "exact", None)
    if exact is None:
        exact = sp.diff(res.f, x).subs(x, sp.nsimplify(res.x0))
    check_expr(report, "Exakte Ableitung f′(x₀) zum Vergleich", exact, ans, label="Ableitung",
               query=q, at=sp.nsimplify(res.x0))
    return report


def wolfram_checks_hoeher(appid, p, res, ask=None) -> WolframReport:
    report = WolframReport()
    action = p.get("action", "mac")
    f = res.f
    wf = wolfram_input(f)
    if action in ("taylor", "mac"):
        at = wolfram_at(res.x0)
        q = f"Taylor series of {wf} at x = {at} to order {res.n}"
        report.url = browser_url(q)
        ans = _ask(appid, {"t": q}, ask)["t"]
        name = f"Taylor-Polynom T{res.n}"
        if isinstance(ans, Exception):
            report.checks.append(Check(name, "error", ours=fmt(res.polynomial), note=str(ans)))
            return report
        _note_text(report, "Reihe", q, ans)
        try:
            theirs = _wolfram_series(ans)
        except ValueError as exc:
            report.checks.append(Check(name, "info", ours=fmt(res.polynomial), note=_nicht_auswertbar(ans, exc)))
            return report
        verdict = exprs_equal(res.polynomial, theirs)
        report.checks.append(Check(name, "ok" if verdict else "bad", ours=fmt(res.polynomial),
                                   theirs="" if verdict else fmt(theirs),
                                   note="Differenz mit SymPy vereinfacht: 0." if verdict else ""))
        return report
    if action == "higher":
        q = f"critical points of {wf}"
        report.url = browser_url(q)
        ans = _ask(appid, {"c": q}, ask)["c"]
        name = f"x₀ = {fmt(res.x0)} ist kritisch" if res.stationary else f"x₀ = {fmt(res.x0)}"
        if not res.stationary:
            report.checks.append(Check(name, "info", ours=res.kind,
                                       note="Keine waagrechte Tangente — nichts zu vergleichen."))
            return report
        if isinstance(ans, Exception):
            report.checks.append(Check(name, "error", note=str(ans)))
            return report
        _note_text(report, "Kritische Punkte", q, ans)
        try:
            theirs = parse_wolfram_points(ans)
        except ValueError as exc:
            report.checks.append(Check(name, "info", note=_nicht_auswertbar(ans, exc)))
            return report
        found = any(abs(t - float(res.x0)) < 1e-6 for t in theirs)
        report.checks.append(Check(name, "ok" if found else "bad", ours=res.kind,
                                   theirs="" if found else "nicht unter den kritischen Punkten"))
        return report
    q = f"roots of {wf}"
    report.url = browser_url(q)
    ans = _ask(appid, {"z": q}, ask)["z"]
    check_points(report, "Nullstellen", [z for z in res.zeros if z.is_number], ans,
                 label="Nullstellen", query=q)
    return report


WOLFRAM_CHECKS = {"steigung": wolfram_checks_steigung, "limits": wolfram_checks_limits,
                  "derivative": wolfram_checks_derivative, "curve": wolfram_checks_curve,
                  "numerik": wolfram_checks_numerik, "hoeher": wolfram_checks_hoeher}


def run_wolfram_checks(appid: str, page: str, p: dict, result, ask=None) -> WolframReport:
    """
    Prüft eine Rechnung mit Wolfram|Alpha. `ask(appid, frage) -> Text` lässt
    sich im Test durch eine Attrappe ersetzen. Wirft RuntimeError, wenn
    Netz oder AppID versagen.
    """
    return WOLFRAM_CHECKS[page](appid, p, result, ask)


def wolfram_subtitle(page: str) -> str:
    return {
        "steigung": "Vergleicht Ableitung, Tangentengleichung beziehungsweise den "
                    "Differenzenquotienten mit Wolfram|Alpha.",
        "limits": "Vergleicht den Grenzwert — beidseitig, von links und von rechts — "
                  "mit Wolfram|Alpha.",
        "derivative": "Vergleicht die Ableitung mit »derivative of f« bei Wolfram|Alpha.",
        "curve": "Vergleicht kritische Punkte, lokale Extrema und Wendepunkte mit "
                 "Wolfram|Alpha.",
        "numerik": "Vergleicht die Nullstelle mit »solve f(x) = 0« beziehungsweise die exakte "
                   "Ableitung mit Wolfram|Alpha.",
        "hoeher": "Vergleicht das Taylor-Polynom (»Taylor series …«), den kritischen Punkt "
                  "beziehungsweise die Nullstellen mit Wolfram|Alpha.",
    }[page]


# ==========================================================================
# Farben und Schriften
# ==========================================================================

# Swiss Style (claude-design-styles): Schwarz/Weiss, ein Rot, keine Rundungen.
LIGHT = dict(
    bg="#ffffff", nav="#ffffff", card="#ffffff", text="#000000", sec="#333333",
    ter="#666666", line="#000000", hair="#cccccc", fill="#f4f4f4", fill2="#f4f4f4",
    seg="#ffffff", blue="#000000", blue_hi="#e60000", blue_soft="#f4f4f4",
    green="#18794e", green_soft="#e6f4ec", red="#e60000", red_soft="#fdecec",
    amber="#ab5200", amber_soft="#fff4e0", flash="#fff3c4",
    entry="#ffffff", shadow=0, dark=False)

DARK = dict(
    bg="#000000", nav="#000000", card="#000000", text="#ffffff", sec="#cccccc",
    ter="#888888", line="#ffffff", hair="#333333", fill="#1a1a1a", fill2="#111111",
    seg="#333333", blue="#e60000", blue_hi="#ff3333", blue_soft="#2a0000",
    green="#46a758", green_soft="#102a16", red="#ff3333", red_soft="#2a0000",
    amber="#ffb224", amber_soft="#3a2a0a", flash="#3d3210",
    entry="#000000", shadow=0, dark=True)

T: dict = dict(LIGHT)


def pick_font(candidates: Sequence[str], fallback: str) -> str:
    families = {f.lower(): f for f in QFontDatabase.families()}
    for name in candidates:
        if name.lower() in families:
            return families[name.lower()]
    return fallback


class Fonts:
    def __init__(self):
        ui = pick_font(["Geist", "Inter", "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue",
                        "Segoe UI Variable Text", "Segoe UI", "Inter",
                        "DejaVu Sans"], "Sans Serif")
        display = pick_font(["Geist", "Inter", "SF Pro Display", "SF Pro Text", "Helvetica Neue",
                             "Segoe UI Variable Display", "Segoe UI", "Inter",
                             "DejaVu Sans"], ui)
        mono = pick_font(["SF Mono", "Menlo", "Cascadia Mono", "Consolas",
                          "DejaVu Sans Mono"], "Monospace")
        self.family, self.display, self.mono_family = ui, display, mono

    def f(self, size: int, weight: int = QFont.Normal,
          display: bool = False, mono: bool = False) -> QFont:
        fam = self.mono_family if mono else (self.display if display else self.family)
        font = QFont(fam, size)
        font.setWeight(weight)
        if display:
            font.setLetterSpacing(QFont.PercentageSpacing, 97)
        return font


FONTS: Fonts | None = None


def qcolor(key: str) -> QColor:
    return QColor(T[key])


# ==========================================================================
# Gesetzte Formeln: LaTeX -> Matplotlib-Mathtext -> Bild
# ==========================================================================
#
# Die Rechenwege von anakit tragen ihre Terme als LaTeX. Matplotlib kann einen
# grossen Teil davon selbst setzen (Brüche, Wurzeln, Potenzen, Grenzwerte),
# ohne dass ein LaTeX auf dem Rechner stehen muss. Aus dem Ergebnis wird eine
# Alpha-Maske; die App färbt sie je nach Thema ein. Was Mathtext nicht kann
# (Fallunterscheidungen setzt die App selbst zeilenweise), fällt auf den
# Unicode-Text von render.tex_to_text zurück.

import numpy as np                                            # noqa: E402

MATH_SCALE = 3                  # Bildpunkte je logischem Punkt: scharf auch auf Retina
_MATH_LOCK = threading.Lock()
_PARSER = None


@dataclass
class Mask:
    """Eine gesetzte Formel als Alpha-Maske; `depth` = Tiefe unter der Grundlinie."""
    alpha: "np.ndarray"
    depth: float

    @property
    def size(self) -> tuple[float, float]:
        """Logische Breite und Höhe."""
        h, w = self.alpha.shape
        return w / MATH_SCALE, h / MATH_SCALE


_MASKS: "OrderedDict[tuple[str, int], Mask | None]" = OrderedDict()
_MASK_LIMIT = 6000


def _escape_text(inner: str) -> str:
    inner = (inner.replace("\\", "").replace("_", r"\_").replace("%", r"\%")
             .replace("&", r"\&").replace("#", r"\#").replace("$", "")
             .replace("{", "(").replace("}", ")").replace("^", r"\^{}"))
    return inner


_SUB = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
        "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
        "\u208a": "+", "\u208b": "-", "\u2093": "x", "\u2095": "h", "\u2099": "n"}
_SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
        "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
        "\u207a": "+", "\u207b": "-", "\u207f": "n"}
_SUBRUN = re.compile("[" + "".join(_SUB) + "]+")
_SUPRUN = re.compile("[" + "".join(_SUP) + "]+")


def _unicode_math(s: str) -> str:
    s = _SUBRUN.sub(lambda m: "_{" + "".join(_SUB[c] for c in m.group()) + "}", s)
    s = _SUPRUN.sub(lambda m: "^{" + "".join(_SUP[c] for c in m.group()) + "}", s)
    return (s.replace("\u2212", "-").replace("\u221e", r"\infty").replace("\u03c0", r"\pi")
             .replace("\u00b1", r"\pm").replace("\u2264", r"\leq").replace("\u2265", r"\geq")
             .replace("\u2192", r"\to").replace("\u22c5", r"\cdot").replace("\u00b7", r"\cdot"))


def mathtext_ready(tex: str) -> str:
    """Passt LaTeX an das an, was Matplotlibs Mathtext versteht."""
    s = _unicode_math(tex.strip())
    s = s.replace(r"\operatorname", r"\mathrm").replace(r"\textbf", r"\mathbf")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left.", "").replace(r"\right.", "")
    if not re.search(r"\\(?:d?frac|sqrt|lim|sum|int|binom|begin)", s):
        # Ohne hohe Inhalte wirken die mitwachsenden Klammern nur winzig.
        s = s.replace(r"\left(", "(").replace(r"\right)", ")")
        s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    s = s.replace(r"\lvert", "|").replace(r"\rvert", "|")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = re.sub(r"\\text\{((?:[^{}]|\{[^{}]*\})*)\}",
               lambda m: r"\text{" + _escape_text(m.group(1)) + "}", s)
    return s


def render_mask(tex: str, px: int) -> "Mask | None":
    """
    Setzt `tex` in der Schriftgrösse px (logische Bildpunkte). Gibt None, wenn
    Mathtext es nicht kann — der Aufrufer nimmt dann Text. Ergebnisse werden
    gemerkt; der Aufruf ist von mehreren Fäden aus sicher.
    """
    global _PARSER
    key = (tex, px)
    if key in _MASKS:
        return _MASKS[key]
    mask = None
    with _MATH_LOCK:
        if key in _MASKS:
            return _MASKS[key]
        try:
            from matplotlib.font_manager import FontProperties
            from matplotlib.mathtext import MathTextParser
            if _PARSER is None:
                _PARSER = MathTextParser("agg")
            prop = FontProperties(size=px, math_fontfamily="cm")
            raster = _PARSER.parse("$" + mathtext_ready(tex) + "$",
                                   dpi=72 * MATH_SCALE, prop=prop)
            alpha = np.array(raster.image, dtype=np.uint8)
            if alpha.ndim == 2 and alpha.size:
                mask = Mask(alpha, float(raster.depth))
        except Exception:
            mask = None
        _MASKS[key] = mask
        if len(_MASKS) > _MASK_LIMIT:
            _MASKS.popitem(last=False)
    return mask


def mask_image(mask: Mask, color: str, inline_px: int | None = None) -> QImage:
    """
    Färbt eine Maske ein. Mit inline_px wird sie oben oder unten aufgefüllt, bis
    die Mathe-Achse (etwa ¼ Schrifthöhe über der Grundlinie) in der Bildmitte
    liegt — so sitzt sie in einem Satz mit `vertical-align: middle` richtig.
    """
    alpha = mask.alpha
    if inline_px is not None:
        h = alpha.shape[0]
        axis = 0.27 * inline_px * MATH_SCALE
        ascent = h - mask.depth
        above, below = ascent - axis, mask.depth + axis
        half = max(above, below)
        top = max(0, int(round(half - above)))
        bottom = max(0, int(round(half - below)))
        alpha = np.pad(alpha, ((top, bottom), (0, 0)))
    h, w = alpha.shape
    col = QColor(color)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = col.red(), col.green(), col.blue()
    rgba[..., 3] = alpha
    image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    image.setDevicePixelRatio(MATH_SCALE)
    return image


def screen_dpr() -> float:
    screen = QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen is not None else 1.0


def scaled_image(mask: Mask, color: str, scale: float = 1.0, dpr: float | None = None,
                 inline_px: int | None = None) -> "tuple[QImage, float, float]":
    """
    Die Maske in Bildschirmauflösung: aus dem dreifach gesetzten Bild glatt
    heruntergerechnet, damit dünne Striche (Klammern, Bruchstriche) nicht
    wegbrechen. Gibt (Bild, logische Breite, logische Höhe) zurück.
    """
    dpr = dpr or screen_dpr()
    image = mask_image(mask, color, inline_px)
    lw = image.width() / MATH_SCALE * scale
    lh = image.height() / MATH_SCALE * scale
    tw, th = max(1, round(lw * dpr)), max(1, round(lh * dpr))
    out = image.scaled(tw, th, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    out.setDevicePixelRatio(dpr)
    return out, lw, lh


_B64: "OrderedDict[tuple, str]" = OrderedDict()


def mask_data_uri(tex: str, px: int, color: str) -> "tuple[str, float, float] | None":
    """(data-URI, Breite, Höhe) für ein Bild im Fliesstext, oder None."""
    dpr = screen_dpr()
    key = (tex, px, color, dpr)
    mask = render_mask(tex, px)
    if mask is None:
        return None
    if key not in _B64:
        image, lw, lh = scaled_image(mask, color, 1.0, dpr, inline_px=px)
        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QIODevice.WriteOnly)
        image.save(buf, "PNG")
        buf.close()
        _B64[key] = ("data:image/png;base64," + base64.b64encode(bytes(data)).decode(), lw, lh)
        if len(_B64) > 4000:
            _B64.popitem(last=False)
    return _B64[key]


# -- Das Ansichtsmodell: was die Karten zeigen, ganz ohne Fenster ------------

_WORD = re.compile(r"[A-Za-zÄÖÜäöüéèàç]{3,}")


def label_tex(label: str) -> str:
    """Eine Beschriftung wie »f'(x_0)« oder »Newton« als Mathtext-Stück."""
    if not label:
        return ""
    if "\\" in label:
        return label.replace(r"\!", "").strip()
    out = []
    for token in label.split(" "):
        if _WORD.fullmatch(token) and not re.search(r"[_^]", token):
            out.append(r"\text{" + token + "}")
        elif token:
            out.append(token)
    return r"\ ".join(out)


def _tex_value(obj, decimal: bool, digits: int) -> str:
    try:
        return tex_any(obj, decimal, digits)
    except Exception:
        return r"\text{" + str(obj) + "}"


_CASES = re.compile(r"\\begin\{cases\}(.*)\\end\{cases\}", re.S)


def split_cases(value_tex: str) -> "list[str] | None":
    """Zerlegt \\begin{cases} … \\end{cases} in einzelne Zeilen »Term für Bedingung«."""
    m = _CASES.search(value_tex)
    if not m:
        return None
    rows = []
    for row in re.split(r"\\\\", m.group(1)):
        if not row.strip():
            continue
        left, _, cond = row.partition("&")
        cond = cond.strip()
        if cond.startswith(r"\text{otherwise}"):
            cond = r"\text{sonst}"
        elif cond.startswith(r"\text{for}"):
            cond = cond[len(r"\text{for}"):].strip()
        if cond.startswith(r"\text{sonst}") or cond == "sonst":
            tail = r"\text{sonst}"
        elif cond:
            tail = r"\text{falls}\ " + cond
        else:
            tail = ""
        rows.append(r"\quad " + left.strip() + (r"\qquad " + tail if tail else ""))
    return rows


@dataclass
class ItemVM:
    """Eine benannte Grösse: »Beschriftung = Wert«, gesetzt oder als Text."""
    tex: "str | None"
    text: str
    lines: "list[str] | None" = None      # Fallunterscheidung: Kopf und Zeilen


def item_vm(name: str, obj, decimal: bool, digits: int, big: bool = False) -> ItemVM:
    value = _tex_value(obj, decimal, digits)
    head = label_tex(name)
    fallback = (tex_to_text(name) + " = " if name else "") + text_any(obj, decimal, digits)
    cases = split_cases(value)
    if cases:
        return ItemVM(None, fallback, [(head + " =") if head else r"\text{abschnittsweise}"] + cases)
    value = re.sub(r"(?<!d)\\frac", r"\\dfrac", value)       # Brüche in voller Grösse
    body = (head + " = " if head else "") + value
    return ItemVM(body, fallback)


def split_math(text: str) -> "list[tuple[str, str]]":
    """Teilt einen Satz mit $…$ in [('t', Text), ('m', LaTeX), …]."""
    parts: list[tuple[str, str]] = []
    pieces = text.split("$")
    if len(pieces) % 2 == 0:               # unpaariges $ — alles als Text
        return [("t", tex_to_text(text) if "\\" in text else text)]
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        if i % 2:
            parts.append(("m", piece))
        else:
            parts.append(("t", tex_to_text(piece) if "\\" in piece else piece))
    return parts


@dataclass
class StepVM:
    index: int
    title: str
    rule: str
    detail: "list[tuple[str, str]]"
    items: "list[ItemVM]"
    note: str


@dataclass
class ResultVM:
    title: str
    summary: "list[tuple[str, str]]"
    warnings: "list[list[tuple[str, str]]]"
    steps: "list[StepVM]"
    highlights: "list[ItemVM]"
    decimal: bool = False


def build_vm(result, decimal: bool = False) -> ResultVM:
    """Macht aus einem anakit-Ergebnis das, was die Karten zeigen."""
    digits = getattr(result, "digits", 4)
    result.decimal = decimal
    steps = []
    for index, step in enumerate(result.steps, 1):
        steps.append(StepVM(
            index=index, title=step.title, rule=step.rule or "",
            detail=split_math(step.detail) if step.detail else [],
            items=[item_vm(n, o, decimal, digits) for n, o in step.items],
            note=step.note or ""))
    return ResultVM(
        title=tex_to_text(result.title),
        summary=split_math(result.summary) if result.summary else [],
        warnings=[split_math(w) for w in result.warnings],
        steps=steps,
        highlights=[item_vm(n, o, decimal, digits, big=True) for n, o in result.highlights],
        decimal=decimal)


#: Schriftgrössen der Formeln in logischen Bildpunkten; setzt `setup_math_sizes`.
MATH_PX = {"detail": 15, "item": 17, "big": 20}


def setup_math_sizes():
    """Nach dem Start: die Formelgrösse an die Schrift der App anpassen."""
    def em(pt: int) -> int:
        return QFontInfo(FONTS.f(pt)).pixelSize() or int(pt * 4 / 3)
    MATH_PX["detail"] = int(em(11) * 1.18)
    MATH_PX["item"] = int(em(12) * 1.2)
    MATH_PX["big"] = int(em(13) * 1.4)


def warm_vm(vm: ResultVM):
    """Setzt alle Formeln eines Ergebnisses vor — im Hintergrundfaden."""
    for step in vm.steps:
        for kind, piece in step.detail:
            if kind == "m":
                render_mask(piece, MATH_PX["detail"])
        for item in step.items:
            _warm_item(item, MATH_PX["item"])
    for kind, piece in vm.summary:
        if kind == "m":
            render_mask(piece, MATH_PX["detail"])
    for item in vm.highlights:
        _warm_item(item, MATH_PX["big"])


def _warm_item(item: ItemVM, px: int):
    if item.tex:
        render_mask(item.tex, px)
    for line in item.lines or []:
        render_mask(line, px)


# ==========================================================================
# Bausteine
# ==========================================================================

class Card(QFrame):
    """Fläche mit runden Ecken und weichem Schatten."""

    def __init__(self, parent=None, pad: int = 26):
        super().__init__(parent)
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(pad, pad, pad, pad)
        self.lay.setSpacing(0)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.setGraphicsEffect(self.shadow)
        self.restyle()

    def restyle(self):
        self.shadow.setBlurRadius(30)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(0, 0, 0, T["shadow"] // (3 if T["dark"] else 1)))


class Segmented(QWidget):
    """Segmentierte Umschaltleiste wie in iOS; jedes Segment so breit wie sein Text."""

    changed = Signal(int)

    def __init__(self, options: Sequence[str], value: int = 0,
                 seg_width: int | None = None, parent=None, pad: int = 30):
        super().__init__(parent)
        self.options = list(options)
        self.value = value
        self.enabled = True
        self.disabled_hint = ""
        self.font_ = FONTS.f(11)
        metrics = QFontMetricsF(self.font_)
        self.widths = [seg_width or int(metrics.horizontalAdvance(o)) + pad
                       for o in self.options]
        self.setFixedSize(sum(self.widths) + 6, 30)
        self.setCursor(Qt.PointingHandCursor)

    def _box(self, i: int) -> QRectF:
        return QRectF(3 + sum(self.widths[:i]), 3, self.widths[i], self.height() - 6)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qcolor("fill2") if not self.enabled else qcolor("fill"))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 0, 0)
        p.setFont(self.font_)
        for i, text in enumerate(self.options):
            box = self._box(i)
            if i == self.value:
                p.setBrush(qcolor("fill") if not self.enabled else qcolor("seg"))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(box, 0, 0)
            if not self.enabled:
                p.setPen(qcolor("ter"))
            else:
                p.setPen(qcolor("text") if i == self.value else qcolor("sec"))
            p.drawText(box, Qt.AlignCenter, text)

    def mousePressEvent(self, event):
        if not self.enabled:
            return
        px = event.position().x() - 3
        edge = 0
        for i, w in enumerate(self.widths):
            edge += w
            if px < edge:
                if i != self.value:
                    self.value = i
                    self.update()
                    self.changed.emit(i)
                return

    def set_value(self, i: int):
        self.value = i
        self.update()

    def set_enabled(self, on: bool):
        """Schaltet die Leiste stumm: sie bleibt stehen, nimmt aber keine Klicks."""
        if self.enabled == on:
            return
        self.enabled = on
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        self.setToolTip("" if on else self.disabled_hint)
        self.update()


class Stepper(QWidget):
    """Minus / Wert / Plus in einer Pille."""

    stepped = Signal(int)

    def __init__(self, value: int, parent=None):
        super().__init__(parent)
        self.value = value
        self.font_ = FONTS.f(12)
        self.setFixedSize(96, 30)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qcolor("fill"))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 0, 0)
        p.setFont(self.font_)
        p.setPen(qcolor("text"))
        h = self.height()
        p.drawText(QRectF(0, 0, 32, h), Qt.AlignCenter, MINUS)
        p.drawText(QRectF(32, 0, 32, h), Qt.AlignCenter, str(self.value))
        p.drawText(QRectF(64, 0, 32, h), Qt.AlignCenter, "+")

    def mousePressEvent(self, event):
        px = event.position().x()
        if px < 32:
            self.stepped.emit(-1)
        elif px > 64:
            self.stepped.emit(1)

    def set_value(self, value: int):
        self.value = value
        self.update()


class Dropdown(QPushButton):
    """Auswahlfeld mit Menü."""

    picked = Signal(str)

    def __init__(self, options: Sequence[str], value: str | None = None,
                 placeholder: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("dropdown")
        self.options = list(options)
        self.value = value if value is not None else (self.options[0] if self.options else "")
        self.placeholder = placeholder
        self.setText(f"  {placeholder or self.value}   ⌄")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.clicked.connect(self._open)

    def set_options(self, options: Sequence[str]):
        self.options = list(options)

    def _open(self):
        if not self.options:
            return
        menu = QMenu(self)
        menu.setFont(FONTS.f(11))
        for opt in self.options:
            menu.addAction(opt, lambda o=opt: self._choose(o))
        menu.exec(self.mapToGlobal(QPoint(0, self.height() + 2)))

    def _choose(self, opt: str):
        self.value = opt
        if not self.placeholder:
            self.setText(f"  {opt}   ⌄")
        self.picked.emit(opt)

    def set_value(self, value: str):
        self.value = value
        if not self.placeholder:
            self.setText(f"  {value}   ⌄")


def pill(text: str, kind: str = "primary", on_click: Callable | None = None,
         small: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName(kind)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedHeight(30 if small else 38)
    btn.setFont(FONTS.f(11 if small else 12))
    if on_click:
        btn.clicked.connect(on_click)
    return btn


def label(text: str, size: int = 12, color: str = "text", weight=QFont.Normal,
          display: bool = False, mono: bool = False, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setFont(FONTS.f(size, weight, display, mono))
    lb.setProperty("colorKey", color)
    lb.setStyleSheet(f"color: {T[color]}; background: transparent;")
    lb.setWordWrap(wrap)
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if wrap:
        lb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
    return lb


def wrapped(text: str, width: int, size: int = 11, color: str = "sec") -> QLabel:
    """
    Umbrochener Text mit selbst berechneter Höhe.

    Qt bestimmt die Höhe eines Labels, bevor es den Umbruch kennt; in einem
    Rahmen mit fester Breite bleibt sonst nur die erste Zeile sichtbar.
    """
    lb = label(text, size, color, wrap=True)
    lb.setFixedWidth(width)
    metrics = QFontMetricsF(lb.font())
    box = metrics.boundingRect(QRectF(0, 0, width, 10000), int(Qt.TextWordWrap), text)
    lb.setMinimumHeight(int(box.height()) + 4)
    return lb


def entry(text: str = "", width: int | None = None, placeholder: str = "",
          size: int = 13, height: int = 36) -> QLineEdit:
    """Ein Eingabefeld im Stil der App."""
    field = QLineEdit(text)
    field.setObjectName("cell")
    field.setFont(FONTS.f(size))
    field.setFixedHeight(height)
    if width:
        field.setFixedWidth(width)
    if placeholder:
        field.setPlaceholderText(placeholder)
    return field


class Badge(QLabel):
    def __init__(self, text: str, fg: str, bg: str, parent=None, small: bool = False):
        super().__init__(text, parent)
        self.fg, self.bg, self.small = fg, bg, small
        self.setFont(FONTS.f(10 if small else 11, QFont.DemiBold))
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.restyle()

    def restyle(self):
        pad = "2px 9px" if self.small else "4px 14px"
        self.setStyleSheet(
            f"color: {T[self.fg]}; background: {T[self.bg]};"
            f"border-radius: 0px; padding: {pad};")


def restyle_labels(root: QWidget):
    """Färbt alle mit label() erzeugten Beschriftungen neu ein."""
    for lb in root.findChildren(QLabel):
        if isinstance(lb, Badge):
            lb.restyle()
        elif isinstance(lb, RichLabel):
            lb.restyle()
        elif lb.property("colorKey"):
            lb.setStyleSheet(
                f"color: {T[lb.property('colorKey')]}; background: transparent;")
    for fl in root.findChildren(FormulaLabel):
        fl.update()


class Toast(QFrame):
    """Kurze Rückmeldung unten rechts, die sich selbst wieder ausblendet."""

    HEIGHT = 40

    def __init__(self, parent: QWidget, text: str, kind: str = "info"):
        super().__init__(parent)
        self.setObjectName("toast")
        tone = {"info": "text", "ok": "green", "bad": "red"}[kind]
        glyph = {"info": "", "ok": "✓  ", "bad": "✕  "}[kind]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 18, 0)
        self.label = label(glyph + text, 11, tone, weight=QFont.DemiBold)
        lay.addWidget(self.label)
        self.setFixedHeight(self.HEIGHT)
        self.setStyleSheet(
            f"#toast {{ background: {T['card']}; border-radius: 0px;"
            f"border: 1px solid {T['line']}; }}")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 60 if not T["dark"] else 150))
        self.setGraphicsEffect(shadow)
        self.adjustSize()
        self.setFixedWidth(max(180, self.label.sizeHint().width() + 40))


class ToastArea(QObject):
    """Stapelt die Meldungen und räumt sie nach ein paar Sekunden weg."""

    def __init__(self, host: QWidget):
        super().__init__(host)
        self.host = host
        self.items: list[Toast] = []

    def show(self, text: str, kind: str = "info", ms: int = 2600):
        toast = Toast(self.host, text, kind)
        self.items.append(toast)
        if len(self.items) > 3:
            self._drop(self.items[0])
        toast.show()
        self._layout()
        QTimer.singleShot(ms, lambda: self._drop(toast))

    def _drop(self, toast: Toast):
        if toast not in self.items:
            return
        self.items.remove(toast)
        toast.hide()
        toast.deleteLater()
        self._layout()

    def _layout(self):
        bottom = self.host.height() - 24
        for toast in reversed(self.items):
            bottom -= Toast.HEIGHT + 8
            toast.move(self.host.width() - toast.width() - 24, bottom)
            toast.raise_()

    def reposition(self):
        self._layout()


# ==========================================================================
# Gesetzte Formeln im Fenster
# ==========================================================================

class RichLabel(QLabel):
    """
    Ein Absatz mit Text und eingebetteten Formeln; bricht um, wie ein Label es tut.

    Die Formeln stehen als Bilder (data-URI) im Fliesstext, mit der
    Mathe-Achse auf Höhe der Textmitte.
    """

    def __init__(self, parts: "list[tuple[str, str]]", size: int = 11,
                 color: str = "sec", weight=QFont.Normal, math_key: str = "detail",
                 parent=None):
        super().__init__(parent)
        self.parts, self.color_key, self.math_key = parts, color, math_key
        self.setFont(FONTS.f(size, weight))
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.restyle()

    def restyle(self):
        col = T[self.color_key]
        self.setStyleSheet(f"color: {col}; background: transparent;")
        math_col = T["text"] if self.color_key in ("sec", "ter") else col
        out = []
        for kind, piece in self.parts:
            if kind == "m":
                uri = mask_data_uri(piece, MATH_PX[self.math_key], math_col)
                if uri is not None:
                    src, w, h = uri
                    out.append(f'<img src="{src}" width="{w:.1f}" height="{h:.1f}" '
                               f'style="vertical-align: middle;">')
                else:
                    out.append(html.escape(tex_to_text(piece)))
            else:
                esc = html.escape(piece).replace("\n", "<br>")
                if esc.startswith(" "):
                    esc = "&nbsp;" + esc[1:]
                if esc.endswith(" "):
                    esc = esc[:-1] + "&nbsp;"
                out.append(esc)
        self.setText("".join(out))

    def heightForWidth(self, w: int) -> int:
        return super().heightForWidth(w)


class FormulaLabel(QWidget):
    """
    Eine Zeile (oder ein paar Zeilen) gesetzter Formel, in Textfarbe gezeichnet.

    Ist sie breiter als der Platz, wird sie verkleinert statt abgeschnitten.
    Hat Mathtext sie nicht setzen können, steht der Unicode-Text da.
    """

    MIN_SCALE = 0.55

    def __init__(self, item: ItemVM, px: int, parent=None):
        super().__init__(parent)
        self.item, self.px = item, px
        lines = item.lines if item.lines else ([item.tex] if item.tex else [])
        masks = [render_mask(t, px) for t in lines]
        self.masks = masks if masks and all(m is not None for m in masks) else None
        self._cache: dict = {}
        self.gap = 4
        if self.masks:
            sizes = [m.size for m in self.masks]
            self.nat_w = max(w for w, _ in sizes) + 2
            self.nat_h = sum(h for _, h in sizes) + self.gap * (len(sizes) - 1) + 2
        else:
            self.font_ = FONTS.f(12)
            fm = QFontMetricsF(self.font_)
            self.nat_w = fm.horizontalAdvance(item.text) + 4
            self.nat_h = fm.height() + 4
        self.setFixedHeight(int(self.nat_h + 0.5))

    def natural_width(self) -> float:
        return self.nat_w

    def fit(self, avail: float) -> QSize:
        """Grösse bei verfügbarer Breite: verkleinert, wenn nötig."""
        if self.masks and self.nat_w > avail > 0:
            s = max(self.MIN_SCALE, avail / self.nat_w)
        else:
            s = 1.0
        return QSize(int(min(self.nat_w, avail) if avail > 0 else self.nat_w),
                     int(self.nat_h * s + 0.5))

    def sizeHint(self) -> QSize:
        return QSize(int(self.nat_w), int(self.nat_h + 0.5))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform
                         | QPainter.TextAntialiasing)
        color = T["text"]
        if not self.masks:
            p.setFont(self.font_)
            p.setPen(QColor(color))
            p.drawText(QRectF(0, 0, self.width(), self.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, self.item.text)
            return
        scale = 1.0
        if self.nat_w > self.width() > 0:
            scale = max(self.MIN_SCALE, self.width() / self.nat_w)
        y = 1.0
        dpr = self.devicePixelRatioF()
        for mask in self.masks:
            key = (id(mask), color, round(scale * 100), dpr)
            if key not in self._cache:
                self._cache[key] = scaled_image(mask, color, scale, dpr)
            image, w, h = self._cache[key]
            p.drawImage(QPointF(1, y), image)
            y += h + self.gap * scale


class FlowBox(QWidget):
    """Legt Kinder nebeneinander und bricht in eine neue Zeile, wenn der Platz fehlt."""

    def __init__(self, hgap: int = 26, vgap: int = 8, parent=None):
        super().__init__(parent)
        self.kids: list[FormulaLabel] = []
        self.hgap, self.vgap = hgap, vgap
        self._last_width = -1

    def add(self, kid: "FormulaLabel"):
        kid.setParent(self)
        self.kids.append(kid)
        kid.show()
        self.relayout(max(self.width(), 560))

    def relayout(self, width: int):
        if width <= 0:
            return
        x0 = y0 = row_h = 0
        for kid in self.kids:
            size = kid.fit(width)
            if x0 and x0 + size.width() > width:
                x0, y0, row_h = 0, y0 + row_h + self.vgap, 0
            kid.setGeometry(x0, y0, size.width(), size.height())
            x0 += size.width() + self.hgap
            row_h = max(row_h, size.height())
        total = y0 + row_h
        if self.height() != total:
            self.setFixedHeight(total)
        self._last_width = width

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != self._last_width:
            self.relayout(event.size().width())


# ==========================================================================
# Ergebnisse darstellen
# ==========================================================================

class ResultView(QScrollArea):
    """Zeigt ein anakit-Ergebnis: Titel, Schrittkarten mit gesetzten Formeln, Endwerte."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.host = QWidget()
        self.host.setObjectName("page")
        self.lay = QVBoxLayout(self.host)
        self.lay.setContentsMargins(0, 0, 8, 0)
        self.lay.setSpacing(10)
        self.setWidget(self.host)
        self.vm: ResultVM | None = None
        self.message = ("info", "Noch nichts gerechnet.")
        self._placeholder()

    def _clear(self):
        while self.lay.count():
            item = self.lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _placeholder(self, text: str = "Noch nichts gerechnet."):
        self._clear()
        self.vm = None
        self.message = ("info", text)
        self.lay.addWidget(label(text, 12, "ter", wrap=True))
        self.lay.addStretch(1)

    def placeholder(self, text: str):
        self._placeholder(text)

    def show_error(self, message: str):
        self._clear()
        self.vm = None
        self.message = ("error", message)
        box = QFrame()
        box.setObjectName("errbox")
        box.setStyleSheet(f"#errbox {{ background: {T['red_soft']}; border-radius: 0px; }}")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(16, 13, 16, 13)
        bl.setSpacing(4)
        bl.addWidget(label("Das hat nicht geklappt", 12, "red", QFont.DemiBold))
        bl.addWidget(label(message, 12, "red", wrap=True))
        self.lay.addWidget(box)
        self.lay.addStretch(1)

    def restyle(self):
        """Nach einem Themawechsel dieselbe Anzeige in neuen Farben aufbauen."""
        if self.vm is not None:
            self.show_vm(self.vm)
        elif self.message[0] == "error":
            self.show_error(self.message[1])
        else:
            self._placeholder(self.message[1])

    def show_result(self, result, decimal: bool = False):
        self.show_vm(build_vm(result, decimal))

    def show_vm(self, vm: ResultVM):
        self._clear()
        self.vm = vm
        self.message = ("info", "")
        head = QVBoxLayout()
        head.setSpacing(3)
        title = label(vm.title, 15, weight=QFont.DemiBold, wrap=True)
        head.addWidget(title)
        if vm.summary:
            head.addWidget(RichLabel(vm.summary, 11, "sec"))
        holder = QWidget()
        holder.setLayout(head)
        self.lay.addWidget(holder)

        for warning in vm.warnings:
            box = QFrame()
            box.setObjectName("warnbox")
            box.setStyleSheet(f"#warnbox {{ background: {T['amber_soft']}; border-radius: 0px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.addWidget(RichLabel(warning, 11, "amber"))
            self.lay.addWidget(box)

        for step in vm.steps:
            self.lay.addWidget(self._step_card(step))

        if vm.highlights:
            self.lay.addWidget(self._result_card(vm.highlights))
        self.lay.addStretch(1)

    def _step_card(self, step: StepVM) -> QWidget:
        card = QFrame()
        card.setObjectName("stepcard")
        card.setStyleSheet(f"#stepcard {{ background: {T['fill2']}; border-radius: 0px;"
                           f"border: 1px solid {T['hair']}; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 13, 16, 13)
        lay.setSpacing(5)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(label(f"{step.index}. {step.title}", 12, weight=QFont.DemiBold),
                       0, Qt.AlignVCenter)
        if step.rule:
            head.addWidget(Badge(step.rule, "blue", "blue_soft", small=True), 0, Qt.AlignVCenter)
        head.addStretch(1)
        lay.addLayout(head)
        if step.detail:
            lay.addWidget(RichLabel(step.detail, 11, "sec"))
        if step.items:
            flow = FlowBox()
            for item in step.items:
                flow.add(FormulaLabel(item, MATH_PX["item"]))
            lay.addWidget(flow)
        if step.note:
            lay.addWidget(label(step.note, 10, "ter", wrap=True))
        return card

    def _result_card(self, highlights: "list[ItemVM]") -> QWidget:
        card = QFrame()
        card.setObjectName("rescard")
        card.setStyleSheet(f"#rescard {{ background: {T['blue_soft']}; border-radius: 0px; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 13, 16, 14)
        lay.setSpacing(6)
        lay.addWidget(label("Ergebnis", 10, "blue", QFont.DemiBold))
        flow = FlowBox(hgap=30, vgap=8)
        for item in highlights:
            flow.add(FormulaLabel(item, MATH_PX["big"]))
        lay.addWidget(flow)
        return card


# ==========================================================================
# Zeichnungen im Fenster
# ==========================================================================

def theme_figure(figure):
    """
    Passt eine Matplotlib-Figur an das gerade gewählte Thema an.

    Die Zeichnungen aus anakit.plots kennen `dark=` und kommen schon im
    richtigen Thema an; hier werden nur noch Hintergründe, Achsen und
    Beschriftungen auf die Kartenfarbe der App gezogen, damit zwischen
    Zeichnung und Karte keine Kante steht. Die Linien der Zeichnung selbst
    bleiben unberührt — ihre Farben tragen Bedeutung.
    """
    card, text, sec, line = T["card"], T["text"], T["sec"], T["line"]
    figure.patch.set_facecolor(card)
    for ax in figure.axes:
        ax.set_facecolor(card)
        ax.title.set_color(text)
        ax.tick_params(colors=sec, which="both")
        for spine in ax.spines.values():
            spine.set_color(line)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_color(sec)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(card)
            legend.get_frame().set_edgecolor(line)
            for entry_ in legend.get_texts():
                entry_.set_color(text)


class PlotCanvas(QWidget):
    """Eine Zeichnung im Fenster selbst, nicht im Browser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self.figure = None
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(10)
        self.empty = label("Noch keine Zeichnung — erst rechnen.", 13, "sec", wrap=True)
        self.lay.addWidget(self.empty, 1, Qt.AlignCenter)
        self.readout = label("", 11, "sec")
        self.lay.addWidget(self.readout)
        self.hint = label("", 11, "ter", wrap=True)
        self.lay.addWidget(self.hint)
        self._drag = None

    @staticmethod
    def available() -> bool:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvas  # noqa
        except Exception:
            return False
        return True

    def show_figure(self, figure, hint: str = ""):
        """Zeigt eine Matplotlib-Figur; die vorige wird freigegeben."""
        from matplotlib.backends.backend_qtagg import FigureCanvas

        self.clear()
        theme_figure(figure)
        self.figure = figure
        self.canvas = FigureCanvas(figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.empty.hide()
        self.lay.insertWidget(0, self.canvas, 1)
        self.canvas.draw_idle()
        self.hint.setText(hint + ("  " if hint else "") +
                          "Mausrad zoomt, Ziehen verschiebt, Doppelklick setzt zurück.")
        self._home = [(ax, ax.get_xlim(), ax.get_ylim()) for ax in figure.axes]
        for ev, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press),
                       ("button_release_event", self._on_release),
                       ("motion_notify_event", self._on_move), ("axes_leave_event", self._on_leave)):
            self.canvas.mpl_connect(ev, fn)

    # Zoom, Verschieben und Koordinaten-Anzeige direkt auf der Zeichnung.
    def _on_scroll(self, ev):
        ax = ev.inaxes
        if ax is None or ev.xdata is None:
            return
        k = 0.8 if ev.step > 0 else 1.25
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ev.xdata - (ev.xdata - x0) * k, ev.xdata + (x1 - ev.xdata) * k)
        ax.set_ylim(ev.ydata - (ev.ydata - y0) * k, ev.ydata + (y1 - ev.ydata) * k)
        self.canvas.draw_idle()

    def _on_press(self, ev):
        if ev.inaxes is None or ev.button != 1:
            return
        if ev.dblclick:
            for ax, xl, yl in self._home:
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            self.canvas.draw_idle()
        else:
            self._drag = (ev.inaxes, ev.x, ev.y, ev.inaxes.get_xlim(), ev.inaxes.get_ylim())
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def _on_release(self, ev):
        self._drag = None
        self.canvas.setCursor(Qt.ArrowCursor)

    def _on_move(self, ev):
        if self._drag is not None:
            ax, px, py, (x0, x1), (y0, y1) = self._drag
            inv = ax.transData.inverted()
            (ax_, ay_), (bx_, by_) = inv.transform((px, py)), inv.transform((ev.x, ev.y))
            dx, dy = ax_ - bx_, ay_ - by_
            ax.set_xlim(x0 + dx, x1 + dx)
            ax.set_ylim(y0 + dy, y1 + dy)
            self.canvas.draw_idle()
            return
        if ev.inaxes is None or ev.xdata is None:
            return self._on_leave(ev)
        text = f"x = {ev.xdata:.3f}   y = {ev.ydata:.3f}"
        for ln in ev.inaxes.get_lines():          # Wert der ersten Kurve unter dem Mauszeiger
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs) > 50 and xs[0] <= ev.xdata <= xs[-1]:
                text += f"   Kurve: {np.interp(ev.xdata, xs, ys):.3f}"
                break
        self.readout.setText(text)

    def _on_leave(self, ev=None):
        self.readout.setText("")

    def show_message(self, text: str):
        self.clear()
        self.empty.setText(text)
        self.empty.show()
        self.hint.setText("")

    def clear(self):
        self._drag = None
        self.readout.setText("")
        if self.canvas is not None:
            self.canvas.hide()
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = None

    def restyle(self):
        if self.figure is not None and self.canvas is not None:
            theme_figure(self.figure)
            self.canvas.draw_idle()


class PlotPane(QWidget):
    """
    Die Ansicht »Zeichnung«: Leinwand, Umschalter zwischen Zeichnungen,
    Bild speichern und — wenn Plotly da ist — dasselbe interaktiv im Browser.
    """

    variant_changed = Signal(int)
    browser_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.head = QHBoxLayout()
        self.head.setSpacing(8)
        self.seg_holder = QHBoxLayout()
        self.head.addLayout(self.seg_holder)
        self.head.addStretch(1)
        self.btn_save = pill("Bild speichern", "secondary", None, small=True)
        self.head.addWidget(self.btn_save)
        self.btn_browser = pill("Interaktiv im Browser", "plain",
                                lambda: self.browser_requested.emit(), small=True)
        self.btn_browser.setVisible(ak.has_plotly())
        self.head.addWidget(self.btn_browser)
        lay.addLayout(self.head)
        self.canvas = PlotCanvas()
        lay.addWidget(self.canvas, 1)
        self.seg = None
        self.names: list[str] = []

    @property
    def figure(self):
        return self.canvas.figure

    def set_variants(self, names: Sequence[str], current: int = 0):
        """Zeigt den Umschalter nur, wenn es mehr als eine Zeichnung gibt."""
        if list(names) == self.names:
            if self.seg is not None:
                self.seg.set_value(current)
            return
        self.names = list(names)
        if self.seg is not None:
            self.seg.hide()
            self.seg.setParent(None)
            self.seg.deleteLater()
            self.seg = None
        if len(names) > 1:
            self.seg = Segmented(names, current)
            self.seg.changed.connect(self.variant_changed.emit)
            self.seg_holder.addWidget(self.seg)

    def show_figure(self, figure, hint: str = ""):
        self.canvas.show_figure(figure, hint)

    def show_message(self, text: str):
        self.canvas.show_message(text)

    def restyle(self):
        self.canvas.restyle()


def save_figure_dialog(parent, figure, default_name: str = "anakit-zeichnung") -> str:
    """Fragt nach einem Dateinamen und speichert die Figur; gibt den Pfad oder '' zurück."""
    if figure is None:
        return ""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Zeichnung speichern", str(Path.home() / f"{default_name}.png"),
        "PNG-Bild (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if not path:
        return ""
    figure.savefig(path, dpi=200, facecolor=figure.get_facecolor(), bbox_inches="tight")
    return path


# ==========================================================================
# Prüfung im Hintergrund
# ==========================================================================

def selftest_report(appid: str) -> WolframReport:
    """Die Verbindungsprüfung als Bericht: Namensauflösung, TLS, Testabfrage."""
    report = WolframReport()
    for name, ok, note in network_selftest(appid):
        report.checks.append(Check(name, "ok" if ok else "bad", note=note))
    return report


class ComputeWorker(QObject):
    """Rechnet eine Seite (Ergebnis, Ansichtsmodell, Zeichnung) ausserhalb des Fensters."""

    finished = Signal(int, object)

    def __init__(self, job_id: int, run):
        super().__init__()
        self.job_id, self.run_job = job_id, run

    def run(self):
        try:
            outcome = self.run_job()
        except BaseException as exc:            # nie den Faden mit Ausnahme beenden
            outcome = Outcome(error=friendly_error(exc))
        self.finished.emit(self.job_id, outcome)


@dataclass
class Outcome:
    """Was ein Rechenlauf liefert: Ergebnis samt Ansichtsmodell, oder einen Fehlersatz."""
    result: object = None
    vm: "ResultVM | None" = None
    figure: object = None
    figure_error: str = ""
    error: str = ""
    seconds: float = 0.0


#: pyplot teilt sich einen Zustand; zwei Zeichnungen gleichzeitig sind nicht sicher.
PLOT_LOCK = threading.Lock()


def run_page_job(page: str, p: dict, decimal: bool, dark: bool, variant: int = 0) -> Outcome:
    """
    Alles, was dauern kann, an einem Ort: rechnen, Formeln setzen, zeichnen.
    Läuft im Hintergrundfaden; gibt nie eine Ausnahme weiter.
    """
    import time
    start = time.time()
    try:
        result = calc(page, p)
    except InputError as exc:
        return Outcome(error=str(exc))
    except Exception as exc:
        return Outcome(error=friendly_error(exc))
    out = Outcome(result=result)
    try:
        out.vm = build_vm(result, decimal)
        warm_vm(out.vm)
    except Exception as exc:
        return Outcome(error=f"Der Rechenweg liess sich nicht darstellen: {exc}")
    try:
        with PLOT_LOCK:
            out.figure = make_figure(page, p, result, dark, variant)
    except NotImplementedError as exc:
        out.figure_error = str(exc) or "Zu dieser Rechnung gibt es keine Zeichnung."
    except Exception as exc:
        out.figure_error = f"Zeichnen ging nicht: {friendly_error(exc)}"
    out.seconds = time.time() - start
    return out


def run_figure_job(page: str, p: dict, result, dark: bool, variant: int) -> Outcome:
    """Nur die Zeichnung neu — nach Themawechsel oder Wechsel der Ansicht."""
    out = Outcome(result=result)
    try:
        with PLOT_LOCK:
            out.figure = make_figure(page, p, result, dark, variant)
    except NotImplementedError as exc:
        out.figure_error = str(exc) or "Zu dieser Rechnung gibt es keine Zeichnung."
    except Exception as exc:
        out.figure_error = f"Zeichnen ging nicht: {friendly_error(exc)}"
    return out


# ==========================================================================
# Animation (anim.py, freiwillig)
# ==========================================================================

#: Qualitätsstufen: (Manim-Stufe, Breite eines GIF, Bilder pro Sekunde, Faktor
#: für die geschätzte Rechenzeit).
ANIM_QUALITY = {
    "Normal": ("medium", 1280, 30, 1.0),
    "Scharf": ("high", 1920, 30, 2.4),
    "Sehr scharf": ("ultra", 2560, 30, 5.0),
}

#: Ohne QtMultimedia bleibt nur das GIF — kleiner, aber nur 256 Farben.
ANIM_FORMAT = "mp4" if HAS_VIDEO else "gif"

ANIM_INSTALL_HINT = ("pip install manimgx   # Rendering und Videoausgabe in einem Paket\n"
                     "(Manim CE mit ffmpeg geht als Rückfall)")

#: Die Szenen je Seite: Name im Auswahlfeld -> Schlüssel für anim.py.
ANIM_SCENES = {
    "steigung": {"Sekante → Tangente": "secant_to_tangent"},
    "limits": {"x nähert sich x₀": "limit_approach"},
    "derivative": {"f und f′": "derivative_graph"},
    "curve": {"Kurvendiskussion": "curve_analysis"},
    "numerik": {"Newton-Verfahren": "newton"},
}


def anim_state() -> tuple[bool, str]:
    """(bereit, Begründung) — einmal zentral statt an jeder Karte neu."""
    if ak_anim is None:
        return False, (ANIM_IMPORT_ERROR or "Die Animation (anim.py) fehlt in diesem Ordner.")
    check = getattr(ak_anim, "manim_available", None)
    if check is None:
        return False, "anim.py bietet manim_available() nicht an."
    try:
        return check()
    except Exception as exc:                                  # pragma: no cover
        return False, str(exc)


def anim_theme():
    """Die Farben der Animation aus dem gerade gewählten Thema."""
    theme_cls = getattr(ak_anim, "Theme", None)
    if theme_cls is None:
        return None
    base = theme_cls.dark() if T["dark"] else theme_cls()
    base.background, base.text, base.muted = T["card"], T["text"], T["sec"]
    base.accent = T["blue"]
    return base


def anim_render(job, out_dir, progress):
    return ak_anim.render(job, out_dir=out_dir, progress=progress)


class AnimationWorker(QObject):
    """Rendert einen Manim-Auftrag, ohne das Fenster zu blockieren."""

    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, job, out_dir: Path):
        super().__init__()
        self.job, self.out_dir = job, out_dir

    def run(self):
        try:
            path = anim_render(self.job, self.out_dir,
                               lambda p, t: self.progress.emit(p, t))
            self.finished.emit(str(path))
        except Exception as exc:                 # Manim wirft vielerlei
            self.failed.emit(str(exc))




def movie_size(path: str) -> QSize:
    """
    Die Bildgrösse einer Animation, ohne sie abzuspielen.

    Beim GIF liest QMovie das erste Bild. Beim MP4 steht die Auflösung im
    Ordnernamen, den Manim anlegt — „1080p60“ —, das erspart ffprobe.
    """
    if path.lower().endswith(".mp4"):
        for part in reversed(Path(path).parts):
            match = re.fullmatch(r"(\d+)p(\d+)", part)
            if match:
                height = int(match.group(1))
                return QSize(int(round(height * 16 / 9)), height)
        return QSize(1920, 1080)
    probe = QMovie(path)
    probe.jumpToFrame(0)
    size = probe.currentImage().size()
    if size.isValid() and size.width() > 0:
        return size
    return QSize(860, 484)


class AnimationWindow(QDialog):
    """Die Animation in voller Grösse, mit Platz für die ganze Auflösung."""

    def __init__(self, parent, path: str, title: str):
        super().__init__(parent)
        self.setWindowTitle("Animation")
        self.setModal(True)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        head = QHBoxLayout()
        head.addWidget(label(title, 13, weight=QFont.DemiBold))
        head.addStretch(1)
        head.addWidget(pill("Schliessen", "plain", self.accept, small=True))
        lay.addLayout(head)

        self.player = MoviePlayer()
        lay.addWidget(self.player, 1)

        source = movie_size(path)
        screen = parent.screen().availableGeometry()
        dpr = self.devicePixelRatioF()
        # Nicht über die echte Auflösung hinaus und nicht über den Bildschirm.
        width = min(int(source.width() / dpr), int(screen.width() * 0.85), 1400)
        height = int(width * source.height() / max(1, source.width()))
        self.resize(width + 40, height + 110)
        self.player.load(path, title)


class MoviePlayer(QWidget):
    """
    Spielt die gerenderte Animation ab.

    Zwei Wege, je nach Datei: ein MP4 über QtMultimedia — volle Farbtiefe und
    bis 1440p — oder ein GIF über QMovie als Rückfall. Beim GIF werden die
    Bilder selbst gesetzt statt über QLabel.setMovie: nur so lässt sich ihre
    Pixeldichte angeben, sonst zeigt ein Retina-Bildschirm es auf die
    doppelte Fläche aufgeblasen und damit unscharf.
    """

    clicked = Signal()

    #: Manim rendert immer im Breitbild; danach richtet sich die Höhe.
    ASPECT = 9 / 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self.movie: QMovie | None = None
        self.source_size = QSize()
        self.path = ""
        self.player = None
        self.audio = None
        self.aspect = self.ASPECT
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.stage = QStackedWidget()
        self.stage.setMinimumHeight(150)
        lay.addWidget(self.stage)

        self.screen = QLabel()
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setStyleSheet(
            f"background: {T['fill2']}; border-radius: 0px;")
        self.stage.addWidget(self.screen)

        self.video = None
        if HAS_VIDEO:
            self.video = QVideoWidget()
            self.video.setAspectRatioMode(Qt.KeepAspectRatio)
            self.video.setAutoFillBackground(True)
            self._paint_video_background()
            self.stage.addWidget(self.video)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.btn_play = pill("Pause", "secondary", self.toggle, small=True)
        controls.addWidget(self.btn_play)
        self.btn_restart = pill("Von vorn", "secondary", self.restart, small=True)
        controls.addWidget(self.btn_restart)
        controls.addStretch(1)
        self.lbl_info = label("", 10, "ter")
        controls.addWidget(self.lbl_info)
        lay.addLayout(controls)

    # -- Zustand ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self.movie is not None or self.player is not None

    def load(self, path: str, note: str = ""):
        self.stop()
        if path.lower().endswith(".mp4") and HAS_VIDEO:
            self._load_video(path)
        else:
            self._load_gif(path)
        self.lbl_info.setText(note)

    # -- MP4 --------------------------------------------------------------

    def _load_video(self, path: str):
        player = QMediaPlayer(self)
        # Ohne Tonausgabe meckert Qt auf manchen Rechnern; die Animation hat
        # keinen Ton, die Ausgabe bleibt darum stumm.
        self.audio = QAudioOutput(self)
        self.audio.setMuted(True)
        player.setAudioOutput(self.audio)
        player.setVideoOutput(self.video)
        player.setLoops(QMediaPlayer.Infinite)
        player.errorOccurred.connect(self._video_failed)
        player.setSource(QUrl.fromLocalFile(path))
        self.player = player
        self.path = path
        self.stage.setCurrentWidget(self.video)
        self.rescale()
        player.play()
        self.btn_play.setText("Pause")

    def _video_failed(self, _error, text: str):
        """Fällt auf das GIF neben dem Video zurück, wenn es eines gibt."""
        gif = Path(self.path).with_suffix(".gif") if self.path else None
        self._drop_player()
        if gif is not None and gif.exists():
            self._load_gif(str(gif))
            self.lbl_info.setText("Als GIF abgespielt.")
            return
        self.stage.setCurrentWidget(self.screen)
        self.screen.setText(f"Die Animation lässt sich nicht abspielen: {text}")

    def _paint_video_background(self):
        if self.video is None:
            return
        pal = self.video.palette()
        pal.setColor(self.video.backgroundRole(), QColor(T["fill2"]))
        self.video.setPalette(pal)

    # -- GIF --------------------------------------------------------------

    def _load_gif(self, path: str):
        movie = QMovie(path)
        if not movie.isValid():
            self.stage.setCurrentWidget(self.screen)
            self.screen.setText("Die Animation lässt sich nicht abspielen.")
            return
        movie.jumpToFrame(0)
        self.source_size = movie.currentImage().size()
        self.movie = movie
        self.path = path
        self.stage.setCurrentWidget(self.screen)
        movie.frameChanged.connect(self._show_frame)
        self.rescale()
        movie.start()
        self.btn_play.setText("Pause")

    def _show_frame(self):
        """Ein Bild übernehmen und ihm die Pixeldichte des Bildschirms geben."""
        if self.movie is None:
            return
        pix = self.movie.currentPixmap()
        if pix.isNull():
            return
        pix.setDevicePixelRatio(self.devicePixelRatioF())
        self.screen.setPixmap(pix)

    # -- Grösse -----------------------------------------------------------

    def rescale(self):
        if self.player is not None:
            # Die Höhe aus der Breite ableiten, sonst bleiben schwarze
            # Balken über und unter dem Bild stehen.
            width = max(220, self.width() - 4)
            self.stage.setFixedHeight(max(150, int(width * self.aspect)))
            return
        if self.movie is None or not self.source_size.isValid():
            return
        if self.source_size.width() <= 0:
            return
        dpr = self.devicePixelRatioF()
        available = max(220, self.width() - 4)
        # In echten Bildpunkten rechnen, dann in logische zurück.
        wanted = available * dpr
        scale = min(1.0, wanted / self.source_size.width())
        size = QSize(max(1, int(self.source_size.width() * scale)),
                     max(1, int(self.source_size.height() * scale)))
        self.movie.setScaledSize(size)
        self.stage.setFixedHeight(int(size.height() / dpr))
        self._show_frame()

    def mouseReleaseEvent(self, event):
        if self.loaded and self.stage.geometry().contains(
                event.position().toPoint()):
            self.clicked.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale()

    # -- Bedienung --------------------------------------------------------

    def toggle(self):
        if self.player is not None:
            running = self.player.playbackState() == QMediaPlayer.PlayingState
            self.player.pause() if running else self.player.play()
            self.btn_play.setText("Abspielen" if running else "Pause")
            return
        if self.movie is None:
            return
        running = self.movie.state() == QMovie.Running
        self.movie.setPaused(running)
        self.btn_play.setText("Abspielen" if running else "Pause")

    def restart(self):
        if self.player is not None:
            self.player.setPosition(0)
            self.player.play()
            self.btn_play.setText("Pause")
            return
        if self.movie is None:
            return
        self.movie.stop()
        self.movie.start()
        self.btn_play.setText("Pause")

    def _drop_player(self):
        if self.player is not None:
            try:
                self.player.errorOccurred.disconnect(self._video_failed)
            except (RuntimeError, TypeError):
                pass
            self.player.stop()
            self.player.setVideoOutput(None)
            self.player.deleteLater()
            self.player = None
        if self.audio is not None:
            self.audio.deleteLater()
            self.audio = None

    def stop(self):
        self._drop_player()
        if self.movie is not None:
            try:
                self.movie.frameChanged.disconnect(self._show_frame)
            except (RuntimeError, TypeError):
                pass
            self.movie.stop()
            self.movie = None
        self.path = ""
        self.screen.clear()
        self.stage.setCurrentWidget(self.screen)

    def restyle(self):
        self.screen.setStyleSheet(
            f"background: {T['fill2']}; border-radius: 0px;")
        self._paint_video_background()



class AppIDDialog(QDialog):
    def __init__(self, parent, current: str):
        super().__init__(parent)
        self.setWindowTitle("Wolfram|Alpha AppID")
        self.setModal(True)
        self.setFixedWidth(430)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24)
        lay.setSpacing(10)
        lay.addWidget(label("AppID hinterlegen", 15, weight=QFont.DemiBold))
        lay.addWidget(label(
            "Eine kostenlose AppID bekommst du unter developer.wolframalpha.com. "
            "Sie wird in deinem Benutzerordner gespeichert und nur für die "
            "Prüfanfragen verwendet.", 11, "sec", wrap=True))
        self.field = QLineEdit(current)
        self.field.setObjectName("cell")
        self.field.setFont(FONTS.f(12))
        self.field.setFixedHeight(38)
        self.field.setPlaceholderText("z. B. ABCDEF-GHIJKLMNOP")
        self.field.returnPressed.connect(self.accept)
        lay.addSpacing(6)
        lay.addWidget(self.field)
        lay.addSpacing(8)

        row = QHBoxLayout()
        link = pill("Seite öffnen", "plain", lambda: webbrowser.open(
            "https://developer.wolframalpha.com/access"), small=True)
        row.addWidget(link)
        row.addStretch(1)
        row.addWidget(pill("Abbrechen", "secondary", self.reject, small=True))
        row.addWidget(pill("Speichern", "primary", self.accept, small=True))
        lay.addLayout(row)

    def key(self) -> str:
        return self.field.text().strip()



class ChecksWorker(QObject):
    """Führt eine Prüffunktion aus, ohne das Fenster anzuhalten."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, run):
        super().__init__()
        self.run_checks = run

    def run(self):
        try:
            self.finished.emit(self.run_checks())
        except (RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                  # nichts darf den Faden still sterben lassen
            self.failed.emit(f"Die Prüfung ist fehlgeschlagen: {exc}")


class WolframPanel(QWidget):
    """
    Die Prüfung mit Wolfram|Alpha als eigene Karte.

    Sie kennt die Rechnung nicht: `build_checks(appid)` liefert die Aufgabe,
    die im Hintergrundfaden den Bericht baut; alles andere — AppID,
    Verbindungstest, Darstellung, Offline-Zustand — ist für jede Seite gleich.
    """

    def __init__(self, build_checks, subtitle: str, parent=None):
        super().__init__(parent)
        self.build_checks = build_checks
        self.report: WolframReport | None = None
        self.status: tuple[str, str] | None = None
        self.thread = None
        self.worker = None

        self.card = Card()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.card)
        lay = self.card.lay
        lay.addWidget(label("Prüfung mit Wolfram|Alpha", 15, weight=QFont.DemiBold))
        lay.addWidget(label(subtitle, 11, "sec", wrap=True))
        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn = pill("Jetzt prüfen", "primary", self.start)
        row.addWidget(self.btn)
        self.btn_key = pill("AppID", "secondary", self.ask_appid, small=True)
        row.addWidget(self.btn_key)
        self.btn_test = pill("Verbindung testen", "plain", self.start_selftest, small=True)
        row.addWidget(self.btn_test)
        row.addStretch(1)
        lay.addLayout(row)
        self.body = QVBoxLayout()
        self.body.setSpacing(0)
        lay.addSpacing(6)
        lay.addLayout(self.body)
        self.render()

    # -- Bedienung --------------------------------------------------------

    def ask_appid(self):
        dialog = AppIDDialog(self.window(), load_appid())
        if dialog.exec() == QDialog.Accepted and dialog.key():
            try:
                save_appid(dialog.key())
            except OSError as exc:
                self.status = ("bad", f"AppID nicht speicherbar: {exc}")
            self.render()

    def _run(self, job, busy_text: str):
        self.report = None
        self.status = ("busy", busy_text)
        self.render()
        self.thread = QThread(self)
        self.worker = ChecksWorker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def start(self):
        if self.thread is not None:
            return
        appid = load_appid()
        if not appid:
            self.ask_appid()
            appid = load_appid()
            if not appid:
                return
        try:
            job = self.build_checks(appid)
        except (ValueError, RuntimeError) as exc:
            self.status = ("bad", str(exc))
            self.render()
            return
        self._run(job, "Frage Wolfram|Alpha …")

    def start_selftest(self):
        if self.thread is not None:
            return
        appid = load_appid()
        self._run(lambda: selftest_report(appid), "Prüfe die Verbindung …")

    def _stop_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None

    def _done(self, report):
        self._stop_thread()
        self.report, self.status = report, None
        self.render()

    def _failed(self, message: str):
        self._stop_thread()
        self.report, self.status = None, ("bad", message)
        self.render()

    def wait(self, ms: int = 30000):
        """Wartet auf einen laufenden Lauf (für Tests); Ereignisse werden weiter verarbeitet."""
        import time
        end = time.time() + ms / 1000
        while self.thread is not None and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.01)

    def invalidate(self):
        """Nach einer Änderung an der Eingabe gilt der alte Bericht nicht mehr."""
        if self.thread is None and (self.report is not None or self.status is not None):
            self.report, self.status = None, None
            self.render()

    def set_enabled(self, on: bool, why: str = ""):
        self.btn.setEnabled(on and self.thread is None)
        self.btn.setToolTip("" if on else why)

    # -- Darstellung ------------------------------------------------------

    def render(self):
        while self.body.count():
            item = self.body.takeAt(0)
            # Nach setParent(None) gibt das Element den Verweis nicht mehr
            # heraus — darum vorher festhalten.
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.btn_key.setText("AppID ändern" if load_appid() else "AppID")

        if self.status:
            level, text = self.status
            color = {"busy": "sec", "bad": "red", "ok": "green"}[level]
            self.body.addSpacing(12)
            box = QFrame()
            box.setObjectName("wolferr" if level == "bad" else "wolfbusy")
            if level == "bad":
                box.setStyleSheet(f"#wolferr {{ background: {T['red_soft']}; border-radius: 0px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.addWidget(label(text, 11, color, wrap=True))
            if level == "bad":
                bl.addWidget(label("Die Rechnung selbst ist davon nicht betroffen — "
                                   "nur die Gegenprobe fehlt.", 10, "ter", wrap=True))
            self.body.addWidget(box)
            self.btn.setEnabled(level != "busy")
            self.btn.setText("Prüfe …" if level == "busy" else "Jetzt prüfen")
            return

        self.btn.setEnabled(True)
        self.btn.setText("Jetzt prüfen")
        if self.report is None:
            if not load_appid():
                self.body.addSpacing(12)
                self.body.addWidget(label(
                    "Dafür braucht es einmalig eine kostenlose AppID.",
                    11, "ter", wrap=True))
            return

        report = self.report
        self.body.addSpacing(14)
        tone = {"ok": ("green", "green_soft"), "bad": ("red", "red_soft"),
                "warn": ("amber", "amber_soft")}[report.verdict]
        mark = {"ok": "✓", "bad": "✕", "warn": "!"}[report.verdict]
        self.body.addWidget(Badge(f"{mark}  {report.headline}", *tone))
        self.body.addSpacing(10)

        for check in report.checks:
            glyph = {"ok": "✓", "bad": "✕", "info": "•", "error": "!"}[check.status]
            color = {"ok": "green", "bad": "red", "info": "sec", "error": "amber"}[check.status]
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(label(glyph, 11, color, weight=QFont.DemiBold), 0, Qt.AlignTop)
            text = check.name
            if check.theirs and check.theirs != check.ours:
                text += f":  wir {check.ours}, Wolfram {check.theirs}"
            elif check.ours:
                text += f":  {check.ours}"
            row.addWidget(label(text, 11, wrap=True), 1)
            holder = QWidget()
            holder.setLayout(row)
            self.body.addWidget(holder)
            if check.note:
                self.body.addWidget(label("     " + check.note, 10, "ter", wrap=True))

        if report.texts:
            self.body.addSpacing(12)
            self.body.addWidget(label("Wolframs Antworten", 11, "sec"))
            self.body.addSpacing(4)
            for name, text in report.texts[:6]:
                self.body.addWidget(label(f"{name}:", 10, "ter"))
                self.body.addWidget(label("  " + text.replace("-", MINUS), 11, mono=True, wrap=True))

        if report.url:
            self.body.addSpacing(10)
            row = QHBoxLayout()
            row.addWidget(pill("Bei Wolfram öffnen", "plain",
                               lambda: webbrowser.open(report.url), small=True))
            row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            self.body.addWidget(holder)

    def restyle(self):
        self.card.restyle()
        self.render()

    def shutdown(self):
        self._stop_thread()



class AnimationPane(QWidget):
    """
    Die Animation als eigenständiger Bereich, für jede Seite verwendbar.

    Sie weiss nichts über die Rechnung. Zwei Rückrufe genügen ihr:
    `build_job(szene, qualität)` liefert einen Auftrag oder einen Text, der
    erklärt, warum es hier nichts zu zeigen gibt; `key_of()` liefert einen
    Fingerabdruck der Eingabe, damit ein veralteter Film als solcher
    gekennzeichnet wird statt stillschweigend falsch dazustehen.
    """

    def __init__(self, build_job, key_of, scenes: dict | None = None,
                 subtitle: str = "", parent=None):
        super().__init__(parent)
        self.build_job = build_job
        self.key_of = key_of
        self.scenes = scenes
        self.scene = next(iter(scenes.values())) if scenes else "full"
        self.quality = "Normal"
        self.thread = None
        self.worker = None
        self.out_dir: Path | None = None
        self.stamp = None
        self.ready, why = anim_state()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.sub = label(subtitle or "Zeigt die Rechnung als kurzen Film.", 11, "sec",
                         wrap=True)
        lay.addWidget(self.sub)

        if not self.ready:
            # Nicht verstecken: Wer die Animation sucht, soll lesen können,
            # warum sie fehlt und was dagegen hilft.
            hint = getattr(ak_anim, "INSTALL_HINT", ANIM_INSTALL_HINT)
            self.sub.setText(f"{why}\n\nSo kommt sie dazu:\n{hint}")
            lay.addStretch(1)
            return

        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        if scenes:
            start = next(k for k, v in scenes.items() if v == self.scene)
            self.pick = Dropdown(list(scenes), value=start)
            self.pick.setFixedWidth(162)
            self.pick.picked.connect(self._set_scene)
            row.addWidget(self.pick)
        self.quality_pick = Dropdown(list(ANIM_QUALITY), value=self.quality)
        self.quality_pick.setFixedWidth(118)
        self.quality_pick.picked.connect(self._set_quality)
        row.addWidget(self.quality_pick)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn = pill("Rendern", "primary", self.render)
        row2.addWidget(self.btn)
        self.btn_big = pill("Gross anzeigen", "secondary", self.show_big,
                            small=True)
        self.btn_big.hide()
        row2.addWidget(self.btn_big)
        row2.addStretch(1)
        lay.addSpacing(10)
        lay.addLayout(row2)

        self.hint = label("", 10, "ter", wrap=True)
        lay.addWidget(self.hint)

        self.bar = QProgressBar()
        self.bar.setObjectName("progress")
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        lay.addSpacing(10)
        lay.addWidget(self.bar)
        self.status = label("", 11, "sec", wrap=True)
        lay.addWidget(self.status)

        self.player = MoviePlayer()
        self.player.hide()
        self.player.clicked.connect(self.show_big)
        lay.addSpacing(12)
        lay.addWidget(self.player)
        lay.addStretch(1)
        self.refresh()

    # -- Auswahl ----------------------------------------------------------

    def _set_scene(self, name: str):
        self.scene = self.scenes[name]
        self.refresh()

    def _set_quality(self, name: str):
        self.quality = name
        self.refresh()

    def refresh(self):
        """Prüft, ob der gezeigte Film noch zur Eingabe passt."""
        if not self.ready:
            return
        busy = self.thread is not None
        self.btn.setEnabled(not busy)
        if busy:
            return
        if self.player.loaded and self.stamp is not None:
            if self.stamp != self._stamp_now():
                self.hint.setText("Zeigt einen früheren Stand. Neu rendern "
                                  "übernimmt die jetzige Eingabe.")
                return
            self.hint.setText("")
            return
        self.hint.setText(
            f"Ungefähr {self.estimate()} Sekunden Rechenzeit — das Rendern "
            "läuft im Hintergrund, die App bleibt bedienbar.")

    def estimate(self) -> int:
        """
        Grobe Schätzung der Rechenzeit.

        Gemessen auf einem Rechner ohne Grafikbeschleunigung: eine räumliche
        Szene mit Kamerafahrt braucht am längsten, eine Matrixfolge wächst
        mit der Zahl der Schritte.
        """
        factor = ANIM_QUALITY[self.quality][3]
        if self.scenes is None:            # eine Vektorszene
            return int(45 * factor)
        return int(25 * factor)

    def _stamp_now(self):
        return (self.key_of(), self.scene, self.quality, T["dark"])

    # -- Rendern ----------------------------------------------------------

    def render(self):
        if not self.ready or self.thread is not None:
            return
        quality, width, fps, _ = ANIM_QUALITY[self.quality]
        try:
            job = self.build_job(self.scene, quality)
        except Exception as exc:
            self._fail(str(exc))
            return
        if isinstance(job, str):              # nichts zu zeigen, mit Grund
            self._fail(job)
            return
        for attr, value in (("fmt", ANIM_FORMAT), ("gif_width", width),
                            ("gif_fps", fps), ("theme", anim_theme())):
            try:
                setattr(job, attr, value)
            except Exception:
                pass

        if self.out_dir is None:
            self.out_dir = Path(tempfile.mkdtemp(prefix="anakit-anim-"))
        target = self.out_dir / f"{self.scene}-{self.quality}"
        shutil.rmtree(target, ignore_errors=True)

        self.player.stop()
        self.player.hide()
        self.btn_big.hide()
        self.btn.setEnabled(False)
        self.hint.setText("")
        self.bar.setRange(0, 100)
        self.bar.setValue(2)
        self.bar.show()
        self.status.setText("Manim startet …")

        self.thread = QThread(self)
        self.worker = AnimationWorker(job, target)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._fail)
        self.thread.start()

    def _progress(self, percent: int, text: str):
        self.bar.setValue(percent)
        self.status.setText(text)

    def _finish_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None
        self.bar.hide()
        self.btn.setEnabled(True)

    def _done(self, path: str):
        self._finish_thread()
        self.stamp = self._stamp_now()
        self.status.setText("")
        size = Path(path).stat().st_size / 1e6
        res = movie_size(path)
        self.player.load(path, f"{res.width()}×{res.height()}, "
                               f"{size:.1f} MB — zum Vergrössern anklicken")
        self.player.show()
        self.btn_big.show()

    def _fail(self, message: str):
        self._finish_thread()
        self.status.setText(message)
        self.player.stop()
        self.player.hide()
        self.btn_big.hide()

    def show_big(self):
        if self.player.loaded and self.player.path:
            AnimationWindow(self.window(), self.player.path,
                            "Animation").exec()

    # -- Thema und Aufräumen ----------------------------------------------

    def restyle(self):
        if self.ready:
            self.player.restyle()
            self.refresh()

    def shutdown(self):
        """Beendet einen laufenden Lauf, damit das Fenster schliessen kann."""
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
            self.thread = None
        if self.ready:
            self.player.stop()
        if self.out_dir is not None:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.out_dir = None



# ==========================================================================
# Die Ergebniskarte: Rechenweg, Zeichnung, Animation
# ==========================================================================

def show_view(page, index: int):
    """Schaltet die Ergebniskarte um, auch wenn der Nutzer nichts geklickt hat."""
    page.seg_out.set_value(index)
    page.stack_out.setCurrentIndex(index)


def anim_job_for(page: str, p: dict, result, scene: str, quality: str):
    """
    Baut den Auftrag für die Animation — oder einen Text, der sagt, warum es
    hier nichts zu zeigen gibt. Das ist die eine Stelle, an der `anim.py`
    eingehängt wird.
    """
    if result is None:
        return "Erst rechnen — die Animation zeigt das Ergebnis der Rechnung."
    if ak_anim is None:
        return ("Die Animation (anim.py) ist in diesem Ordner nicht vorhanden. "
                "Sie ist freiwillig; alles andere funktioniert auch ohne.")
    if page == "numerik" and (p.get("action", "newton") != "newton" or scene != "newton"):
        return "Animiert wird nur das Newton-Verfahren."
    if page == "hoeher":
        return ("Zu dieser Seite gibt es noch keine Animation — Rechenweg und Zeichnung "
                "zeigen das Ergebnis.")
    if not scene:
        return "Zu dieser Seite gibt es keine Animation."
    if page == "steigung" and p.get("method") == "line":
        return "Für eine Gerade durch zwei Punkte gibt es keine Animation."
    if page == "derivative" and p.get("action") in ("rule", "check"):
        return "Für diese Aufgabe gibt es keine Animation — wähle »Ableiten mit Regeln«."
    text = (p.get("f") or "").strip()
    if not text:
        return "Ohne Funktion gibt es nichts zu animieren."
    kw = {}
    if page == "limits" and p.get("action") not in ("limit", "table", "cont"):
        return "Für diese Aufgabe gibt es keine Animation — wähle einen Grenzwert."
    if scene == "newton":
        kw["n"] = min(max(len(getattr(result, "rows", ())), 1), 6)
    if scene in ("secant_to_tangent", "limit_approach", "newton"):
        try:
            x0 = read_number(p.get("x0") or p.get("x1") or "", "die Stelle x₀")
            kw["x0"] = float(x0)
        except Exception as exc:
            return f"Die Animation braucht eine Stelle x₀: {exc}"
    if scene == "curve_analysis":
        try:
            lo, hi = p.get("a", "").strip(), p.get("b", "").strip()
            if lo and hi:
                kw["xlim"] = (float(read_number(lo)), float(read_number(hi)))
        except Exception:
            pass
    try:
        return ak_anim.build_job(scene, text.replace("^", "**"), quality=quality,
                                 theme=anim_theme(), **kw)
    except Exception as exc:
        return friendly_error(exc)


# ==========================================================================
# Eine Seite: Eingabe links, Ergebnis rechts
# ==========================================================================

HISTORY_LIMIT = 10


class AnalysisPage(QWidget):
    """
    Gemeinsames Gerüst der sechs Seiten.

    Links Eingabe (Funktion, Rechnung) und die Prüfung mit Wolfram|Alpha,
    rechts die Ergebniskarte. Die Unterklassen sagen nur, welche Felder es
    gibt (`build_form`), wie ein Eintrag im Verlauf heisst (`describe`) und
    wie sich die Ansicht an die Eingabe anpasst (`adapt`).
    """

    KEY = ""
    TITLE = ""
    LEDE = ""
    ACTION_KEY = "action"
    F_CAPTION = "f(x)"
    F_PLACEHOLDER = "z. B. x^2 - 3x + 1"
    HELP = ""
    SEG_PAD = 30

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.window_ = window
        self.p: dict = dict(DEFAULTS[self.KEY])
        self.decimal = False
        self.result = None
        self.result_p: dict = {}
        self.vm: ResultVM | None = None
        self.stale = False
        self.history: list[tuple[str, dict]] = []
        self.job_id = 0
        self.running = False
        self.rerun = False
        self._jobs: dict = {}
        self.variant = 0
        self.inputs: dict[str, QLineEdit] = {}
        self.extra: dict[str, QWidget] = {}
        self.figure_job_id = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 26, 32, 26)
        outer.setSpacing(0)
        outer.addWidget(label(self.TITLE, 26, weight=QFont.Bold, display=True))
        outer.addWidget(label(self.LEDE, 13, "sec", display=True, wrap=True))
        outer.addSpacing(18)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        outer.addLayout(columns, 1)

        # Links: eigene Rolle, damit lange Prüfberichte scrollen, ohne die Ergebniskarte zu dehnen.
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setFixedWidth(600)
        left_host = QWidget()
        left_host.setObjectName("page")
        self.left_scroll.setWidget(left_host)
        left = QVBoxLayout(left_host)
        left.setContentsMargins(0, 0, 14, 8)
        left.setSpacing(18)
        columns.addWidget(self.left_scroll)
        right = QVBoxLayout()
        right.setSpacing(18)
        columns.addLayout(right, 1)

        self._build_input_card(left)
        self._build_action_card(left)

        self.wolfram = WolframPanel(self._wolfram_run, wolfram_subtitle(self.KEY))
        left.addWidget(self.wolfram)
        left.addStretch(1)

        self._build_output_card(right)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(280)
        self.preview_timer.timeout.connect(self.update_preview)

        self.sync_widgets()
        self.update_preview()

    # -- Aufbau -----------------------------------------------------------

    def _build_input_card(self, left: QVBoxLayout):
        self.card_input = Card(pad=24)
        left.addWidget(self.card_input)
        lay = self.card_input.lay
        self.lbl_f_title = label("Funktion", 15, weight=QFont.DemiBold)
        lay.addWidget(self.lbl_f_title)
        self.f_box = QWidget()
        fl = QVBoxLayout(self.f_box)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)
        fl.addSpacing(12)
        self.lbl_caption = label(self.F_CAPTION, 11, "sec")
        fl.addWidget(self.lbl_caption)
        fl.addSpacing(4)
        self.f_edit = entry(self.p.get("f", ""), placeholder=self.F_PLACEHOLDER, size=14, height=40)
        self.f_edit.textChanged.connect(lambda t: self.set_param("f", t))
        self.f_edit.returnPressed.connect(self.compute)
        fl.addWidget(self.f_edit)
        fl.addSpacing(8)
        self.preview_box = QVBoxLayout()
        self.preview_box.setContentsMargins(0, 0, 0, 0)
        fl.addLayout(self.preview_box)
        if self.HELP:
            fl.addSpacing(4)
            self.lbl_help = label(self.HELP, 10, "ter", wrap=True)
            fl.addWidget(self.lbl_help)
        lay.addWidget(self.f_box)
        lay.addSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.dd_examples = Dropdown([n for n, _ in EXAMPLES[self.KEY]], placeholder="Beispiele")
        self.dd_examples.picked.connect(self.load_example)
        row.addWidget(self.dd_examples)
        self.dd_history = Dropdown([], placeholder="Verlauf")
        self.dd_history.picked.connect(self.load_history)
        self.dd_history.setEnabled(False)
        row.addWidget(self.dd_history)
        row.addStretch(1)
        lay.addLayout(row)
        self.lbl_err = label("", 11, "red", wrap=True)
        self.lbl_err.hide()
        lay.addWidget(self.lbl_err)

    def _build_action_card(self, left: QVBoxLayout):
        self.card_action = Card(pad=24)
        left.addWidget(self.card_action)
        lay = self.card_action.lay
        lay.addWidget(label("Rechnung", 15, weight=QFont.DemiBold))
        lay.addSpacing(12)
        names = [short for _, short, _ in self.ACTIONS]
        self.seg_action = Segmented(names, self.action_index(), pad=self.SEG_PAD)
        self.seg_action.changed.connect(self.set_action)
        lay.addWidget(self.seg_action)
        lay.addSpacing(8)
        self.lbl_action = label("", 11, "sec", wrap=True)
        lay.addWidget(self.lbl_action)
        lay.addSpacing(12)
        self.form = QVBoxLayout()
        self.form.setSpacing(8)
        lay.addLayout(self.form)
        lay.addSpacing(16)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_compute = pill("Rechnen", "primary", self.compute)
        row.addWidget(self.btn_compute)
        self.seg_fmt = Segmented(["Exakt", "Dezimal"], 0)
        self.seg_fmt.changed.connect(self.set_format)
        row.addWidget(self.seg_fmt)
        row.addStretch(1)
        lay.addLayout(row)

    def _build_output_card(self, right: QVBoxLayout):
        self.card_out = Card()
        right.addWidget(self.card_out, 1)
        head = QHBoxLayout()
        head.setSpacing(10)
        self.seg_out = Segmented(["Rechenweg", "Zeichnung", "Animation"], 0, seg_width=98)
        self.seg_out.changed.connect(lambda i: self.stack_out.setCurrentIndex(i))
        head.addWidget(self.seg_out)
        self.lbl_status = label("", 11, "ter")
        head.addWidget(self.lbl_status)
        head.addStretch(1)
        self.btn_export = pill("Exportieren  ⌄", "secondary", self.open_export_menu, small=True)
        head.addWidget(self.btn_export)
        self.card_out.lay.addLayout(head)
        self.card_out.lay.addSpacing(12)
        self.stack_out = QStackedWidget()
        # Mit Auswahl, nicht pauschal: eine Regel ohne Auswahl vererbt sich auf
        # alle Kinder und nähme den Knöpfen darin ihre Farbe.
        self.stack_out.setObjectName("outstack")
        self.stack_out.setStyleSheet("#outstack { background: transparent; }")
        self.view = ResultView()
        self.plot = PlotPane()
        self.plot.variant_changed.connect(self.set_variant)
        self.plot.btn_save.clicked.connect(self.save_figure)
        self.plot.browser_requested.connect(self.open_in_browser)
        self.anim = AnimationPane(self.build_job, self.anim_key,
                                  scenes=ANIM_SCENES.get(self.KEY),
                                  subtitle=self.ANIM_SUBTITLE)
        self.stack_out.addWidget(self.view)
        self.stack_out.addWidget(self.plot)
        self.stack_out.addWidget(self.anim)
        self.card_out.lay.addWidget(self.stack_out, 1)

    ANIM_SUBTITLE = "Zeigt die Rechnung als kurzen Film."

    # -- Hooks für die Unterklassen --------------------------------------

    ACTIONS: list = []

    def build_form(self):
        """Baut die Felder zur gewählten Aktion in self.form auf."""

    def adapt(self):
        """Blendet je nach Eingabe Teile der Seite ein oder aus."""

    def describe(self) -> str:
        return self.p.get("f", "")

    # -- Hilfen für Felder -----------------------------------------------

    def action_index(self) -> int:
        keys = [k for k, _, _ in self.ACTIONS]
        try:
            return keys.index(self.p.get(self.ACTION_KEY))
        except ValueError:
            return 0

    def clear_form(self):
        self.inputs = {k: v for k, v in self.inputs.items() if k == "f"}
        self.extra = {}
        while self.form.count():
            item = self.form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._drop_layout(item.layout())

    def _drop_layout(self, lay):
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._drop_layout(item.layout())

    def field_row(self, caption: str, keys: Sequence[tuple[str, str]], width: int = 96,
                  tail: QWidget | None = None) -> QHBoxLayout:
        """Eine Zeile »Beschriftung  [Feld] [Feld]« in der Rechnungskarte."""
        row = QHBoxLayout()
        row.setSpacing(8)
        cap = label(caption, 11, "sec")
        cap.setFixedWidth(104)
        row.addWidget(cap)
        for key, placeholder in keys:
            ed = entry(self.p.get(key, ""), width=width, placeholder=placeholder, size=13)
            ed.textChanged.connect(lambda t, k=key: self.set_param(k, t))
            ed.returnPressed.connect(self.compute)
            self.inputs[key] = ed
            row.addWidget(ed)
        if tail is not None:
            row.addWidget(tail)
        row.addStretch(1)
        self.form.addLayout(row)
        return row

    def seg_row(self, caption: str, options: Sequence[str], value: int, on_change) -> Segmented:
        row = QHBoxLayout()
        row.setSpacing(8)
        cap = label(caption, 11, "sec")
        cap.setFixedWidth(104)
        row.addWidget(cap)
        seg = Segmented(options, value)
        seg.changed.connect(on_change)
        row.addWidget(seg)
        row.addStretch(1)
        self.form.addLayout(row)
        return seg

    def note_row(self, text: str):
        lb = label(text, 10, "ter", wrap=True)
        self.form.addWidget(lb)
        return lb

    # -- Zustand ----------------------------------------------------------

    def set_param(self, key: str, value: str):
        if self.p.get(key) == value:
            return
        self.p[key] = value
        self.touch(key == "f")

    def touch(self, function_changed: bool = False):
        """Die Eingabe hat sich geändert: Ergebnis und Prüfbericht gelten nicht mehr."""
        if function_changed:
            self.preview_timer.start()
        if self.result is not None and not self.stale:
            self.stale = True
            self.update_status()
        self.wolfram.invalidate()
        self.dd_examples_reset()

    def dd_examples_reset(self):
        pass

    def update_status(self, text: str | None = None):
        if text is not None:
            self.lbl_status.setText(text)
        elif self.running:
            self.lbl_status.setText("Rechne …")
        elif self.stale:
            self.lbl_status.setText("Eingabe geändert — neu rechnen")
        else:
            self.lbl_status.setText("")
        key = "amber" if self.stale and not self.running else "ter"
        self.lbl_status.setProperty("colorKey", key)
        self.lbl_status.setStyleSheet(f"color: {T[key]}; background: transparent;")

    def sync_widgets(self):
        """Zeigt den Zustand von self.p in den Widgets (nach Beispiel oder Verlauf)."""
        self.f_edit.blockSignals(True)
        self.f_edit.setText(self.p.get("f", ""))
        self.f_edit.blockSignals(False)
        self.seg_action.set_value(self.action_index())
        self.lbl_action.setText(next((d for k, _, d in self.ACTIONS
                                      if k == self.p.get(self.ACTION_KEY)), ""))
        self.clear_form()
        self.build_form()
        self.adapt()

    def set_action(self, index: int):
        self.p[self.ACTION_KEY] = self.ACTIONS[index][0]
        self.sync_widgets()
        self.touch()

    def set_format(self, index: int):
        self.decimal = bool(index)
        if self.result is not None:
            self.compute()

    def set_variant(self, index: int):
        """Eine andere Zeichnung zur selben Rechnung."""
        self.variant = index
        if self.result is not None:
            self.draw_variant()

    # -- Beispiele und Verlauf --------------------------------------------

    def load_example(self, name: str):
        for n, params in EXAMPLES[self.KEY]:
            if n == name:
                self.p = dict(DEFAULTS[self.KEY])
                self.p.update(params)
                self.variant = 0
                self.sync_widgets()
                self.update_preview()
                self.compute()
                return

    def remember(self):
        text = self.describe()
        self.history = [(t, q) for t, q in self.history if q != self.result_p]
        self.history.insert(0, (text, dict(self.result_p)))
        del self.history[HISTORY_LIMIT:]
        self.dd_history.set_options([f"{i + 1}.  {t}" for i, (t, _) in enumerate(self.history)])
        self.dd_history.setEnabled(True)

    def load_history(self, name: str):
        index = int(name.split(".")[0]) - 1
        self.p = dict(self.history[index][1])
        self.sync_widgets()
        self.update_preview()
        self.compute()

    # -- Vorschau der Funktion --------------------------------------------

    def set_err(self, text: str):
        self.lbl_err.setText(text)
        self.lbl_err.setVisible(bool(text))

    def update_preview(self):
        while self.preview_box.count():
            item = self.preview_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        text = self.p.get("f", "")
        if not self.f_box.isVisibleTo(self) and not text:
            return
        if not text.strip():
            self.set_err("")
            return
        try:
            _, expr = read_function(text)
        except InputError as exc:
            msg = str(exc)
            self.set_err(msg if len(msg) < 200 else msg[:197] + "…")
            return
        self.set_err("")
        try:
            item = item_vm("f(x)", expr, False, 4)
            self.preview_box.addWidget(FormulaLabel(item, MATH_PX["item"]))
        except Exception:
            pass

    # -- Rechnen ----------------------------------------------------------

    def params(self) -> dict:
        return dict(self.p)

    def compute(self):
        """Startet die Rechnung im Hintergrund; das Fenster bleibt bedienbar."""
        if self.running:
            self.rerun = True
            return
        self.job_id += 1
        job = self.job_id
        p, decimal, dark, variant = self.params(), self.decimal, T["dark"], self.variant
        self.running = True
        self.btn_compute.setEnabled(False)
        self.btn_compute.setText("Rechne …")
        self.update_status()
        if self.result is None:
            self.view.placeholder("Rechne …")
        self.start_worker(job, lambda: run_page_job(self.KEY, p, decimal, dark, variant),
                          self._computed, (p, decimal))

    def start_worker(self, job: int, run, handler, extra=()):
        """
        Startet `run` in einem eigenen Faden. Das Ergebnis kommt über ein Signal
        an eine Methode dieser Seite zurück, damit `handler` im Fenster-Faden
        läuft — eine blosse Funktion als Empfänger liefe im Hintergrundfaden.
        """
        thread = QThread(self)
        worker = ComputeWorker(job, run)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._job_finished)
        self._jobs[job] = (thread, worker, handler, extra)
        thread.start()

    def _job_finished(self, job_id: int, outcome):
        entry = self._jobs.pop(job_id, None)
        if entry is None:
            return
        thread, worker, handler, extra = entry
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()
        handler(job_id, outcome, *extra)

    def _computed(self, job: int, outcome: Outcome, p: dict, decimal: bool):
        self.running = False
        self.btn_compute.setEnabled(True)
        self.btn_compute.setText("Rechnen")
        if outcome.error:
            self.result, self.vm = None, None
            self.stale = False
            self.view.show_error(outcome.error)
            self.plot.show_message("Keine Zeichnung — die Rechnung ist nicht gelungen.")
            self.update_status("")
            show_view(self, 0)
        else:
            self.result, self.vm = outcome.result, outcome.vm
            self.result_p = dict(p)
            self.stale = False
            self.view.show_vm(outcome.vm)
            self._set_plot(outcome)
            self.anim.refresh()
            self.remember()
            self.update_status("")
            self.lbl_status.setText(f"{outcome.seconds:.1f} s" if outcome.seconds >= 0.3 else "")
        self.wolfram.invalidate()
        if self.rerun:
            self.rerun = False
            self.compute()

    def wait(self, ms: int = 60000):
        """Wartet, bis keine Rechnung mehr läuft (für Tests)."""
        import time
        end = time.time() + ms / 1000
        while (self.running or self._jobs) and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()

    def _set_plot(self, outcome: Outcome):
        names = figure_names(self.KEY, self.result_p)
        self.plot.set_variants(names, min(self.variant, len(names) - 1))
        if outcome.figure is not None:
            self.plot.show_figure(outcome.figure, self.figure_hint())
        else:
            self.plot.show_message(outcome.figure_error or "Zu dieser Rechnung gibt es keine Zeichnung.")

    def figure_hint(self) -> str:
        return ""

    def draw_variant(self):
        """Zeichnung neu bauen (Themawechsel, andere Ansicht) — im Hintergrund."""
        if self.result is None:
            return
        self.figure_job_id += 1
        job = self.figure_job_id
        page, p, result = self.KEY, dict(self.result_p), self.result
        dark, variant = T["dark"], self.variant

        def handler(job_id, outcome):
            if job_id != self.figure_job_id or self.result is not result:
                return
            self._set_plot(outcome)

        self.start_worker(job, lambda: run_figure_job(page, p, result, dark, variant), handler)

    # -- Prüfung mit Wolfram|Alpha ----------------------------------------

    def _wolfram_run(self, appid: str):
        """Gibt die Prüfung als Aufgabe zurück; ausgeführt wird sie im Faden."""
        if self.result is None:
            raise ValueError("Erst rechnen — die Prüfung vergleicht das Ergebnis der Rechnung.")
        if self.stale:
            raise ValueError("Die Eingabe hat sich seit der Rechnung geändert. Erst neu rechnen, "
                             "dann prüfen.")
        page, p, result = self.KEY, dict(self.result_p), self.result
        return lambda: run_wolfram_checks(appid, page, p, result)

    # -- Animation --------------------------------------------------------

    def anim_key(self):
        return (self.KEY, tuple(sorted(self.result_p.items())), self.decimal)

    def build_job(self, scene: str, quality: str):
        return anim_job_for(self.KEY, self.result_p, self.result, scene, quality)

    # -- Exportieren ------------------------------------------------------

    def open_export_menu(self):
        menu = QMenu(self)
        menu.setFont(FONTS.f(11))
        menu.addAction("Rechenweg als LaTeX kopieren", self.copy_latex)
        menu.addAction("Rechenweg als Text kopieren", self.copy_text)
        menu.addSeparator()
        menu.addAction("Zeichnung speichern …", self.save_figure)
        menu.addAction("Zeichnung kopieren", self.copy_figure)
        menu.exec(self.btn_export.mapToGlobal(QPoint(0, self.btn_export.height() + 2)))

    def toast(self, text: str, kind: str = "info"):
        self.window_.toasts.show(text, kind)

    def latex_text(self) -> str:
        return self.result.latex() if self.result is not None else ""

    def copy_latex(self):
        if self.result is None:
            self.toast("Erst rechnen.", "bad")
            return
        QGuiApplication.clipboard().setText(self.latex_text())
        self.toast("LaTeX kopiert", "ok")

    def copy_text(self):
        if self.result is None:
            self.toast("Erst rechnen.", "bad")
            return
        QGuiApplication.clipboard().setText(repr(self.result))
        self.toast("Rechenweg als Text kopiert", "ok")

    def save_figure(self):
        figure = self.plot.figure
        if figure is None:
            self.toast("Es gibt keine Zeichnung zu speichern.", "bad")
            return
        try:
            path = save_figure_dialog(self, figure, f"anakit-{self.KEY}")
        except Exception as exc:
            self.toast(f"Speichern ging nicht: {exc}", "bad")
            return
        if path:
            self.toast(f"Gespeichert: {Path(path).name}", "ok")

    def copy_figure(self):
        canvas = self.plot.canvas.canvas
        if canvas is None:
            self.toast("Es gibt keine Zeichnung zu kopieren.", "bad")
            return
        QGuiApplication.clipboard().setPixmap(canvas.grab())
        self.toast("Zeichnung kopiert", "ok")

    def open_in_browser(self):
        """Dieselbe Funktion mit Plotly, mit Zoom und Beschriftung beim Überfahren."""
        try:
            _, f = read_function(self.result_p.get("f", self.p.get("f", "")))
            figure = ak.plotly_function(f)
            path = Path(tempfile.mkdtemp(prefix="anakit-plot-")) / "plot.html"
            figure.write_html(str(path), include_plotlyjs="cdn")
            webbrowser.open(path.as_uri())
        except Exception as exc:
            self.toast(f"Browser-Zeichnung ging nicht: {friendly_error(exc)}", "bad")

    # -- Thema ------------------------------------------------------------

    def restyle(self):
        for card in (self.card_input, self.card_action, self.card_out):
            card.restyle()
        self.view.restyle()
        self.plot.restyle()
        self.anim.restyle()
        self.wolfram.restyle()
        self.update_status()
        self.update_preview()
        if self.result is not None:
            self.draw_variant()

    def shutdown(self):
        self.anim.shutdown()
        self.wolfram.shutdown()
        for thread, _worker, _h, _e in list(self._jobs.values()):
            thread.quit()
            thread.wait(3000)
        self._jobs.clear()


# ==========================================================================
# Seite 1: Steigung und Tangente
# ==========================================================================

class SteigungPage(AnalysisPage):
    KEY = "steigung"
    TITLE = "Steigung & Tangente"
    LEDE = ("Von der Sekante zur Tangente: Differenzenquotient, Tangentengleichung "
            "und die Ableitung aus der Definition.")
    ACTION_KEY = "method"
    ACTIONS = STEIGUNG_METHODS
    SEG_PAD = 26

    def build_form(self):
        m = self.p.get("method", "sek12")
        if m == "sek12":
            self.field_row("Stellen", [("x1", "x₁"), ("x2", "x₂")])
            self.note_row("Die Sekante geht durch (x₁ | f(x₁)) und (x₂ | f(x₂)).")
        elif m == "sekh":
            self.field_row("Stelle und h", [("x0", "x₀"), ("h", "h")])
            self.note_row("Zweite Stelle ist x₀ + h; h darf auch negativ sein, aber nicht 0.")
        elif m == "tan":
            self.field_row("Stelle", [("x0", "x₀")])
            idx = 1 if self.p.get("variant") == "h" else 0
            self.seg_row("Schreibweise", ["x → x₀", "h → 0"], idx,
                         lambda i: self.set_param("variant", "h" if i else "x"))
            self.note_row("Beide Schreibweisen des Differentialquotienten liefern dasselbe; "
                          "in der Prüfung wird oft eine bestimmte verlangt.")
        elif m == "def":
            self.note_row("Keine weiteren Eingaben: es wird f′(x) = lim (f(x+h) − f(x))/h "
                          "für beliebiges x hergeleitet.")
        else:
            self.field_row("Punkt P₁", [("p1x", "x₁"), ("p1y", "y₁")])
            self.field_row("Punkt P₂", [("p2x", "x₂"), ("p2y", "y₂")])
            self.note_row("Steigung m = Δy/Δx und Geradengleichung y = m·x + t.")

    def adapt(self):
        line = self.p.get("method") == "line"
        self.f_box.setVisible(not line)
        self.lbl_f_title.setText("Gerade durch zwei Punkte" if line else "Funktion")
        if line:
            self.set_err("")

    def describe(self) -> str:
        p = self.result_p or self.p
        m = p.get("method")
        if m == "line":
            return f"Gerade ({p.get('p1x')}|{p.get('p1y')}), ({p.get('p2x')}|{p.get('p2y')})"
        tail = {"sek12": f"Sekante {p.get('x1')}, {p.get('x2')}",
                "sekh": f"Sekante x₀={p.get('x0')}, h={p.get('h')}",
                "tan": f"Tangente x₀={p.get('x0')}",
                "def": "Ableitung aus Definition"}.get(m, "")
        return f"{p.get('f')} · {tail}"

    def set_action(self, index: int):
        self.variant = 0
        super().set_action(index)

    def figure_hint(self) -> str:
        if self.result_p.get("method") == "tan" and self.variant == 1:
            return "Die Sekanten drehen sich in die Tangente, wenn h gegen 0 geht."
        return ""


# ==========================================================================
# Seite 2: Grenzwerte und Stetigkeit
# ==========================================================================

class LimitsPage(AnalysisPage):
    KEY = "limits"
    TITLE = "Grenzwerte"
    LEDE = ("Grenzwert, Stetigkeit und Wertetabelle — auch an Polstellen, im Unendlichen "
            "und bei abschnittsweise definierten Funktionen.")
    ACTIONS = LIMIT_ACTIONS
    HELP = PIECEWISE_HELP
    SEG_PAD = 26

    def build_form(self):
        a = self.p.get("action", "limit")
        if a != "contall":
            inf = QWidget()
            hl = QHBoxLayout(inf)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)
            for sym, txt in (("∞", "oo"), ("−∞", "-oo"), ("π", "pi")):
                hl.addWidget(pill(sym, "secondary", lambda _=False, t=txt: self.insert_x0(t), small=True))
            self.field_row("Stelle x₀", [("x0", "z. B. 1, pi/2, oo")], width=150, tail=inf)
        if a in ("limit", "table"):
            idx = [k for k, _ in LIMIT_SIDES].index(self.p.get("side", "both"))
            self.seg_row("Seite", [n for _, n in LIMIT_SIDES], idx,
                         lambda i: self.set_param("side", LIMIT_SIDES[i][0]))
        if a == "limit":
            self.note_row("Links und rechts getrennt zu rechnen zeigt, ob der Grenzwert existiert.")
        elif a == "cont":
            self.note_row("Geprüft wird: linksseitiger Grenzwert = rechtsseitiger Grenzwert = f(x₀).")
        elif a == "contall":
            self.note_row("Verdachtsstellen sind Nullstellen des Nenners, Übergänge bei "
                          "abschnittsweisen Funktionen, Betrag, tan und cot.")
        else:
            self.note_row("Die Tabelle nähert sich x₀ in Zehnerschritten, von beiden Seiten.")

    def insert_x0(self, text: str):
        ed = self.inputs.get("x0")
        if ed is not None:
            ed.setText(text)
            ed.setFocus()

    def adapt(self):
        pass

    def describe(self) -> str:
        p = self.result_p or self.p
        a = p.get("action")
        name = {"limit": "lim", "cont": "stetig?", "contall": "stetig (ganz)", "table": "Tabelle"}[a]
        where = "" if a == "contall" else f" x→{p.get('x0')}" + (
            {"+": "⁺", "-": "⁻"}.get(p.get("side"), "") if a in ("limit", "table") else "")
        return f"{p.get('f')} · {name}{where}"


# ==========================================================================
# Seite 3: Ableiten
# ==========================================================================

class DerivativePage(AnalysisPage):
    KEY = "derivative"
    TITLE = "Ableiten"
    LEDE = ("Ableitungsregeln mit jedem Zwischenschritt — Summen-, Produkt-, Quotienten- und "
            "Kettenregel beim Namen genannt.")
    ACTIONS = DERIVATIVE_ACTIONS
    SEG_PAD = 26
    #: Beschriftung des zweiten Felds je Regel; leer = es gibt keins.
    RULE_SECOND = {"produkt": "g(x)", "quotient": "g(x)", "summe": "g(x)", "differenz": "g(x)",
                   "kette": "innere u(x)", "umkehr": "Umkehrfkt.", "faktor": "Faktor c",
                   "summand": "Konstante c", "potenz": ""}

    def build_form(self):
        a = self.p.get("action", "diff")
        if a == "diff":
            row = QHBoxLayout()
            row.setSpacing(8)
            cap = label("Ordnung n", 11, "sec")
            cap.setFixedWidth(104)
            row.addWidget(cap)
            stepper = Stepper(int(float(self.p.get("n", "1") or 1)))
            stepper.stepped.connect(self.step_order)
            self.extra["stepper"] = stepper
            row.addWidget(stepper)
            row.addWidget(label("1 = f′, 2 = f″ …", 10, "ter"))
            row.addStretch(1)
            self.form.addLayout(row)
        elif a == "at":
            self.field_row("Stelle x₀", [("x0", "x₀")])
            self.note_row("Erst f′(x) bilden, dann x₀ einsetzen — so steht der Rechenweg in der Prüfung.")
        elif a == "rule":
            row = QHBoxLayout()
            row.setSpacing(8)
            cap = label("Regel", 11, "sec")
            cap.setFixedWidth(104)
            row.addWidget(cap)
            current = next((n for k, n, _, _ in RULES if k == self.p.get("rule")), RULES[0][1])
            dd = Dropdown([n for _, n, _, _ in RULES], value=current)
            dd.setFixedWidth(200)
            dd.picked.connect(self.pick_rule)
            self.extra["rule"] = dd
            row.addWidget(dd)
            row.addStretch(1)
            self.form.addLayout(row)
            caption = self.RULE_SECOND.get(self.p.get("rule"), "g(x)")
            if caption:
                self.field_row(caption, [("g", "")], width=210)
            self.note_row("Beide Felder leer lassen zeigt das Beispiel aus der Vorlesung.")
        elif a == "table":
            self.note_row("Keine Eingabe nötig: die Tabelle zeigt jede Regel in der Schreibweise "
                          "von Newton und von Leibniz.")
        elif a == "check":
            row = QHBoxLayout()
            row.setSpacing(8)
            cap = label("Deine Ableitung", 11, "sec")
            cap.setFixedWidth(104)
            row.addWidget(cap)
            self.form.addLayout(row)
            ed = entry(self.p.get("claimed", ""), placeholder="z. B. e^x + x*e^x", size=13)
            ed.textChanged.connect(lambda t: self.set_param("claimed", t))
            ed.returnPressed.connect(self.compute)
            self.inputs["claimed"] = ed
            self.form.addWidget(ed)
            self.note_row("Die App vergleicht sie mit der richtigen Ableitung — symbolisch und an "
                          "mehreren Stellen — und sagt, wo sie abweicht.")

    def step_order(self, delta: int):
        n = max(1, min(6, int(float(self.p.get("n", "1") or 1)) + delta))
        self.extra["stepper"].set_value(n)
        self.set_param("n", str(n))

    def pick_rule(self, name: str):
        key = next(k for k, n, _, _ in RULES if n == name)
        self.p["rule"] = key
        example = next(e for k, _, _, e in RULES if k == key)
        parts = [s.strip() for s in example.split(",")]
        self.p["f"] = parts[0]
        self.p["g"] = parts[1] if len(parts) > 1 else ""
        self.sync_widgets()
        self.touch(True)

    def adapt(self):
        a = self.p.get("action", "diff")
        self.f_box.setVisible(a != "table")
        self.lbl_f_title.setText("Funktion" if a != "table" else "Regeltabelle")
        if a == "rule":
            r = self.p.get("rule")
            self.lbl_caption.setText("äussere Funktion f(u)" if r == "kette" else
                                     "Funktion f(x)")
        else:
            self.lbl_caption.setText("Funktion f(x)")
        if a == "table":
            self.set_err("")

    def describe(self) -> str:
        p = self.result_p or self.p
        a = p.get("action")
        if a == "table":
            return "Regeltabelle"
        if a == "rule":
            name = next((n for k, n, _, _ in RULES if k == p.get("rule")), "Regel")
            return f"{name}: {p.get('f') or 'Beispiel'}"
        tail = {"diff": "f′" if p.get("n") in ("1", "") else f"f^({p.get('n')})",
                "at": f"f′({p.get('x0')})", "check": "prüfen"}.get(a, "")
        return f"{p.get('f')} · {tail}"


# ==========================================================================
# Seite 4: Kurvendiskussion
# ==========================================================================

class CurvePage(AnalysisPage):
    KEY = "curve"
    TITLE = "Kurvendiskussion"
    LEDE = ("Monotonie, kritische Punkte, Krümmung, Extrema und Wendepunkte — mit "
            "Vorzeichentabellen und der Zeichnung von f, f′ und f″.")
    ACTIONS = CURVE_ACTIONS
    SEG_PAD = 14

    def build_form(self):
        self.field_row("Bereich", [("a", "a"), ("b", "b")], width=100)
        self.note_row("Optional: ein Intervall [a, b]. Leer gelassen wird die ganze Zahlengerade "
                      "untersucht und der Zeichenbereich automatisch gewählt.")

    def describe(self) -> str:
        p = self.result_p or self.p
        a = next((s for k, s, _ in CURVE_ACTIONS if k == p.get("action")), "")
        dom = f" [{p.get('a')}, {p.get('b')}]" if p.get("a") or p.get("b") else ""
        return f"{p.get('f')}{dom} · {a}"


# ==========================================================================
# Seite 5: Numerik (SW5)
# ==========================================================================

class NumerikPage(AnalysisPage):
    KEY = "numerik"
    TITLE = "Numerik"
    LEDE = ("Nullstellen mit Bisektion und Newton-Verfahren, Ableitungen numerisch genähert — "
            "mit Fehlerordnung und optimaler Schrittweite.")
    ACTIONS = NUMERIK_ACTIONS
    SEG_PAD = 22

    def _kind_row(self):
        keys = [k for k, _ in DIFF_KINDS]
        idx = keys.index(self.p.get("kind")) if self.p.get("kind") in keys else 2
        self.seg_row("Differenz", [n for _, n in DIFF_KINDS], idx,
                     lambda i: self.set_param("kind", keys[i]))

    def build_form(self):
        a = self.p.get("action", "newton")
        if a == "bisect":
            self.field_row("Intervall", [("a", "a"), ("b", "b")], width=100)
            self.field_row("Toleranz", [("tol", "z. B. 1e-6")], width=120)
            self.note_row("f muss auf [a, b] stetig sein und an den Enden verschiedene Vorzeichen "
                          "haben. Abbruch, wenn die Intervalllänge ≤ Toleranz.")
        elif a == "newton":
            self.field_row("Startwert", [("x0", "x₀")], width=100)
            self.field_row("Toleranz", [("tol", "z. B. 1e-10")], width=120)
            self.seg_row("Ableitung", ["exakt", "numerisch"], 1 if self.p.get("numerisch") == "1" else 0,
                         lambda i: self.set_param("numerisch", "1" if i else "0"))
            self.note_row("xₖ₊₁ = xₖ − f(xₖ)/f′(xₖ). Abbruch, wenn |xₖ₊₁ − xₖ| ≤ Toleranz.")
        elif a == "numdiff":
            self.field_row("Stelle", [("x0", "x₀")], width=100)
            self.field_row("Schrittweite h", [("h", "leer = optimal")], width=140)
            self._kind_row()
            self.note_row("Vorwärts/rückwärts: Fehlerordnung h, optimal √ε ≈ 1.5e−8. "
                          "Zentral: Fehlerordnung h², optimal ∛ε ≈ 6e−6.")
        else:
            self.field_row("Stelle", [("x0", "x₀")], width=100)
            self._kind_row()
            self.note_row("Fehler gegen die exakte Ableitung, solange h kleiner wird: zuerst sinkt der "
                          "Verfahrensfehler, dann steigen die Rundungsfehler.")

    def describe(self) -> str:
        p = self.result_p or self.p
        a = p.get("action")
        tail = {"bisect": f"Bisektion [{p.get('a')}, {p.get('b')}]",
                "newton": f"Newton ab {p.get('x0')}",
                "numdiff": f"{dict(DIFF_KINDS).get(p.get('kind'), '')}e Differenz bei {p.get('x0')}",
                "errtab": f"Fehlertabelle bei {p.get('x0')}"}.get(a, "")
        return f"{p.get('f')} · {tail}"


# ==========================================================================
# Seite 6: Höhere Ableitungen (SW6)
# ==========================================================================

class HoeherePage(AnalysisPage):
    KEY = "hoeher"
    TITLE = "Höhere Ableitungen"
    LEDE = ("Taylor- und MacLaurin-Polynome, die Sonderfälle der Extrema und Wendepunkte und der "
            "Funktionsüberblick einer Kurvendiskussion.")
    ACTIONS = HOEHER_ACTIONS
    SEG_PAD = 22

    def build_form(self):
        a = self.p.get("action", "mac")
        if a == "taylor":
            self.field_row("Stelle und Grad", [("x0", "x₀"), ("n", "n")], width=100)
            self.note_row("T_n(x) = Σ f⁽ᵏ⁾(x₀)/k! · (x − x₀)ᵏ, k = 0 … n. f muss bei x₀ n-mal "
                          "ableitbar sein.")
        elif a == "mac":
            self.field_row("Grad", [("n", "n")], width=100)
            self.note_row("Der Spezialfall x₀ = 0: T_n(x) = Σ f⁽ᵏ⁾(0)/k! · xᵏ.")
        elif a == "higher":
            self.field_row("Stelle", [("x0", "x₀")], width=100)
            self.note_row("Ist f′(x₀) = 0 und f⁽ᵏ⁾(x₀) die erste Ableitung ≠ 0: k gerade ⇒ "
                          "Minimum bzw. Maximum, k ungerade ⇒ Wendepunkt (Sattelpunkt).")
        else:
            self.note_row("Keine weitere Eingabe: Definitions- und Wertebereich, Symmetrie, Nullstellen, "
                          "Pole und Asymptoten. Ableitungen dazu: Seite »Kurvendiskussion«.")

    def describe(self) -> str:
        p = self.result_p or self.p
        a = p.get("action")
        tail = {"taylor": f"Taylor um {p.get('x0')}, n = {p.get('n')}",
                "mac": f"MacLaurin, n = {p.get('n')}",
                "higher": f"Sonderfall bei {p.get('x0')}",
                "basics": "Überblick"}.get(a, "")
        return f"{p.get('f')} · {tail}"


PAGE_CLASSES = [SteigungPage, LimitsPage, DerivativePage, CurvePage, NumerikPage, HoeherePage]


# ==========================================================================
# Hilfe
# ==========================================================================

class HelpDialog(QDialog):
    """Kurze Erklärung der Schreibweise und der sechs Seiten."""

    SECTIONS = [
        ("Funktionen eingeben", "x^2 − 3x",
         "Potenz mit ^, Produkt mit * oder einfach hintereinander (3x). Bekannt sind "
         "sin, cos, tan, ln, e^x, sqrt, abs, pi und Brüche wie 1/x."),
        ("Abschnittsweise", "x^2 if x<1 else 2-x",
         "»if« nennt die Bedingung, »else« den Rest. Mehrere Äste: "
         "x if x<0 else 1 if x<2 else 3-x. Betrag und Vorzeichen: abs(x), sgn(x)."),
        ("Stellen und Unendlich", "pi/2, oo, -oo",
         "Stellen dürfen Brüche, Wurzeln und pi enthalten. Für ∞ genügt oo; "
         "die Knöpfe ∞, −∞ und π neben dem Feld tragen es ein."),
        ("Exakt oder Dezimal", "1/3 ↔ 0.3333",
         "Gerechnet wird immer exakt. »Dezimal« rundet nur die Anzeige auf vier Stellen."),
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Hilfe")
        self.setModal(True)
        self.setFixedWidth(560)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(12)
        lay.addWidget(label("So gibst du etwas ein", 15, weight=QFont.DemiBold))
        for title, example, text in self.SECTIONS:
            box = QFrame()
            box.setObjectName("helpbox")
            box.setStyleSheet(f"#helpbox {{ background: {T['fill2']}; border-radius: 0px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(16, 13, 16, 13)
            bl.setSpacing(4)
            row = QHBoxLayout()
            row.addWidget(label(title, 12, weight=QFont.DemiBold))
            row.addStretch(1)
            row.addWidget(label(example, 11, "blue", mono=True))
            bl.addLayout(row)
            bl.addWidget(label(text, 11, "sec", wrap=True))
            lay.addWidget(box)
        lay.addSpacing(4)
        lay.addWidget(label("Prüfen mit Wolfram|Alpha", 12, weight=QFont.DemiBold))
        lay.addWidget(label("Jede Seite kann ihr Ergebnis gegen Wolfram|Alpha prüfen. Das ist "
                            "freiwillig und braucht eine kostenlose AppID. Ohne Netz bleibt "
                            "die Rechnung selbst unberührt, nur die Gegenprobe meldet, was fehlt.",
                            11, "sec", wrap=True))
        lay.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(pill("Verstanden", "primary", self.accept, small=True))
        lay.addLayout(row)


# ==========================================================================
# Hauptfenster
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("anakit — Analysis mit Rechenweg")
        self.resize(1320, 900)
        self.setMinimumSize(1040, 600)
        self._build()
        self.toasts = ToastArea(self.centralWidget())
        self.apply_theme()

    # -- Aufbau -----------------------------------------------------------

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav = QWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedHeight(56)
        nav_lay = QHBoxLayout(self.nav)
        nav_lay.setContentsMargins(32, 0, 32, 0)
        self.brand = label("anakit", 13, weight=QFont.DemiBold)
        nav_lay.addWidget(self.brand)
        nav_lay.addSpacing(18)
        self.seg_page = Segmented([cls.TITLE for cls in PAGE_CLASSES], 0, pad=28)
        self.seg_page.changed.connect(self.set_page)
        nav_lay.addWidget(self.seg_page)
        nav_lay.addStretch(1)
        self.seg_theme = Segmented(["Hell", "Dunkel"], 0)
        self.seg_theme.changed.connect(self.set_theme)
        nav_lay.addWidget(self.seg_theme)
        nav_lay.addSpacing(6)
        nav_lay.addWidget(pill("Hilfe", "plain", self.show_help, small=True))
        outer.addWidget(self.nav)

        self.hairline = QFrame()
        self.hairline.setFixedHeight(1)
        outer.addWidget(self.hairline)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.pages = [cls(self) for cls in PAGE_CLASSES]
        for page in self.pages:
            self.stack.addWidget(page)

    # -- Aussehen ---------------------------------------------------------

    def apply_theme(self):
        sheet = f"""
        QWidget {{ color: {T['text']}; }}
        QMainWindow, #page, QStackedWidget {{ background: {T['bg']}; }}
        QScrollArea {{ background: transparent; }}
        #nav {{ background: {T['nav']}; }}
        #card {{ background: {T['card']}; border-radius: 0px;
                 border: {'0px' if T['dark'] else '1px'} solid {T['hair']}; }}
        QPushButton#primary {{ background: {T['blue']}; color: #ffffff;
            border: none; border-radius: 0px; padding: 0 22px; }}
        QPushButton#primary:hover {{ background: {T['blue_hi']}; }}
        QPushButton#primary:disabled {{ background: {T['fill']};
            color: {T['ter']}; }}
        QPushButton#secondary {{ background: {T['fill']}; color: {T['text']};
            border: none; border-radius: 0px; padding: 0 18px; }}
        QPushButton#secondary:hover {{ background: {T['line']}; }}
        QPushButton#secondary:disabled {{ color: {T['ter']}; }}
        QPushButton#plain {{ background: transparent; color: {T['blue']};
            border: none; padding: 0 10px; }}
        QPushButton#plain:hover {{ text-decoration: underline; }}
        QPushButton#plain:disabled {{ color: {T['ter']}; }}
        QPushButton#dropdown {{ background: {T['fill']}; color: {T['text']};
            border: none; border-radius: 0px; padding: 0 10px; text-align: left; }}
        QPushButton#dropdown:hover {{ background: {T['line']}; }}
        QPushButton#dropdown:disabled {{ color: {T['ter']}; }}
        QLineEdit#cell {{ background: {T['entry']}; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 0px; padding: 0 10px;
            selection-background-color: {T['blue']}; }}
        QLineEdit#cell:focus {{ border: 2px solid {T['blue']}; }}
        QMenu {{ background: {T['card']}; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 0px; padding: 5px; }}
        QMenu::item {{ padding: 6px 22px; border-radius: 0px; }}
        QMenu::item:selected {{ background: {T['blue']}; color: #ffffff; }}
        QMenu::separator {{ height: 1px; background: {T['line']}; margin: 4px 8px; }}
        QToolTip {{ background: {T['card']}; color: {T['text']};
            border: 1px solid {T['line']}; padding: 4px 8px; }}
        QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {T['line']};
            border-radius: 0px; min-height: 40px; }}
        QScrollBar::handle:vertical:hover {{ background: {T['ter']}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {T['line']};
            border-radius: 0px; min-width: 40px; }}
        QDialog {{ background: {T['card']}; }}
        QProgressBar#progress {{ background: {T['fill']}; border: none;
            border-radius: 0px; }}
        QProgressBar#progress::chunk {{ background: {T['blue']};
            border-radius: 0px; }}
        """
        self.setStyleSheet(sheet)
        self.hairline.setStyleSheet(f"background: {T['line']};")
        for page in getattr(self, "pages", []):
            page.restyle()
        restyle_labels(self)

    def set_theme(self, index: int):
        T.clear()
        T.update(DARK if index else LIGHT)
        self.apply_theme()
        for widget in self.findChildren(QWidget):
            widget.update()

    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        self.seg_page.set_value(index)

    @property
    def page(self) -> AnalysisPage:
        return self.pages[self.stack.currentIndex()]

    def show_help(self):
        HelpDialog(self).exec()

    # -- Tastatur ---------------------------------------------------------

    def keyPressEvent(self, event):
        mods = event.modifiers()
        if mods & (Qt.ControlModifier | Qt.MetaModifier):
            if Qt.Key_1 <= event.key() < Qt.Key_1 + len(PAGE_CLASSES):
                self.set_page(event.key() - Qt.Key_1)
                return
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.page.compute()
                return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toasts"):
            self.toasts.reposition()

    def closeEvent(self, event):
        """Hintergrundläufe beenden und temporäre Renderdateien aufräumen."""
        for page in self.pages:
            page.shutdown()
        super().closeEvent(event)


def _prewarm_matplotlib():
    try:
        import matplotlib.pyplot  # noqa: F401
        from matplotlib.backends.backend_qtagg import FigureCanvas  # noqa: F401
        render_mask(r"\frac{a}{b}", 14)         # Mathtext und Schriften einmal laden
    except Exception:
        pass            # fehlt Matplotlib, meldet »Zeichnung« das selbst


def create_app(argv: Sequence[str] | None = None):
    """Legt QApplication, Schriften und Fenster an; gibt (app, window) zurück."""
    global FONTS
    app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
    app.setApplicationName("anakit")
    FONTS = Fonts()
    setup_math_sizes()
    window = MainWindow()
    return app, window


def main():
    app, window = create_app()
    window.show()
    # Matplotlib im Hintergrund vorladen, damit die erste Rechnung nicht stockt.
    threading.Thread(target=_prewarm_matplotlib, daemon=True).start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
