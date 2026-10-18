#!/usr/bin/env python3
"""
Zwei Programmierübungen zur Ableitung — ein Prüfstand, keine Lösung.

Die Folien rechnen die Ableitung als Grenzwert des Differenzenquotienten.
Ein Computer kann keinen Grenzwert bilden, er kann nur mit einem kleinen
h rechnen. Das selbst zu programmieren zeigt, warum „klein“ nicht „so klein
wie möglich“ heisst, und wie man damit Extrema findet.

Darum steht hier keine Lösung, sondern ein Prüfstand: du schreibst deine
Funktion, rufst `pruefe_...` damit auf und siehst sofort, ob sie stimmt —
an Standardfunktionen und an ein paar Fällen, die gern übersehen werden.

    from anakit.uebung import (pruefe_ableitung, vorlage_ableitung,
                               pruefe_extrema, vorlage_extrema)

    print(vorlage_ableitung())        # Gerüst zum Abschreiben

    def meine_ableitung(f, x):
        ...
        return wert

    pruefe_ableitung(meine_ableitung)

    print(vorlage_extrema())

    def meine_extrema(f, a, b, n):
        ...
        return minima_x, maxima_x

    pruefe_extrema(meine_extrema)

Als Vergleich dient anakit selbst (exakt gerechnet mit `diff` bzw.
`extrema`). Nach dem Üben kannst du mit `vergleiche_mit_anakit(...)` den
exakten Rechenweg ansehen.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import sympy as sp

from ..rechnen.core import parse, x
from ..rechnen.derivative import diff
from ..rechnen.analysis import extrema
from ..rechnen.render import Result

__all__ = ["pruefe_ableitung", "vorlage_ableitung", "pruefe_extrema",
           "vorlage_extrema", "vergleiche_mit_anakit", "Pruefung",
           "VORLAGE_ABLEITUNG", "VORLAGE_EXTREMA"]


# ==========================================================================
# Gerüste zum Abschreiben
# ==========================================================================

VORLAGE_ABLEITUNG = '''
def meine_ableitung(f, x):
    """
    Näherung für f'(x) über den Differenzenquotienten.

    f ist eine Python-Funktion (f(2.0) liefert eine Zahl), x ein float.
    Gib die Näherung für die Steigung der Tangente bei x zurück.
    """
    # Wahl von h — hier lauern zwei Fehler gegeneinander:
    #   h zu gross:  der Differenzenquotient ist noch eine Sekante, nicht die
    #                Tangente (Abbruchfehler).
    #   h zu klein:  f(x+h) und f(x) unterscheiden sich nur noch in den
    #                letzten Stellen der Gleitkommazahl; die Differenz besteht
    #                aus Rundungsfehlern und wird durch das winzige h geteilt
    #                (Auslöschung). Doppelte Genauigkeit hat etwa 16 Stellen.
    h = None  # TODO: wähle h (probier 1e-1, 1e-5, 1e-8, 1e-12 und beobachte)

    # TODO: Differenzenquotient bilden. Einseitig (f(x+h) − f(x))/h geht,
    #       aber ein symmetrischer Quotient mit f(x+h) und f(x−h) ist bei
    #       gleichem h viel genauer. Wodurch teilst du dann?
    pass
'''.strip()


VORLAGE_EXTREMA = '''
def meine_extrema(f, a, b, n):
    """
    Findet lokale Minima und Maxima von f auf [a, b].

    Teile [a, b] in n gleiche Stücke, berechne an den Gitterpunkten die
    Steigung (z. B. mit deiner meine_ableitung) und suche Vorzeichenwechsel.
    Gib zwei Listen zurück: (minima_x, maxima_x).

      Steigung von minus nach plus  -> Minimum
      Steigung von plus nach minus  -> Maximum
    """
    dx = (b - a) / n
    xs = [a + i * dx for i in range(n + 1)]

    # TODO: Steigung an jedem Gitterpunkt (Liste d, gleich lang wie xs)
    d = []

    minima, maxima = [], []

    # Achtung, drei Stolpersteine:
    #  1. Die Steigung kann an einem Gitterpunkt genau 0 sein (oder durch
    #     Rundung fast 0). Ein Test "d[i] * d[i+1] < 0" verpasst dann das
    #     Extremum. Gleichzeitig soll ein Sattelpunkt (Steigung ">= 0, 0, >= 0")
    #     kein Extremum sein.
    #  2. In einer konstanten Funktion gibt es keinen Vorzeichenwechsel.
    #  3. Der Rand a und b ist kein lokales Extremum — dort fehlt die
    #     Nachbarseite.
    for i in range(n):
        # TODO: Vorzeichen von d[i] und d[i+1] vergleichen, die Stelle
        #       (z. B. die Mitte zwischen xs[i] und xs[i+1]) eintragen
        pass

    return minima, maxima
'''.strip()


def vorlage_ableitung() -> str:
    """Das Gerüst für `meine_ableitung` zum Abschreiben."""
    return VORLAGE_ABLEITUNG


def vorlage_extrema() -> str:
    """Das Gerüst für `meine_extrema` zum Abschreiben."""
    return VORLAGE_EXTREMA


# ==========================================================================
# Ergebnisobjekt
# ==========================================================================

@dataclass
class Fall:
    """Ein Prüffall: ok, Name, eine Zeile Befund und eventuell Einzelheiten."""
    name: str
    ok: bool
    text: str = ""
    details: list[str] = field(default_factory=list)


@dataclass
class Stolperstein:
    """Ein Fall ohne Bestehen oder Durchfallen: nur ein Befund."""
    name: str
    text: str
    status: str = "!"   # "!" Warnung, "i" Information, "✓" in Ordnung


class Pruefung:
    """
    Das Ergebnis eines Prüfstands. `bool(p)` ist True, wenn alle Fälle
    bestanden sind; `p.bestanden`, `p.gesamt`, `p.faelle`, `p.hinweise`
    liefern die Einzelheiten.
    """

    def __init__(self, titel: str):
        self.titel = titel
        self.faelle: list[Fall] = []
        self.stolpersteine: list[Stolperstein] = []
        self.hinweise: list[str] = []
        self.infos: list[str] = []
        self.ausfuehrlich = True
        self._gezeigt = False

    # -- Auswertung -------------------------------------------------------

    @property
    def bestanden(self) -> int:
        return sum(1 for f in self.faelle if f.ok)

    @property
    def gesamt(self) -> int:
        return len(self.faelle)

    @property
    def ok(self) -> bool:
        return self.gesamt > 0 and self.bestanden == self.gesamt

    def __bool__(self) -> bool:
        return self.ok

    # -- Anzeige ----------------------------------------------------------

    def kopf(self) -> str:
        return f"{self.titel}: {self.bestanden} von {self.gesamt} Fällen bestanden"

    def bericht(self) -> str:
        lines = [self.kopf()]
        for f in self.faelle:
            lines.append(f"  {'✓' if f.ok else '✗'} {f.name}" + (f" — {f.text}" if f.text else ""))
            if not f.ok or self.ausfuehrlich:
                lines += [f"      {d}" for d in f.details]
        if self.stolpersteine:
            lines += ["", "Stolpersteine — hier gibt es kein Bestehen, aber etwas zu lernen:"]
            for s in self.stolpersteine:
                lines.append(f"  {s.status} {s.name}: {s.text}")
        if self.infos:
            lines.append("")
            lines += [f"  · {i}" for i in self.infos]
        if self.hinweise:
            lines += ["", "Hinweise:"]
            lines += [f"  • {h}" for h in self.hinweise]
        if self.ok:
            lines += ["", "Alles bestanden."]
        return "\n".join(lines)

    __str__ = bericht

    def __repr__(self) -> str:
        if self._gezeigt:
            return f"<{self.kopf()}>"
        return self.bericht()

    def _repr_html_(self):
        if self._gezeigt:
            return None
        return self.html()

    def html(self) -> str:
        def esc(s):
            return html.escape(s)
        parts = [Result.CSS, "<div class='ak'>",
                 f"<p class='ak-h'>{esc(self.titel)}</p>",
                 f"<p class='ak-sum'>{self.bestanden} von {self.gesamt} Fällen bestanden</p>"]
        for f in self.faelle:
            mark = ("<span style='color:#1f9d55;font-weight:700'>✓</span>" if f.ok
                    else "<span style='color:#d93025;font-weight:700'>✗</span>")
            body = f"<div class='ak-t'>{mark} {esc(f.name)}</div>"
            if f.text:
                body += f"<div class='ak-d'>{esc(f.text)}</div>"
            if not f.ok or self.ausfuehrlich:
                for d in f.details:
                    body += f"<div class='ak-n'>{esc(d)}</div>"
            parts.append(f"<div class='ak-step'>{body}</div>")
        if self.stolpersteine:
            parts.append("<p class='ak-sum' style='margin-top:14px'>Stolpersteine — hier gibt es "
                         "kein Bestehen, aber etwas zu lernen:</p>")
            for s in self.stolpersteine:
                parts.append(f"<div class='ak-warn'><b>{esc(s.name)}</b>: {esc(s.text)}</div>")
        for i in self.infos:
            parts.append(f"<div class='ak-n'>· {esc(i)}</div>")
        for h in self.hinweise:
            parts.append(f"<div class='ak-warn'>{esc(h)}</div>")
        if self.ok:
            parts.append("<div class='ak-res'><b>Alles bestanden.</b></div>")
        parts.append("</div>")
        return "".join(parts)

    def show(self) -> "Pruefung":
        """Im Notebook gesetzt anzeigen, sonst ausdrucken."""
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import HTML, display
                display(HTML(self.html()))
                self._gezeigt = True
                return self
        except ImportError:
            pass
        print(self.bericht())
        self._gezeigt = True
        return self


def _abschluss(p: Pruefung, ausgabe: bool | None) -> Pruefung:
    if ausgabe is None or ausgabe:
        p.show()
    return p


# ==========================================================================
# Hilfen
# ==========================================================================

def _g(v: float) -> str:
    """Zahl knapp und lesbar."""
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.8g}"


def _zahl(v) -> float:
    """Wandelt Rückgabewerte (auch numpy-Skalare) in float; wirft ValueError."""
    if v is None:
        raise ValueError("None")
    if isinstance(v, complex):
        raise ValueError("komplex")
    return float(v)


@lru_cache(maxsize=None)
def _exakte_ableitung(expr: str) -> Callable[[float], float]:
    """Die exakte Ableitung als schnelle Funktion — berechnet von anakit.diff."""
    d = diff(expr).expr
    return sp.lambdify(x, d, modules=["numpy"])


@lru_cache(maxsize=None)
def _exakte_extrema(expr: str, a: str, b: str):
    """Referenz: (minima, maxima, sattel) auf (a, b), berechnet von anakit.extrema."""
    res = extrema(expr, domain=(parse(a), parse(b)))

    def werte(punkte):
        return sorted(float(sp.N(p[0])) for p in punkte)

    return werte(res.minima), werte(res.maxima), werte(res.saddles)


class _Spur:
    """
    Verpackt f und merkt sich, an welchen Stellen die Funktion des
    Studierenden sie aufgerufen hat. Daraus lässt sich ihr h ablesen.
    """

    def __init__(self, f: Callable[[float], float]):
        self._f = f
        self.punkte: list[float] = []
        self.__name__ = getattr(f, "__name__", "f")

    def __call__(self, t):
        try:
            self.punkte.append(float(t))
        except (TypeError, ValueError):
            pass
        return self._f(t)


# ==========================================================================
# Übung 1: Ableitung als Differenzenquotient
# ==========================================================================

#: (Name, Funktion, Term für anakit, Stellen)
ABLEITUNG_FAELLE: list[tuple[str, Callable[[float], float], str, list[float]]] = [
    ("x²", lambda t: t * t, "x^2", [-2.0, 0.5, 3.0]),
    ("x³", lambda t: t ** 3, "x^3", [-1.5, 1.0, 2.0]),
    ("sin x", math.sin, "sin(x)", [0.0, 1.0, 2.5]),
    ("eˣ", math.exp, "exp(x)", [-1.0, 0.0, 2.0]),
    ("1/x", lambda t: 1.0 / t, "1/x", [0.5, -2.0, 4.0]),
    ("|x| (ausserhalb von 0)", abs, "abs(x)", [-1.5, 2.0]),
    ("√x", math.sqrt, "sqrt(x)", [0.25, 1.0, 9.0]),
]

#: Toleranz: relativ zum exakten Wert, mindestens absolut 1e-6.
TOLERANZ_ABLEITUNG = 1e-6


def pruefe_ableitung(funktion: Callable, ausfuehrlich: bool = True,
                     ausgabe: bool | None = None) -> Pruefung:
    """
    Prüft deine numerische Ableitung an sieben Funktionen und mehreren Stellen.

    `funktion(f, x)` bekommt eine Python-Funktion f und eine Stelle x (float)
    und soll eine Näherung für f'(x) zurückgeben. Verglichen wird mit der
    exakten Ableitung von anakit, Toleranz 1e-6 (relativ).

    Der Prüfstand sieht auch, an welchen Stellen deine Funktion f aufruft,
    und kann daraus auf dein h schliessen — das steht dann in den Hinweisen.

    Gibt eine Pruefung zurück (in Bedingungen wie True/False benutzbar).
    `ausgabe=False` unterdrückt die Anzeige.
    """
    p = Pruefung("Ableitung als Differenzenquotient")
    p.ausfuehrlich = ausfuehrlich

    abstaende: list[float] = []
    einseitig_alle = True
    faktoren: list[float] = []
    vorzeichen_falsch = 0
    ausnahmen: list[str] = []
    nicht_definiert = False

    for name, f, expr, punkte in ABLEITUNG_FAELLE:
        exakt = _exakte_ableitung(expr)
        ok = True
        groesste = 0.0
        details: list[str] = []
        for pt in punkte:
            soll = float(exakt(pt))
            spur = _Spur(f)
            try:
                ist = _zahl(funktion(spur, pt))
            except ValueError as exc:
                if str(exc) == "None":
                    details.append(f"x = {_g(pt)}: keine Zahl zurückgegeben (None) — fehlt ein return?")
                elif str(exc) == "komplex":
                    details.append(f"x = {_g(pt)}: komplexer Wert — wurde über den Definitionsbereich hinaus gerechnet?")
                    nicht_definiert = True
                elif "math domain" in str(exc):
                    details.append(f"x = {_g(pt)}: f wurde ausserhalb ihres Definitionsbereichs "
                                   f"ausgewertet ({exc}).")
                    nicht_definiert = True
                else:
                    details.append(f"x = {_g(pt)}: Fehler beim Aufruf — {type(exc).__name__}: {exc}")
                ok = False
                continue
            except ZeroDivisionError as exc:
                details.append(f"x = {_g(pt)}: Division durch Null — {exc}. "
                               "Ist h = 0, oder fällt ein Funktionswert auf 0?")
                ok = False
                ausnahmen.append("Division durch Null")
                continue
            except Exception as exc:
                details.append(f"x = {_g(pt)}: Fehler beim Aufruf — {type(exc).__name__}: {exc}")
                ok = False
                continue

            # Wo hat deine Funktion f aufgerufen?
            ferne = [abs(q - pt) for q in spur.punkte if q != pt]
            if ferne:
                abstaende.append(min(ferne))
                links = any(q < pt for q in spur.punkte)
                rechts = any(q > pt for q in spur.punkte)
                if links and rechts:
                    einseitig_alle = False
            else:
                einseitig_alle = False

            if not math.isfinite(ist):
                details.append(f"x = {_g(pt)}: kein endlicher Wert ({ist}) — "
                               "h = 0 oder Division durch Null?")
                ok = False
                continue

            abw = abs(ist - soll)
            groesste = max(groesste, abw)
            if abw > TOLERANZ_ABLEITUNG * max(1.0, abs(soll)):
                ok = False
                details.append(f"x = {_g(pt)}: {_g(ist)} statt {_g(soll)} "
                               f"(Abweichung {_g(abw)})")
                if soll != 0 and math.isfinite(ist):
                    faktoren.append(ist / soll)
                    if ist / soll < -0.5:
                        vorzeichen_falsch += 1

        if ok:
            p.faelle.append(Fall(name, True,
                                 f"{len(punkte)} Stellen, grösste Abweichung {_g(groesste)}"))
        else:
            p.faelle.append(Fall(name, False, "", details))

    # -- Stolperstein: |x| bei 0 -------------------------------------------
    spur = _Spur(abs)
    try:
        wert = _zahl(funktion(spur, 0.0))
        if abs(wert) < 0.25:
            p.stolpersteine.append(Stolperstein(
                "|x| bei x = 0",
                f"Dein Verfahren liefert {_g(wert)}. Eine Ableitung gibt es dort aber nicht: "
                "links ist die Steigung −1, rechts +1. Der symmetrische Quotient "
                "mittelt beides zu 0 und verschweigt den Knick. Prüfe in der Praxis "
                "einseitige Quotienten, bevor du dem Ergebnis traust.", "!"))
        elif abs(wert - 1) < 0.25 or abs(wert + 1) < 0.25:
            p.stolpersteine.append(Stolperstein(
                "|x| bei x = 0",
                f"Dein Verfahren liefert {_g(wert)} — das ist die Steigung einer Seite. "
                "Die Ableitung existiert aber nicht: links −1, rechts +1. Der "
                "einseitige Quotient zeigt nur eine der beiden Seiten.", "!"))
        else:
            p.stolpersteine.append(Stolperstein(
                "|x| bei x = 0", f"Dein Verfahren liefert {_g(wert)}; die Ableitung "
                "existiert dort nicht (links −1, rechts +1).", "i"))
    except Exception as exc:
        p.stolpersteine.append(Stolperstein(
            "|x| bei x = 0", f"Bricht mit {type(exc).__name__} ab — in Ordnung, die "
            "Ableitung existiert dort nicht.", "✓"))

    # -- Hinweise zu h -------------------------------------------------------
    if abstaende:
        abstaende.sort()
        h_med = abstaende[len(abstaende) // 2]
        p.infos.append(f"Dein h liegt bei etwa {_g(h_med)} ("
                       + ("einseitiger Quotient" if einseitig_alle else "symmetrischer Quotient")
                       + ").")
    else:
        h_med = None

    if not p.ok:
        if nicht_definiert:
            p.hinweise.append(
                "Bei √x und 1/x darfst du mit f(x − h) nicht über den Definitionsbereich "
                "hinaus: ist h grösser als x (z. B. x = 0.25), liegt x − h links von 0.")
        if h_med is not None and h_med > 1e-3:
            p.hinweise.append(
                f"h ≈ {_g(h_med)} ist zu gross: der Differenzenquotient ist noch eine "
                "Sekante, nicht die Tangente (Abbruchfehler). Ein kleineres h bringt "
                "die Werte näher an die Tangentensteigung.")
        elif h_med is not None and h_med < 1e-9:
            p.hinweise.append(
                f"h ≈ {_g(h_med)} ist zu klein: f(x+h) und f(x) unterscheiden sich nur noch "
                "in den letzten Stellen. Die Differenz besteht dann aus Rundungsfehlern "
                "(Auslöschung), und das Teilen durch das winzige h vergrössert sie. "
                "Bei 16 Stellen Genauigkeit ist etwa 1e-5 bis 1e-8 ein guter Bereich.")
        elif h_med is not None and einseitig_alle:
            p.hinweise.append(
                "Der einseitige Quotient (f(x+h) − f(x))/h hat einen Fehler, der mit h "
                "wächst. Mit demselben h ist der symmetrische Quotient "
                "(f(x+h) − f(x−h))/(2h) deutlich genauer — oder du wählst h kleiner.")
        if faktoren and all(abs(q - 2) < 0.02 for q in faktoren):
            p.hinweise.append("Alle Werte sind doppelt so gross wie erwartet: beim "
                              "symmetrischen Quotienten ist der Abstand der beiden Stellen 2h — "
                              "dann musst du auch durch 2h teilen.")
        elif faktoren and all(abs(q - 0.5) < 0.02 for q in faktoren):
            p.hinweise.append("Alle Werte sind halb so gross wie erwartet: der Abstand der "
                              "beiden Stellen ist h (nicht 2h)?")
        if vorzeichen_falsch:
            p.hinweise.append("Das Vorzeichen stimmt nicht: der Differenzenquotient ist "
                              "(f(später) − f(früher)) / (später − früher).")
        if not p.hinweise:
            p.hinweise.append("Die Abweichungen sind nicht systematisch. Rechne einen Fall von "
                              "Hand nach (x² bei x = 2 ergibt 4) und vergleiche Schritt für Schritt.")
    return _abschluss(p, ausgabe)


# ==========================================================================
# Übung 2: Extrema numerisch finden
# ==========================================================================

_SW4_F = "x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27"


def _sw4(t: float) -> float:
    return t**5 - 7 * t**4 + 10 * t**3 + 18 * t**2 - 27 * t - 27


#: (Name, Funktion, Term, a, b, n, Hinweis bei Misserfolg)
EXTREMA_FAELLE: list[tuple[str, Callable[[float], float], str, str, str, int, str]] = [
    ("Beispielfunktion SW4", _sw4, _SW4_F, "-2", "4", 600,
     "Bei x = 3 hat f' eine doppelte Nullstelle: die Steigung berührt die x-Achse, "
     "wechselt aber das Vorzeichen nicht — das ist ein Sattelpunkt, kein Extremum."),
    ("Sattelpunkt x³", lambda t: t ** 3, "x^3", "-1", "1", 100,
     "f'(x) = 3x² ist links und rechts von 0 positiv. Dass sie bei 0 selbst 0 ist, "
     "reicht nicht: gesucht ist ein Vorzeichenwechsel."),
    ("x⁴ (f'' ist 0)", lambda t: t ** 4, "x^4", "-1", "1", 100,
     "Das Minimum bei 0 liegt genau auf einem Gitterpunkt, dort ist die Steigung 0. "
     "Mit einem strengen Test d[i]*d[i+1] < 0 siehst du es nicht."),
    ("konstante Funktion", lambda t: 5.0, "5", "0", "1", 50,
     "Die Steigung ist überall 0. Ohne Vorzeichenwechsel gibt es kein Extremum."),
    ("Minimum auf dem Gitterpunkt", lambda t: (t - 1) ** 2, "(x-1)^2", "0", "2", 10,
     "Das Minimum bei 1 ist ein Gitterpunkt (n = 10, Schritt 0.2). Dort ist die Steigung "
     "höchstens durch Rundung nicht genau 0 — vergleiche mit einer kleinen Schranke."),
    ("Maximum auf dem Gitterpunkt", lambda t: -(t - 1) ** 2, "-(x-1)^2", "0", "2", 10,
     "Wie vorher, nur mit umgekehrtem Vorzeichen: Steigung von plus nach minus ist ein Maximum."),
    ("Extremum zwischen Gitterpunkten", lambda t: (t - 0.33) ** 2, "(x-33/100)^2", "0", "1", 10,
     "Das Minimum 0.33 liegt zwischen 0.3 und 0.4. Es genügt die Lage auf einen Gitterschritt "
     "genau."),
    ("Rand ist kein Extremum", lambda t: t * t, "x^2", "0", "2", 20,
     "Auf [0, 2] ist x = 0 der kleinste Wert, aber kein lokales Minimum im Innern: dort fehlt "
     "die linke Nachbarseite. Melde nur Vorzeichenwechsel von f'."),
    ("sin x", math.sin, "sin(x)", "0", "4*pi", 400,
     "Zwei Minima und zwei Maxima, wieder genau auf Gitterpunkten (π/2 sind 50 Schritte). "
     "Prüfe auch, ob Minima und Maxima nicht vertauscht sind."),
]


#: Fälle, in denen ein Extremum genau auf einem Gitterpunkt liegt.
GITTER_FAELLE = {"Beispielfunktion SW4", "x⁴ (f'' ist 0)", "Minimum auf dem Gitterpunkt",
                 "Maximum auf dem Gitterpunkt", "sin x"}


def _pkt(v: float) -> str:
    return f"x ≈ {_g(v)}"


def _vergleiche_listen(art: str, ist: list[float], soll: list[float], gegen: list[float],
                       sattel: list[float], tol: float) -> list[str]:
    """Befunde zu einer Liste (Minima oder Maxima). Leer heisst: stimmt."""
    meldungen = []
    zugeordnet = [False] * len(ist)
    for s in soll:
        treffer = [i for i, v in enumerate(ist) if abs(v - s) <= tol and not zugeordnet[i]]
        if treffer:
            zugeordnet[min(treffer, key=lambda i: abs(ist[i] - s))] = True
        else:
            if any(abs(v - s) <= tol for v in gegen):
                meldungen.append(f"{art} bei {_pkt(s)} steht in der falschen Liste "
                                 "(Minimum und Maximum vertauscht?).")
            else:
                meldungen.append(f"{art} bei {_pkt(s)} nicht gefunden.")
    for i, v in enumerate(ist):
        if zugeordnet[i]:
            continue
        if any(abs(v - s) <= tol for s in sattel):
            meldungen.append(f"{_pkt(v)} als {art} gemeldet, ist aber ein Sattelpunkt "
                             "(kein Vorzeichenwechsel von f').")
        elif not any(abs(v - s) <= tol for s in gegen):
            meldungen.append(f"{_pkt(v)} als {art} gemeldet, dort gibt es keins.")
    return meldungen


def _als_liste(v) -> list[float]:
    return sorted(_zahl(t) for t in v)


def pruefe_extrema(funktion: Callable, ausfuehrlich: bool = True,
                   ausgabe: bool | None = None) -> Pruefung:
    """
    Prüft deine Suche nach Extremstellen an der Beispielfunktion aus SW4 und an
    Fällen, die man leicht übersieht.

    `funktion(f, a, b, n)` bekommt eine Python-Funktion f, das Intervall
    [a, b] und die Anzahl n der Gitterstücke. Sie soll zwei Listen
    `(minima_x, maxima_x)` zurückgeben: die x-Werte der lokalen Minima und
    Maxima im Innern von (a, b). Eine Lage auf einen Gitterschritt
    genau genügt. Referenz ist `anakit.extrema` (exakt).

    Gibt eine Pruefung zurück (in Bedingungen wie True/False benutzbar).
    `ausgabe=False` unterdrückt die Anzeige.
    """
    p = Pruefung("Lokale Extrema numerisch finden")
    p.ausfuehrlich = ausfuehrlich
    gitterproblem = False
    sattelproblem = False
    vertauscht = False

    for name, f, expr, a_s, b_s, n, hinweis in EXTREMA_FAELLE:
        a_f = float(sp.N(parse(a_s)))
        b_f = float(sp.N(parse(b_s)))
        schritt = (b_f - a_f) / n
        tol = 1.5 * schritt + 1e-9
        r_min, r_max, r_sat = _exakte_extrema(expr, a_s, b_s)
        titel = f"{name} auf [{a_s.replace('*pi', 'π')}, {b_s.replace('*pi', 'π')}], n = {n}"
        erwartet = (f"erwartet: Minima {[round(v, 3) for v in r_min]}, "
                    f"Maxima {[round(v, 3) for v in r_max]}")
        try:
            ergebnis = funktion(f, a_f, b_f, n)
            if ergebnis is None:
                raise ValueError("None")
            if not (isinstance(ergebnis, (tuple, list)) and len(ergebnis) == 2):
                raise TypeError("erwartet wird ein Paar (minima_x, maxima_x) aus zwei Listen")
            if ergebnis[0] is None or ergebnis[1] is None:
                raise ValueError("None")
            ist_min, ist_max = _als_liste(ergebnis[0]), _als_liste(ergebnis[1])
        except Exception as exc:
            if isinstance(exc, ValueError) and str(exc) == "None":
                text = "Es kam None zurück (oder eine Liste ist None) — fehlt ein return?"
            else:
                text = f"Fehler beim Aufruf — {type(exc).__name__}: {exc}"
            p.faelle.append(Fall(titel, False, "", [text]))
            continue

        befunde = (_vergleiche_listen("Minimum", ist_min, r_min, r_max + ist_max, r_sat, tol)
                   + _vergleiche_listen("Maximum", ist_max, r_max, r_min + ist_min, r_sat, tol))
        if not befunde:
            ist_text = (f"Minima {[round(v, 3) for v in ist_min]}, "
                        f"Maxima {[round(v, 3) for v in ist_max]}")
            p.faelle.append(Fall(titel, True, ist_text if (r_min or r_max) else
                                 "keine Extrema gemeldet — richtig", [hinweis]))
        else:
            p.faelle.append(Fall(titel, False, "", befunde + [erwartet, "Hinweis: " + hinweis]))
            if any("falschen Liste" in b for b in befunde):
                vertauscht = True
            if any("Sattelpunkt" in b for b in befunde):
                sattelproblem = True
            if any("nicht gefunden" in b for b in befunde) and name in GITTER_FAELLE:
                gitterproblem = True

    if not p.ok:
        if gitterproblem:
            p.hinweise.append(
                "Extremum auf dem Gitterpunkt verpasst: dort ist die Steigung 0 (oder wegen "
                "Rundung winzig), das Produkt d[i]·d[i+1] ist nicht negativ. Ordne die Steigung "
                "in ‹klar positiv›, ‹klar negativ› und ‹ungefähr 0› ein und suche den Wechsel von "
                "positiv zu negativ (oder umgekehrt) über die Nullen hinweg.")
        if sattelproblem:
            p.hinweise.append(
                "Ein Sattelpunkt ist kein Extremum: dort wird die Steigung null, aber sie "
                "wechselt das Vorzeichen nicht. Wenn du über Nullstellen von f' suchst, "
                "musst du das Vorzeichen davor und danach vergleichen.")
        if vertauscht:
            p.hinweise.append("Minimum: Steigung von minus nach plus. Maximum: von plus nach minus.")
        if not p.hinweise:
            p.hinweise.append("Vergleiche mit der erwarteten Lage in der Meldung. Zeichne f' "
                              "und markiere die Vorzeichenwechsel.")
    return _abschluss(p, ausgabe)


# ==========================================================================
# Nach dem Üben
# ==========================================================================

def vergleiche_mit_anakit(f, domain=None) -> Result:
    """Zeigt die exakten Extrema von anakit mit Rechenweg — nach dem Üben."""
    return extrema(f, domain)
