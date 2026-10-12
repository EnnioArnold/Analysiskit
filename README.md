# anakit

Analysis mit nachvollziehbarem Rechenweg, gebaut zum Modul ANA-G der HSLU
(Grundlagen der Analysis). Deckt die Semesterwochen 1 bis 4 ab. Aufbau und
Bedienung folgen [lialkit](https://github.com/EnnioArnold/lialkit).

| Woche | Thema | Funktionen |
|---|---|---|
| SW1 | Sekanten und Tangenten | `slope`, `line_through`, `line_point_slope`, `secant`, `secant_h`, `secant_table`, `difference_quotient`, `tangent_slope`, `tangent`, `derivative_by_definition` |
| SW2 | Grenzwert und Stetigkeit | `limit`, `limit_infinity`, `limit_laws`, `limit_table`, `continuity`, `is_continuous` |
| SW3 | Ableitungsregeln | `diff`, `diff_at`, `rule`, `rule_table`, `check_derivative`, `diff_sum/product/quotient/chain/inverse` |
| SW4 | Monotonie und Extrema | `monotony`, `critical_points`, `curvature`, `extrema`, `inflection_points`, `curve_discussion` |

## Installieren

```bash
pip install -r requirements.txt      # sympy, numpy, matplotlib, plotly, PySide6
brew install cairo pango pkg-config ffmpeg   # Unterbau für Manim, freiwillig
pip install manim                    # Animationen, freiwillig
```

Der Ordner `anakit/` liegt neben deinen Wochen-Ordnern. Aus einem Notebook
in `Week 3/` heraus:

```python
import sys
sys.path.insert(0, '..')
import anakit as ak
```

## Benutzen

```python
ak.secant("x^2", -1, 3)                    # Sekantensteigung
ak.tangent("1/x", 1)                       # Tangente: y = 2 - x
ak.derivative_by_definition("x^3")         # über h -> 0, binomischer Satz

ak.limit("(x^2-1)/(x-1)", 1)               # 2
ak.limit("x^2 if x<1 else x-1", 1)         # links 1, rechts 0: existiert nicht
ak.continuity("x^2 if x<1 else x-1", 1)    # Sprungstelle

ak.diff("ln(x^2+1)")                       # Kettenregel, Schritt für Schritt
ak.rule_table()                            # Tabelle Newton / Leibniz

ak.curve_discussion("x^5-7x^4+10x^3+18x^2-27x-27")
```

`^` bedeutet Potenz, `3x` bedeutet `3*x`. Abschnittsweise Funktionen
schreibst du als `x^2 if x<1 else 2-x`. Jedes Ergebnis zeigt sich im Notebook
von selbst; `.as_decimal()` schaltet auf Dezimalzahlen um, `.latex()` liefert
den Rechenweg für eine Abgabe, `.figure()` die Zeichnung.

## Genauigkeit

Gerechnet wird exakt mit SymPy: `1/3` bleibt `1/3`, `sqrt(2)` bleibt
`sqrt(2)`. Der Rechenweg folgt den Regeln der Vorlesung: Jeder Schritt nennt
seine Regel (Produktregel, Kettenregel, …), und das Ergebnis wird am Ende
gegen `sympy.diff` geprüft.

Hinweise zu den Folien: In SW2 sind die Vorzeichen von `tan` bei π/2 vertauscht
dargestellt (von rechts ist der Grenzwert −∞); anakit rechnet richtig und
sagt es im Rechenweg. Die Heaviside-Funktion gilt wie auf der Folie mit
H(0) = 1.

## Die Übungsaufgaben

```python
from anakit.exercises import liste, aufgabe, loese

liste('SW3')          # welche es gibt
aufgabe('SW3.2')      # die Angabe
loese('SW3.2')        # durchgerechnet
```

Enthalten sind alle Rechenaufgaben und Beispiele der Folien. Die Aufgaben aus
Papula und Hughes-Hallet stehen nur als Leseliste (`SWn.DIY`) mit Seitenzahlen
da, ohne Lösung.

**Zum Selbstprogrammieren** gibt es zwei Prüfstände ohne Musterlösung:

```python
from anakit.uebung import vorlage_ableitung, pruefe_ableitung
print(vorlage_ableitung())

def meine_ableitung(f, x):
    ...

pruefe_ableitung(meine_ableitung)     # numerische Ableitung prüfen
# ebenso: vorlage_extrema(), pruefe_extrema()
```

## Das Fenster

Ordner entpacken, dann im Terminal:

```bash
cd anakit                        # Ordnername ist egal
pip install -r requirements.txt  # einmalig
python3 anakit_qt.py
```

Vier Seiten: **Steigung & Tangente**, **Grenzwerte**, **Ableiten**,
**Kurvendiskussion**. Jedes Ergebnis zeigt **Rechenweg**, **Zeichnung** und
**Animation**. Die Prüfung über Wolfram|Alpha gibt es auf jeder Seite.

## Zeichnungen und Animationen

`plot_function`, `plot_secant`, `plot_secants_to_tangent`, `plot_tangent`,
`plot_derivative`, `plot_curve`, `plot_limit` liefern matplotlib-Figuren
(`dark=True` für den Dunkelmodus). `plotly_function` öffnet dieselbe Funktion
interaktiv. Manim rendert drei Szenen: Sekante wird zur Tangente, Entstehung
des Graphen von f′, Kurvenanalyse (`anakit.anim`).

## Tests

```bash
python3 tests/tests_steigung.py     # ebenso limits, derivative, analysis,
python3 tests/tests_exercises.py    # exercises, plots, app
```
