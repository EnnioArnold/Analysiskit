# anakit

Analysis mit nachvollziehbarem Rechenweg, gebaut zum Modul ANA-G der HSLU.
Deckt die Semesterwochen 1 bis 4 ab: Sekanten und Tangenten, Grenzwerte und
Stetigkeit, Ableitungsregeln, Extrema und Kurvendiskussion.

## Installieren

Nötig ist nur Python mit SymPy und NumPy. Für die Zeichnungen und die App
kommt dazu:

```bash
pip install -r requirements.txt      # sympy, numpy, matplotlib, plotly, PySide6
brew install cairo pango pkg-config ffmpeg   # Unterbau für Manim
pip install manim                    # Animationen, freiwillig
```

Der Ordner `anakit/` liegt neben deinen Wochen-Ordnern. Aus einem
Notebook in `Week 3/` heraus:

```python
import sys
sys.path.insert(0, '..')
import anakit as ak

ak.tangent("1/x", 1)                       # y = 2 - x
ak.limit("(x^2-1)/(x-1)", 1)               # 2
ak.diff("ln(x^2+1)")                       # Kettenregel, Schritt für Schritt
ak.curve_discussion("x^5-7x^4+10x^3+18x^2-27x-27")
```

`^` bedeutet Potenz, `3x` bedeutet `3*x`, abschnittsweise Funktionen
schreibst du als `x^2 if x<1 else 2-x`. Jedes Ergebnis zeigt sich im Notebook
von selbst; `.as_decimal()` schaltet auf Dezimalzahlen um, `.latex()` liefert
den Rechenweg für eine Abgabe, `.figure()` die Zeichnung.

## Aufbau

![Notebook und Fenster rechnen mit demselben Modul](docs/architektur.svg)

Notebook und Fenster teilen sich den Rechenkern, darum können ihre Ergebnisse
nicht auseinanderlaufen.

## Genauigkeit

Gerechnet wird exakt mit SymPy: `1/3` bleibt `1/3`, `sqrt(2)` bleibt
`sqrt(2)`. Jeder Schritt nennt seine Regel (Produktregel, Kettenregel, …),
und das Ergebnis wird am Ende gegen `sympy.diff` geprüft.

In den Folien zu SW2 sind die Vorzeichen von `tan` bei π/2 vertauscht
dargestellt (von rechts ist der Grenzwert −∞). anakit rechnet richtig und
sagt es im Rechenweg.

## Die Übungsaufgaben

```python
from anakit.exercises import liste, aufgabe, loese

liste('SW3')          # welche es gibt
aufgabe('SW3.2')      # die Angabe
loese('SW3.2')        # durchgerechnet
```

Enthalten sind alle Rechenaufgaben und Beispiele der Folien. Die Aufgaben aus
Papula und Hughes-Hallet stehen nur als Leseliste (`SWn.DIY`) mit Seitenzahlen
da.

**Zum Selbstprogrammieren** gibt es zwei Prüfstände ohne Musterlösung:

```python
from anakit.uebung import vorlage_ableitung, pruefe_ableitung
print(vorlage_ableitung())

def meine_ableitung(f, x):
    ...

pruefe_ableitung(meine_ableitung)     # ebenso: vorlage_extrema(), pruefe_extrema()
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

Die Zeichnungen (`plot_*`) sind matplotlib-Figuren, `dark=True` schaltet den
Dunkelmodus ein, `plotly_function` öffnet sie interaktiv. Manim rendert drei
Szenen: Sekante wird zur Tangente, Entstehung des Graphen von f′ und die
Kurvenanalyse (`anakit.anim`).

## Tests

```bash
python3 tests/tests_steigung.py     # ebenso limits, derivative, analysis,
python3 tests/tests_exercises.py    # exercises, plots, app
```
