# anakit (Analysiskit, ANA-G, HSLU)

Analysis mit nachvollziehbarem Rechenweg. Sprache: Deutsch (Docstrings, Texte,
Commits). Rechnen exakt mit SymPy; nur Numerik (SW5) rechnet mit float.

## Aufbau

- `rechnen/` Rechenkern, je Semesterwoche ein Modul (`steigung` SW1, `limits` SW2,
  `derivative` SW3, `analysis` SW4, `numerik` SW5, `kurve` + `reihen` SW6)
- `grafik/` `plots.py` (Matplotlib), `anim.py` (Manim/manimgx)
- `aufgaben/` `exercises.py` (Aufgaben als Daten + `loese`), `uebung.py` (Prüfstände)
- `app/anakit_qt.py` das Qt-Fenster, `tests/` ein Skript je Modul, `docs/` Diagramm
- Der Ordner muss als Paket `anakit` importierbar sein (Elternordner auf `sys.path`,
  heisst der Ordner anders: Symlink `anakit` anlegen). Tests laufen als Skripte.

## Konventionen

- Jedes Verfahren liefert ein `Result` (`rechnen/render.py`) mit `steps` (`Step`),
  `highlights`, `summary`, `.figure()`; Regeln im Step beim Namen nennen (`rule=`).
- Modulname darf nicht gleich einem exportierten Funktionsnamen sein (Alias-Schleife in
  `__init__.py` überschreibt sonst die Funktion: darum `reihen.py`, nicht `taylor.py`).
- Aufgaben nur aus den Unterlagen; fehlt ein Beispiel auf den Folien, eigenes Beispiel
  mit „(eigenes Beispiel)“ im Titel und rechnerisch verifiziert im Test. Unklares als TODO.
- Keine Breaking Changes an früheren Wochen.

## Neue Woche hinzufügen (Checkliste)

1. Folien (PDF) komplett lesen: Themen, Definitionen, Formeln, Beispiele, DIY-Seite.
2. Rechenmodul `rechnen/<thema>.py` mit Docstring (Folienbezug), `Result`-Klassen,
   `__all__`; vorhandene Module wiederverwenden (z. B. `analysis._find_roots`).
3. Zeichnung `plot_*` in `grafik/plots.py` (+ `__all__`); Animation in `grafik/anim.py`
   nur falls sinnvoll (`SCENES`, `ANIM_SCENES` in der App).
4. `__init__.py`: Import, `__all__`, `info()`, Docstring und Alias-Liste ergänzen.
5. `aufgaben/exercises.py`: `_ex(...)`-Einträge `SWn.k`, `SWn.DIY`-Leseliste (`_diy`), das
   Modul in `_lookup()` und die Argumentnamen in `_ARGS` eintragen.
6. App: Aktionen, `EXAMPLES`, `DEFAULTS`, `calc_<seite>` + `CALC`, `WOLFRAM_CHECKS` +
   `wolfram_subtitle`, Seitenklasse (`build_form`, `describe`), `PAGE_CLASSES`,
   `PAGE_KEYS`; Animation ggf. in `anim_job_for` ablehnen.
7. Tests: `tests/tests_<modul>.py` (Werte aus Folien/Notebooks), `tests_exercises.py`
   (Wochenliste, Werte), `tests_app.py` (Beispiele laufen automatisch).
8. README (Wochentabelle, Beispiele) und dieses Dokument aktualisieren.
9. Alle `tests/tests_*.py` ausführen; Commits klein, ohne Co-Authored-By-Trailer.
