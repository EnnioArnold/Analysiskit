#!/usr/bin/env python3
"""
Tests für anakit_qt (Fenster ohne Bildschirm): python3 tests/tests_app.py

Läuft mit QT_QPA_PLATFORM=offscreen, braucht kein Netz. Legt Bilder der
vier Seiten in tests/shots/ ab (oder dort, wo ANAKIT_SHOTS hinzeigt).
"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'pkg'))
import sympy as sp
import anakit_qt as A

SHOTS = os.environ.get("ANAKIT_SHOTS", os.path.join(HERE, "shots"))
os.makedirs(SHOTS, exist_ok=True)
N = {"ok": 0}


def check(cond, msg=""):
    assert cond, msg
    N["ok"] += 1


def raises_input(page, p, word=""):
    try:
        A.calc(page, p)
    except A.InputError as exc:
        check(word.lower() in str(exc).lower(), f"{p}: {exc}")
        return
    raise AssertionError(f"InputError erwartet: {p}")


# ---------------------------------------------------------------- Wolfram-Parser
check(sp.simplify(A.parse_wolfram_expr("3 x^2 - 3") - (3 * A.x**2 - 3)) == 0)
check(sp.simplify(A.parse_wolfram_expr("d/dx(x^3) = 3 x^2") - 3 * A.x**2) == 0)
check(sp.simplify(A.parse_wolfram_expr("y = 2 - x") - (2 - A.x)) == 0)
try:
    A.parse_wolfram_expr("(no result found)")
    check(False, "sollte werfen")
except ValueError:
    check(True)
check(A.parse_wolfram_limit("2") == ("value", 2))
check(A.parse_wolfram_limit("∞")[1] == sp.oo)
check(A.parse_wolfram_limit("-∞")[1] == -sp.oo)
check(A.parse_wolfram_limit("(limit does not exist)")[0] == "dne")
check(A.parse_wolfram_points("x = -1 and x = 1") == [-1.0, 1.0])
check(A.parse_wolfram_points("no critical points") == [])
ext = A.parse_wolfram_extrema("max{x^3 - 3 x} = 2 at x = -1\nmin{x^3 - 3 x} = -2 at x = 1")
check([(k, xv) for k, xv, _ in ext] == [("max", -1.0), ("min", 1.0)], ext)
check(A.exprs_equal(A.x**2 - 1, (A.x - 1) * (A.x + 1)) is True)
check(A.exprs_equal(A.x**2, A.x**3) is False)
check(A.values_equal(sp.sqrt(2), 1.41421356237))

# ---------------------------------------------------------------- Prüfläufe mit Attrappe
r = A.calc("derivative", {"f": "x^3", "action": "diff", "n": "1"})
rep = A.run_wolfram_checks("X", "derivative", {"f": "x^3", "action": "diff", "n": "1"}, r,
                           ask=lambda appid, q: "3 x^2")
check(rep.verdict == "ok", [(c.name, c.status) for c in rep.checks])
rep = A.run_wolfram_checks("X", "derivative", {"f": "x^3", "action": "diff", "n": "1"}, r,
                           ask=lambda appid, q: "2 x")
check(rep.verdict == "bad")
rep = A.run_wolfram_checks("X", "derivative", {"f": "x^3", "action": "diff", "n": "1"}, r,
                           ask=lambda appid, q: "Wolfram weiss es nicht")
check(rep.verdict != "bad" and any(c.status == "info" for c in rep.checks))


def offline(appid, q):
    raise RuntimeError("Kein Netz")


try:
    A.run_wolfram_checks("X", "derivative", {"f": "x^3", "action": "diff", "n": "1"}, r, ask=offline)
    check(False)
except RuntimeError:
    check(True)
r = A.calc("limits", {"f": "(x^2-1)/(x-1)", "action": "limit", "x0": "1", "side": "both"})
rep = A.run_wolfram_checks("X", "limits", {"f": "(x^2-1)/(x-1)", "action": "limit", "x0": "1",
                                           "side": "both"}, r, ask=lambda a, q: "2")
check(rep.verdict == "ok")
p = {"f": "x^3-3x", "action": "all", "a": "", "b": ""}
r = A.calc("curve", p)
answers = {"critical": "x = -1 and x = 1", "local extrema": "max{x^3 - 3 x} = 2 at x = -1\n"
           "min{x^3 - 3 x} = -2 at x = 1", "inflection": "x = 0"}


def curve_ask(appid, q):
    for k, v in answers.items():
        if q.startswith(k):
            return v
    raise RuntimeError(q)


rep = A.run_wolfram_checks("X", "curve", p, r, ask=curve_ask)
check(rep.verdict == "ok", [(c.name, c.status) for c in rep.checks])
r = A.calc("steigung", {"f": "x^2", "method": "tan", "x0": "1", "variant": "x"})
rep = A.run_wolfram_checks("X", "steigung", {"f": "x^2", "method": "tan", "x0": "1", "variant": "x"},
                           r, ask=lambda a, q: "2" if "derivative" in q else "y = 2 x - 1")
check(rep.verdict == "ok", [(c.name, c.status, c.theirs) for c in rep.checks])

# ---------------------------------------------------------------- Eingabefehler
raises_input("derivative", {"f": "x^^", "action": "diff", "n": "1"})
raises_input("derivative", {"f": "", "action": "diff", "n": "1"})
raises_input("limits", {"f": "1/0", "action": "limit", "x0": "1", "side": "both"})
raises_input("steigung", {"f": "x^2", "method": "sek12", "x1": "1", "x2": "1"}, "verschieden")
raises_input("limits", {"f": "x", "action": "limit", "x0": "abc", "side": "both"})
msg = A.friendly_error(ZeroDivisionError("x"))
check(isinstance(msg, str) and msg)

# ---------------------------------------------------------------- Rechnen ohne Fenster
for key, examples in A.EXAMPLES.items():
    for name, params in examples:
        q = dict(A.DEFAULTS[key]); q.update(params)
        res = A.calc(key, q)
        vm = A.build_vm(res, False)
        check(vm.steps or vm.highlights, (key, name))
        check(A.build_vm(res, True) is not None)
        check(isinstance(res.latex(), str))

# ---------------------------------------------------------------- Fenster
app, win = A.create_app([])
win.show()
app.processEvents()
for i, page in enumerate(win.pages):
    win.set_page(i)
    page.load_example(A.EXAMPLES[page.KEY][0][0])
    page.wait()
    check(page.result is not None and page.vm is not None, page.KEY)
    for view, tab in ((0, "rechenweg"), (1, "zeichnung"), (2, "animation")):
        A.show_view(page, view)
        page.wait()
        for _ in range(10):
            app.processEvents()
        win.grab().save(os.path.join(SHOTS, f"app_{page.KEY}_{tab}.png"))
    A.show_view(page, 0)
    check(page.plot.figure is not None, f"{page.KEY}: keine Zeichnung")

# Fehlerkarte statt Absturz
page = win.pages[2]
win.set_page(2)
page.p["f"] = "x^^"
page.sync_widgets()
page.compute(); page.wait()
check(page.result is None)
app.processEvents()
win.grab().save(os.path.join(SHOTS, "app_fehler.png"))

# Verlauf, Export, Dezimal, Thema
page = win.pages[0]
win.set_page(0)
for name, _ in A.EXAMPLES["steigung"][:12]:
    page.load_example(name); page.wait()
check(len(page.history) <= A.HISTORY_LIMIT)
page.copy_latex()
check(A.QGuiApplication.clipboard().text() == page.result.latex())
win.set_theme(1); app.processEvents()
page.wait()
win.grab().save(os.path.join(SHOTS, "app_steigung_dunkel.png"))
win.set_theme(0)

# Wolfram-Bereich offline (Attrappe statt Netz)
panel = page.wolfram
panel.build_checks = lambda appid: (lambda: A.run_wolfram_checks(
    appid, page.KEY, dict(page.result_p), page.result, ask=offline))
A.load_appid = lambda: "TEST"
panel.start(); panel.wait(20000)
check(panel.report is None and panel.status and panel.status[0] == "bad", panel.status)
panel.build_checks = lambda appid: (lambda: A.run_wolfram_checks(
    appid, page.KEY, dict(page.result_p), page.result, ask=lambda a, q: "y = 2 x - 1"))
panel.start(); panel.wait(20000)
check(panel.report is not None, (panel.status, page.result_p))
win.grab().save(os.path.join(SHOTS, "app_wolfram.png"))

# Animation: ohne Manim sagt die Karte, warum nichts geht, und stürzt nicht ab
ok, why = A.anim_state()
check(isinstance(ok, bool))
job = A.anim_job_for("steigung", {"f": "x^2", "method": "tan", "x0": "1"}, object(),
                     "secant_to_tangent", "medium") if A.ak_anim else None
check(job is None or hasattr(job, "scene") or isinstance(job, str))

# Zeichnung: Mausrad zoomt, Doppelklick setzt zurück, Hover zeigt Koordinaten
import matplotlib.pyplot as plt
from types import SimpleNamespace as NS
pc = A.PlotCanvas()
fig, ax_ = plt.subplots()
xs_ = [i / 10 for i in range(-50, 51)]
ax_.plot(xs_, [x * x for x in xs_])
pc.show_figure(fig)
xl0 = ax_.get_xlim()
pc._on_scroll(NS(inaxes=ax_, xdata=0.0, ydata=0.0, step=1))
check(ax_.get_xlim()[1] - ax_.get_xlim()[0] < xl0[1] - xl0[0], "Zoom")
pc._on_press(NS(inaxes=ax_, button=1, dblclick=True, x=0, y=0))
check(tuple(ax_.get_xlim()) == tuple(xl0), "Reset")
pc._on_move(NS(inaxes=ax_, xdata=2.0, ydata=0.0, x=0, y=0))
check("Kurve: 4.000" in pc.readout.text(), pc.readout.text())
pc.clear()

win.close()
print(f"tests_app: {N['ok']} Prüfungen ok. Bilder in {SHOTS}")
