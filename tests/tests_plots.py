#!/usr/bin/env python3
"""
Tests für anakit.plots — laufen ohne Bildschirm:

    MPLBACKEND=Agg python3 tests/tests_plots.py
    MPLBACKEND=Agg python3 -m pytest tests/tests_plots.py     # geht auch

Geprüft wird, dass für jede Zeichnung eine Figure entsteht und die
Spezialfälle (Pole, Sprünge, Lücken, Knicke, abschnittsweise Funktionen)
keine Ausnahme auslösen. Dazu ein paar inhaltliche Prüfungen der
Unstetigkeitssuche.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
# Das Paket heisst anakit; sein Elternordner muss auf dem Suchpfad liegen.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.figure import Figure  # noqa: E402

from anakit import plots as P  # noqa: E402

CURVE = "x^5 - 7x^4 + 10x^3 + 18x^2 - 27x - 27"
SPECIAL = [
    "x^2", "x", "5", "sin(x)", "exp(x)", "ln(x)", "sqrt(x)", "abs(x)", "sign(x)",
    "1/x", "1/x^2", "tan(x)", "Heaviside(x)", "(x^2-1)/(x-1)", "sin(x)/x",
    "x^2 if x<1 else x-1", "x^2 if x<0 else 1/x", "sin(1/x)", "x^(1/3)",
    "1/(x^2-1)", "x*abs(x)", "ln(x^2+1)", "exp(-x^2)", "sqrt(1-x^2)",
]


def is_figure(obj) -> bool:
    return isinstance(obj, Figure)


def finish(fig):
    """Wirklich zeichnen, damit Layout-Fehler auffallen."""
    fig.canvas.draw()
    return fig


class TestGrundfunktionen(unittest.TestCase):
    def test_plot_function(self):
        for f in SPECIAL:
            with self.subTest(f=f):
                self.assertTrue(is_figure(finish(P.plot_function(f))))

    def test_plot_function_optionen(self):
        fig = P.plot_function("1/x", xlim=(-3, 3), ylim=(-5, 5),
                              points=[1, (2, 0.5), (-1, -1, "Q")], title="Hyperbel")
        self.assertTrue(is_figure(finish(fig)))
        self.assertEqual(fig.axes[0].get_ylim(), (-5.0, 5.0))
        self.assertEqual(fig.axes[0].get_title(loc="left"), "Hyperbel")

    def test_dark(self):
        light = P.plot_function("x^2")
        dark = P.plot_function("x^2", dark=True)
        self.assertNotEqual(light.get_facecolor(), dark.get_facecolor())
        self.assertTrue(is_figure(finish(P.plot_curve(CURVE, (-1.5, 4.5), dark=True))))

    def test_callable_und_sympy(self):
        import sympy as sp
        from anakit.core import x
        self.assertTrue(is_figure(finish(P.plot_function(lambda t: t ** 2))))
        self.assertTrue(is_figure(finish(P.plot_function(sp.sin(x) / x))))

    def test_schlechte_eingaben(self):
        with self.assertRaises(ValueError):
            P.plot_function("x^2", xlim=(3, 1))
        with self.assertRaises(ValueError):
            P.plot_function("x^2", ylim=(1, 1))
        with self.assertRaises(ValueError):
            P.plot_function("x^2", points=["abc"])
        with self.assertRaises(ValueError):
            P.plot_secant("x^2", 1, 1)
        with self.assertRaises(ValueError):
            P.plot_tangent("1/x", 0)            # Pol: kein Funktionswert
        with self.assertRaises(ValueError):
            P.plot_limit("x^2", 10, xlim=(-1, 1))


class TestSekanteTangente(unittest.TestCase):
    def test_secant(self):
        self.assertTrue(is_figure(finish(P.plot_secant("x^2", -1, 3))))
        self.assertTrue(is_figure(finish(P.plot_secant("x^2", 3, -1, xlim=(-2, 4)))))
        self.assertTrue(is_figure(finish(P.plot_secant("sin(x)", 0, "pi/2"))))
        self.assertTrue(is_figure(finish(P.plot_secant("1/x", 1, 3, dark=True))))
        # fallende Sekante
        self.assertTrue(is_figure(finish(P.plot_secant("-x^2", 1, 3))))

    def test_secants_to_tangent(self):
        self.assertTrue(is_figure(finish(P.plot_secants_to_tangent("x^2", 1))))
        self.assertTrue(is_figure(finish(
            P.plot_secants_to_tangent("sin(x)", 0, hs=(-1, -0.5, 0.5, 1), dark=True))))
        # nicht differenzierbar: nur Sekanten
        self.assertTrue(is_figure(finish(P.plot_secants_to_tangent("abs(x)", 0))))

    def test_tangent(self):
        for f, x0 in [("1/x", 1), ("x^2", 1), ("sin(x)", "pi/2"), ("exp(x)", 0),
                      ("sqrt(x)", 4), ("tan(x)", 0.5), ("x^2 if x<1 else x-1", 0.5)]:
            with self.subTest(f=f):
                self.assertTrue(is_figure(finish(P.plot_tangent(f, x0))))
        self.assertTrue(is_figure(finish(P.plot_tangent("x^3", 1, secants_h=(1, 0.5)))))

    def test_tangent_sonderfaelle(self):
        # Knick, senkrechte Tangente, Sprung: keine Ausnahme, keine Tangente
        for f, x0 in [("abs(x)", 0), ("x^(1/3)", 0), ("x^2 if x<1 else x-1", 1)]:
            with self.subTest(f=f):
                try:
                    fig = finish(P.plot_tangent(f, x0))
                except ValueError:                  # x^(1/3) links nicht definiert
                    continue
                self.assertTrue(is_figure(fig))

    def test_steigung(self):
        fn = P._Fn("x^2")
        m, bad = P._slope(fn, P._deriv(fn), 1.0)
        self.assertAlmostEqual(m, 2.0, places=6)
        self.assertIsNone(bad)
        fn = P._Fn("abs(x)")
        m, bad = P._slope(fn, P._deriv(fn), 0.0)
        self.assertIsNone(m)
        self.assertAlmostEqual(bad[0], -1.0, places=4)
        self.assertAlmostEqual(bad[1], 1.0, places=4)


class TestAbleitungKurve(unittest.TestCase):
    def test_derivative(self):
        for f in ["sin(x)", "x^3 - 3x", "abs(x)", "1/x", "ln(x)", "x^2 if x<1 else 2-x",
                  "Heaviside(x)", "exp(-x^2)"]:
            with self.subTest(f=f):
                fig = finish(P.plot_derivative(f))
                self.assertTrue(is_figure(fig))
                self.assertEqual(len(fig.axes), 2)

    def test_curve_beispiel(self):
        fig = finish(P.plot_curve(CURVE, (-1.5, 4.5)))
        self.assertTrue(is_figure(fig))
        self.assertEqual(len(fig.axes), 3)

    def test_curve_spezialfaelle(self):
        for f in SPECIAL:
            with self.subTest(f=f):
                self.assertTrue(is_figure(finish(P.plot_curve(f))))

    def test_kritische_punkte(self):
        fn = P._Fn(CURVE)
        d1 = P._deriv(fn)
        d2 = P._deriv(d1)
        lo, hi = -1.5, 4.5
        crit, infl = P._analyse(fn, d1, d2, P._scan(fn, lo, hi), P._scan(d1, lo, hi),
                                P._scan(d2, lo, hi))
        kinds = sorted((p.kind, round(p.x, 3)) for p in crit)
        # Hochpunkt bei -1, Tiefpunkt bei 0.6, Sattelpunkt bei 3
        self.assertEqual([k for k, _ in kinds], ["max", "min", "sat"])
        self.assertAlmostEqual(kinds[0][1], -1.0, places=3)
        self.assertAlmostEqual(kinds[1][1], 0.6, places=3)
        self.assertAlmostEqual(kinds[2][1], 3.0, places=3)
        self.assertEqual(len(infl), 2)          # die dritte Nullstelle von f'' ist der Sattelpunkt

    def test_knick_ist_extremum(self):
        fn = P._Fn("abs(x)")
        d1 = P._deriv(fn)
        d2 = P._deriv(d1)
        crit, infl = P._analyse(fn, d1, d2, P._scan(fn, -3, 3), P._scan(d1, -3, 3),
                                P._scan(d2, -3, 3))
        self.assertEqual([(p.kind, round(p.x, 6)) for p in crit], [("min", 0.0)])
        self.assertEqual(infl, [])


class TestGrenzwert(unittest.TestCase):
    def test_limit(self):
        for f, x0 in [("tan(x)", "pi/2"), ("x^2 if x<1 else x-1", 1), ("(x^2-1)/(x-1)", 1),
                      ("sin(x)/x", 0), ("1/x", 0), ("1/x^2", 0), ("x^2", 2), ("sqrt(x)", 0),
                      ("ln(x)", 0), ("sin(1/x)", 0), ("Heaviside(x)", 0), ("abs(x)/x", 0)]:
            with self.subTest(f=f):
                self.assertTrue(is_figure(finish(P.plot_limit(f, x0))))
        self.assertTrue(is_figure(finish(P.plot_limit("tan(x)", "pi/2", dark=True))))

    def test_einseitige_grenzwerte(self):
        def sides(f, c):
            fn = P._Fn(f)
            sc = P._scan(fn, c - 3, c + 3)
            return P._side(fn, c, -1, sc.S, sc.M), P._side(fn, c, 1, sc.S, sc.M)
        (ls, lv), (rs, rv) = sides("x^2 if x<1 else x-1", 1)
        self.assertEqual((ls, rs), ("f", "f"))
        self.assertAlmostEqual(lv, 1.0, places=5)
        self.assertAlmostEqual(rv, 0.0, places=5)
        (ls, lv), (rs, rv) = sides("1/x", 0)
        self.assertEqual((ls, rs), ("p", "p"))
        self.assertLess(lv, 0)
        self.assertGreater(rv, 0)
        (ls, lv), (rs, rv) = sides("sin(x)/x", 0)
        self.assertEqual((ls, rs), ("f", "f"))
        self.assertAlmostEqual(lv, 1.0, places=5)
        (ls, _), (rs, _) = sides("ln(x)", 0)
        self.assertEqual((ls, rs), ("n", "p"))
        (ls, _), (rs, _) = sides("sin(1/x)", 0)
        self.assertEqual((ls, rs), ("o", "o"))


class TestUnstetigkeiten(unittest.TestCase):
    def events(self, f, lo=-5, hi=5):
        return P._scan(P._Fn(f), lo, hi).events

    def test_pole(self):
        ev = self.events("1/x")
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0].x, 0.0)
        self.assertTrue(ev[0].pole)
        ev = self.events("tan(x)")
        self.assertEqual(sorted(round(e.x, 4) for e in ev),
                         [round(v, 4) for v in (-3 * math.pi / 2, -math.pi / 2,
                                                math.pi / 2, 3 * math.pi / 2)])
        self.assertTrue(all(e.pole for e in ev))
        # Pol ohne Vorzeichenwechsel und Pol zwischen Rasterpunkten
        self.assertEqual(len(self.events("1/x^2")), 1)
        ev = self.events("1/(x-0.3)^2")
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0].x, 0.3, places=4)

    def test_sprung(self):
        ev = self.events("x^2 if x<1 else x-1")
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual((round(e.x, 6), round(e.lv, 4), round(e.rv, 4)), (1.0, 1.0, 0.0))
        self.assertEqual(round(e.val, 6), 0.0)          # f(1) = 0 gehört zum rechten Stück
        e = self.events("Heaviside(x)")[0]
        self.assertEqual((e.lv, e.rv, e.val), (0.0, 1.0, 0.5))

    def test_luecke(self):
        e = self.events("(x^2-1)/(x-1)")[0]
        self.assertEqual((round(e.x, 6), round(e.lv, 4), round(e.rv, 4)), (1.0, 2.0, 2.0))
        self.assertIsNone(e.val)
        e = self.events("sin(x)/x")[0]
        self.assertAlmostEqual(e.lv, 1.0, places=5)
        self.assertIsNone(e.val)

    def test_definitionsrand(self):
        e = self.events("ln(x)")[0]
        self.assertEqual((e.ls, e.rs), ("n", "p"))
        e = self.events("sqrt(x)")[0]
        self.assertEqual((e.ls, e.rs, e.val), ("n", "f", 0.0))

    def test_stetige_ohne_ereignisse(self):
        for f in ["x^2", "sin(x)", "exp(x)", "abs(x)", "x^3-3x", "5"]:
            with self.subTest(f=f):
                self.assertEqual(self.events(f), [])

    def test_kurve_wird_am_pol_getrennt(self):
        sc = P._scan(P._Fn("1/x"), -5, 5)
        self.assertEqual(len(sc.segments), 2)
        for sx, _sy in sc.segments:
            self.assertTrue((sx < 0).all() or (sx > 0).all())

    def test_ylim_robust_gegen_pole(self):
        for f in ["tan(x)", "1/x", "1/x^2", "1/(x-1)"]:
            sc = P._scan(P._Fn(f), -5, 5)
            lo, hi = P._auto_ylim(sc)
            self.assertLess(hi - lo, 60, f)


class TestPlotly(unittest.TestCase):
    def test_ohne_plotly_klare_meldung(self):
        if P.has_plotly():
            self.skipTest("Plotly ist installiert")
        with self.assertRaises(RuntimeError) as cm:
            P.plotly_function("x^2")
        self.assertIn("pip install plotly", str(cm.exception))

    def test_mit_plotly(self):
        if not P.has_plotly():
            self.skipTest("Plotly fehlt")
        fig = P.plotly_function("1/x")
        self.assertTrue(hasattr(fig, "write_html"))
        self.assertEqual(len(P.plotly_function("x^2", with_derivatives=False).data), 1)

    def test_save_html(self):
        with tempfile.TemporaryDirectory() as d:
            p = P.save_html(P.plot_function("x^2"), Path(d) / "a" / "x.html")
            self.assertTrue(p.exists())
            self.assertIn("<svg", p.read_text(encoding="utf-8"))
            if P.has_plotly():
                q = P.save_html(P.plotly_function("x^2"), Path(d) / "y.html")
                self.assertGreater(q.stat().st_size, 1000)
        with self.assertRaises(TypeError):
            P.save_html(object(), "x.html")


class TestAnim(unittest.TestCase):
    """Auftragsbeschreibung der Animationen (ohne Manim lauffähig)."""

    def test_modul_importierbar_und_kompiliert(self):
        import py_compile
        from anakit import anim
        py_compile.compile(anim.__file__, doraise=True)
        self.assertEqual(set(anim.SCENES), set(anim.SCENE_NAMES))

    def test_build_job(self):
        from anakit import anim
        job = anim.build_job("secant_to_tangent", "x^2", x0=1, quality="1080p60")
        self.assertEqual((job.quality, job.x0, job.xlim), ("high", 1.0, [-2.0, 4.0]))
        back = anim.AnimationJob.from_json(job.to_json())
        self.assertEqual(back, job)
        job = anim.build_job("curve_analysis", "x^2 if x<1 else x-1", dark=True)
        self.assertEqual(anim.AnimationJob.from_json(job.to_json()).f, "x^2 if x<1 else x-1")
        self.assertEqual(job.theme.background, "#000000")
        job = anim.build_job("derivative_graph", lambda t: t ** 3)
        self.assertIn("x**3", job.f)

    def test_build_job_fehler(self):
        from anakit import anim
        with self.assertRaises(ValueError):
            anim.build_job("unbekannt", "x^2")
        with self.assertRaises(ValueError):
            anim.build_job("secant_to_tangent", "x^2")              # x0 fehlt
        with self.assertRaises(ValueError):
            anim.build_job("secant_to_tangent", "1/x", x0=0)        # nicht definiert
        with self.assertRaises(ValueError):
            anim.build_job("curve_analysis", "x^2", quality="8k")
        with self.assertRaises(ValueError):
            anim.build_job("curve_analysis", "x^2", fmt="avi")

    def test_render_ohne_manim(self):
        from anakit import anim
        ok, _why = anim.manim_available()
        if ok:
            self.skipTest("Manim ist installiert")
        with self.assertRaises(RuntimeError) as cm:
            anim.render(anim.build_job("curve_analysis", "x^2"))
        self.assertIn("Manim", str(cm.exception))

    def test_hilfen(self):
        from anakit import anim
        self.assertEqual(anim._tick_values(-2, 4, 6), [-2, -1, 0, 1, 2, 3, 4])
        self.assertEqual(anim._clip_line(0, 0, 1, (-1, 1), (-5, 5)), (-1, 1))
        self.assertEqual(anim._clip_line(0, 0, 10, (-1, 1), (-5, 5)), (-0.5, 0.5))
        self.assertIsNone(anim._clip_line(0, 9, 0, (-1, 1), (-5, 5)))


if __name__ == "__main__":
    unittest.main(verbosity=1)
