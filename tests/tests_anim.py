#!/usr/bin/env python3
"""Tests für anakit.anim: python3 tests/tests_anim.py  (Rendering nur, wenn die Engine da ist)"""
import importlib.util, os, sys
D = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Ordner heißt evtl. nicht 'anakit' -> als Paket 'anakit' laden
spec = importlib.util.spec_from_file_location("anakit", D + "/__init__.py", submodule_search_locations=[D])
m = importlib.util.module_from_spec(spec); sys.modules["anakit"] = m; spec.loader.exec_module(m)
from anakit import anim
from anakit.anim import AnimationJob, Theme, build_job

def raises(exc, fn, *a, **k):
    try: fn(*a, **k)
    except exc: return True
    return False

# (a) Aufträge
for s in anim.SCENES:
    j = build_job(s, "x^2-2" if s == "bisection" else "x^2", x0=1, a=1, b=2)
    assert j.scene == s and j.f and j.quality == "low" and j.fmt == "mp4"
    assert AnimationJob.from_json(j.to_json()) == j, s
assert build_job("limit_approach", "(x^2-1)/(x-1)", x0=1).x0 == 1.0
assert raises(ValueError, build_job, "nix", "x^2", x0=1)
jb = build_job("bisection", "x^2-2", a=1, b=2)
assert (jb.a, jb.b) == (1.0, 2.0) and jb.xlim[0] < 1 and jb.xlim[1] > 2
assert raises(ValueError, build_job, "bisection", "x^2-2")                # a, b fehlen
assert raises(ValueError, build_job, "bisection", "x^2+1", a=0, b=1)      # kein Vorzeichenwechsel
assert raises(ValueError, build_job, "bisection", "x^2-2", a=2, b=1)      # a >= b
for s in ("limit_approach", "secant_to_tangent"):
    assert raises(ValueError, build_job, s, "x^2"), s

assert build_job("newton", "x^2-2", x0=1, n=3).n == 3 and raises(ValueError, build_job, "newton", "x^2-2", x0=1, n=0)
j = build_job("taylor", "exp(x)", x0=0, n=3)
assert j.n == 3 and j.x0 == 0.0 and j.xlim == [-3.0, 3.0]
assert raises(ValueError, build_job, "taylor", "x^2")
assert raises(ValueError, build_job, "taylor", "x^2", x0=0, n=11)

# (b) Swiss-Theme
t, d = Theme(), Theme.dark()
assert (t.background, d.background) == ("#ffffff", "#000000")
assert t.tangent.startswith("#e") and d.tangent.startswith("#ff")
assert build_job("derivative_graph", "x^2", dark=True).theme == d
assert build_job("derivative_graph", "x^2", style="3b1b").theme == Theme.b3b1()
assert raises(ValueError, build_job, "derivative_graph", "x^2", style="nix")

# (c) Engine-Wahl
old = os.environ.get("ANAKIT_ENGINE")
try:
    for e in ("manim", "manimgx"):
        os.environ["ANAKIT_ENGINE"] = e; assert anim.engine() == e
    os.environ.pop("ANAKIT_ENGINE")
    assert anim.engine() in ("manim", "manimgx")
finally:
    if old is None: os.environ.pop("ANAKIT_ENGINE", None)
    else: os.environ["ANAKIT_ENGINE"] = old

# (d) echtes Rendering
ok, why = anim.manim_available()
if os.environ.get("ANAKIT_ENGINE") and os.environ["ANAKIT_ENGINE"] != anim.engine(): ok, why = False, "ANAKIT_ENGINE gesetzt"
if ok:
    p = anim.render(build_job("limit_approach", "(x^2-1)/(x-1)", x0=1, quality="low"))
    assert os.path.exists(p) and str(p).endswith(".mp4") and os.path.getsize(p) > 0, p
    print("Rendering ok:", anim.engine(), p)
    p = anim.render(build_job("bisection", "x^2-2", a=1, b=2, quality="low"))
    assert os.path.exists(p) and os.path.getsize(p) > 0, p
    assert os.path.exists(p) and os.path.getsize(p) > 0, p
    p = anim.render(build_job("newton", "x^2-2", x0=3, n=3, quality="low"))
    assert os.path.exists(p) and os.path.getsize(p) > 0, p
    p = anim.render(build_job("taylor", "exp(x)", x0=0, n=2, quality="low"))
    assert os.path.exists(p) and os.path.getsize(p) > 0, p
else:
    print("Rendering übersprungen:", why)
print("alle Tests bestanden")
