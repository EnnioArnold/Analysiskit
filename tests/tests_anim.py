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
    j = build_job(s, "x^2", x0=1)
    assert j.scene == s and j.f and j.quality == "low" and j.fmt == "mp4"
    assert AnimationJob.from_json(j.to_json()) == j, s
assert build_job("limit_approach", "(x^2-1)/(x-1)", x0=1).x0 == 1.0
assert raises(ValueError, build_job, "nix", "x^2", x0=1)
for s in ("limit_approach", "secant_to_tangent"):
    assert raises(ValueError, build_job, s, "x^2"), s

# (b) Swiss-Theme
t, d = Theme(), Theme.dark()
assert (t.background, d.background) == ("#ffffff", "#000000")
assert t.tangent.startswith("#e") and d.tangent.startswith("#ff")
assert build_job("derivative_graph", "x^2", dark=True).theme == d

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
else:
    print("Rendering übersprungen:", why)
print("alle Tests bestanden")
