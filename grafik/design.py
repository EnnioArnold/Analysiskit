"""
Gestaltungs-Token von anakit: EINE Stelle für Farben, Radien, Schriften, Abstände.

Die App (Qt), die Zeichnungen (Matplotlib) und die Animationen (Manim) lesen
alle von hier. Vorlage ist eine unabhängige Analyse des ElevenLabs-
Designsystems (getdesign.md, DESIGN.md): heller Off-White-Canvas, warme Tinte,
weisse Karten mit Haarlinie, Pillen-Buttons, Pastell nur als Atmosphäre.
Die Display-Schrift Waldenburg ist nicht frei; Ersatz ist EB Garamond Light.
Schriftdateien liegen nicht im Repo; fehlt eine Schrift, greift der Fallback.

Das Modul kennt weder Qt noch Matplotlib noch Manim.
"""

from __future__ import annotations

# --- Rohfarben ---------------------------------------------------------------
INK, INK_ACTIVE = "#292524", "#0c0a09"
BODY, MUTED, MUTED_SOFT = "#4e4e4e", "#777169", "#a8a29e"
HAIRLINE, HAIRLINE_SOFT, HAIRLINE_STRONG = "#e7e5e4", "#f0efed", "#d6d3d1"
CANVAS, CANVAS_SOFT, CARD = "#f5f5f5", "#fafafa", "#ffffff"
DARK_CANVAS, DARK_CARD = "#0c0a09", "#1c1917"
ERROR, SUCCESS = "#dc2626", "#16a34a"

#: Pastell-Orbs: nur Atmosphäre (Hintergrund, leere Zustände), nie Füllung von Controls.
ORBS = dict(mint="#a7e5d3", peach="#f4c5a8", lavender="#c8b8e0", sky="#a8c8e8", rose="#e8b8c4")
#: Dunklere Verwandte der Pastelltöne, die auf Weiss als Linie genug Kontrast haben.
DEEP = dict(mint="#2f9e80", peach="#d9895a", lavender="#7c6bb0", sky="#4f86c0", rose="#c0607a")
DEEP_DARK = dict(mint="#6fd3b6", peach="#f4a876", lavender="#b8a8e0", sky="#8fb8e8", rose="#e8a0b4")

# --- Radien, Abstände, Schriften ---------------------------------------------
RADIUS = dict(card=16, input=8, pill=9999, small=6)
SPACE = dict(xxs=4, xs=8, sm=12, base=16, md=20, lg=24, xl=32, xxl=48, section=96)
#: Erste vorhandene Familie gewinnt; zuletzt der Fallback.
FONT_UI = ["Inter", "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue",
           "Segoe UI Variable Text", "Segoe UI", "DejaVu Sans"]
FONT_DISPLAY = ["EB Garamond", "Cormorant Garamond", "Georgia", "Times New Roman", "serif"]
FONT_MONO = ["SF Mono", "Menlo", "Cascadia Mono", "Consolas", "DejaVu Sans Mono"]
UI_LETTER_SPACING = 100.6       # Prozent: Fliesstext leicht geweitet
MOTION_MS = (150, 250)          # kurze, sanfte Übergänge

# --- App (Qt): Schlüssel von anakit_qt.T --------------------------------------
APP_LIGHT = dict(
    bg=CANVAS, nav=CANVAS, card=CARD, text=INK_ACTIVE, sec=BODY, ter=MUTED,
    line=HAIRLINE_STRONG, hair=HAIRLINE, fill=HAIRLINE_SOFT, fill2=CANVAS_SOFT,
    seg=CARD, blue=INK, blue_hi=INK_ACTIVE, blue_soft=HAIRLINE_SOFT, on_blue="#ffffff",
    green="#15803d", green_soft="#e4f5ee", red=ERROR, red_soft="#fbeaee",
    amber="#a15c1c", amber_soft="#fbeadd", flash="#fdf0e0",
    entry=CARD, shadow=10, dark=False)

APP_DARK = dict(
    bg=DARK_CANVAS, nav=DARK_CANVAS, card=DARK_CARD, text="#fafaf9", sec="#d6d3d1",
    ter=MUTED_SOFT, line="#44403c", hair="#292524", fill="#292524", fill2="#14110f",
    seg="#44403c", blue="#fafaf9", blue_hi="#ffffff", blue_soft="#292524", on_blue=DARK_CANVAS,
    green="#4ade80", green_soft="#12261a", red="#f87171", red_soft="#2c1517",
    amber="#fdba74", amber_soft="#2e1d10", flash="#3a2a14",
    entry="#14110f", shadow=10, dark=True)

# --- Zeichnungen (Matplotlib) -------------------------------------------------
# Bedeutung -> Farbe. Tinte = f, Grau = Sekanten, Rot = Tangente, Lavendel = f',
# Grün/Rot = steigend/fallend bzw. positiv/negativ.
PLOT_LIGHT = dict(
    dark=False, bg=CARD, fg=INK_ACTIVE, muted=MUTED, grid=HAIRLINE,
    spine=HAIRLINE_STRONG, axis="#57534e",
    f=INK_ACTIVE, d1=DEEP["lavender"], d2=DEEP["sky"], sec="#78716c", sec2="#44403c",
    tan=ERROR, pt=INK_ACTIVE, pos=SUCCESS, neg=ERROR,
    hi=ERROR, lo=DEEP["sky"], sat=DEEP["peach"], wen=DEEP["lavender"])
PLOT_DARK = dict(
    dark=True, bg=DARK_CARD, fg="#fafaf9", muted=MUTED_SOFT, grid="#292524",
    spine="#44403c", axis="#a8a29e",
    f="#fafaf9", d1=DEEP_DARK["lavender"], d2=DEEP_DARK["sky"], sec="#a8a29e", sec2="#d6d3d1",
    tan="#f87171", pt="#fafaf9", pos="#4ade80", neg="#f87171",
    hi="#f87171", lo=DEEP_DARK["sky"], sat=DEEP_DARK["peach"], wen=DEEP_DARK["lavender"])
#: Reihenfolge für Kurvenscharen.
PLOT_COLORS = [INK_ACTIVE, DEEP["lavender"], DEEP["sky"], ERROR, DEEP["peach"], "#78716c"]

# --- Animationen (Manim): Felder von anim.Theme --------------------------------
ANIM_LIGHT = dict(
    background=CANVAS, text=INK_ACTIVE, muted=MUTED, grid=HAIRLINE_STRONG,
    accent=INK_ACTIVE, secant="#78716c", tangent=ERROR, derivative=DEEP["lavender"],
    good=SUCCESS, bad=ERROR, curved_left=DEEP["sky"], curved_right=DEEP["peach"],
    hi=ERROR, lo=DEEP["sky"], sat=DEEP["peach"], wen=DEEP["lavender"])
ANIM_DARK = dict(
    background=DARK_CANVAS, text="#fafaf9", muted=MUTED_SOFT, grid="#292524",
    accent="#fafaf9", secant="#a8a29e", tangent="#f87171", derivative=DEEP_DARK["lavender"],
    good="#4ade80", bad="#f87171", curved_left=DEEP_DARK["sky"], curved_right=DEEP_DARK["peach"],
    hi="#f87171", lo=DEEP_DARK["sky"], sat=DEEP_DARK["peach"], wen=DEEP_DARK["lavender"])


def pick_family(candidates, available, fallback):
    """Erste Familie aus `candidates`, die in `available` vorkommt (ohne Gross/Klein)."""
    have = {a.lower(): a for a in available}
    for name in candidates:
        if name.lower() in have:
            return have[name.lower()]
    return fallback
