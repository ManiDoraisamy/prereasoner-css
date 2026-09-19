"""
Correctness tests for the walker + colour-anchor detection pipeline.

The single most important assertion in this file is NO_HSL_ANCHORS: every
emergence claim the project makes rests on hsl()/hsla() never contributing
a training anchor. If that test fails, nothing downstream is publishable.

Run:  python tests/test_detection.py     (plain asserts, no pytest needed)
"""

# --- repo path shim ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
# --- end shim ---

from color_detection import detect_colors
from css_walker import walk_css

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "ok " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def anchors_of(css):
    anchors, hsls = detect_colors(walk_css(css))
    return anchors, hsls


def by_source(anchors):
    return {a.source: a for a in anchors}


# ---------------------------------------------------------------------------
print("hex per-channel anchoring")
# ---------------------------------------------------------------------------
anchors, _ = anchors_of("a { color: #a3f9c2; }")
check("hex yields exactly 3 anchors", len(anchors) == 3, f"got {len(anchors)}")
if len(anchors) == 3:
    ar, ag, ab = anchors
    check("hex channels 0/1/2", (ar.channel, ag.channel, ab.channel) == (0, 1, 2))
    check("hex R target", abs(ar.r - 0xA3 / 255) < 1e-9)
    check("hex G target", abs(ag.g - 0xF9 / 255) < 1e-9)
    check("hex B target", abs(ab.b - 0xC2 / 255) < 1e-9)
    check("hex stores FULL rgb in every record",
          abs(ar.g - 0xF9 / 255) < 1e-9 and abs(ab.r - 0xA3 / 255) < 1e-9)
    check("hex anchors at consecutive pair positions",
          (ag.position - ar.position, ab.position - ag.position) == (1, 1))

# ---------------------------------------------------------------------------
print("rgb() per-argument anchoring")
# ---------------------------------------------------------------------------
anchors, _ = anchors_of("a { color: rgb(255, 128, 0); }")
check("rgb yields 3 per-arg anchors", len(anchors) == 3, f"got {len(anchors)}")
if len(anchors) == 3:
    srcs = [a.source for a in anchors]
    check("rgb sources", srcs == ["rgb_r", "rgb_g", "rgb_b"], str(srcs))
    check("rgb channels", [a.channel for a in anchors] == [0, 1, 2])
    check("rgb targets", abs(anchors[0].r - 1.0) < 1e-9
          and abs(anchors[1].g - 128 / 255) < 1e-9
          and abs(anchors[2].b - 0.0) < 1e-9)
    check("rgb anchors at three distinct positions",
          len({a.position for a in anchors}) == 3)
    check("rgb full triple stored on every record",
          abs(anchors[2].r - 1.0) < 1e-9 and abs(anchors[0].b - 0.0) < 1e-9)

anchors, _ = anchors_of("a { color: rgb(100%, 50%, 0%); }")
check("rgb percent args yield 3 anchors", len(anchors) == 3, f"got {len(anchors)}")
if len(anchors) == 3:
    check("rgb percent scaled to 0-1",
          abs(anchors[0].r - 1.0) < 1e-9 and abs(anchors[1].g - 0.5) < 2e-3)

anchors, _ = anchors_of("a { color: rgb(255 128 0 / 50%); }")
check("rgb space syntax yields 3 anchors", len(anchors) == 3, f"got {len(anchors)}")
if len(anchors) == 3:
    check("rgb space-syntax targets",
          abs(anchors[0].r - 1.0) < 1e-9 and abs(anchors[1].g - 128 / 255) < 1e-9)

anchors, _ = anchors_of("a { color: rgba(10, 20, 30, 0.5); }")
check("rgba yields 3 anchors (alpha ignored)", len(anchors) == 3, f"got {len(anchors)}")
if len(anchors) == 3:
    check("rgba does not anchor the alpha token",
          all(a.source in ("rgba_r", "rgba_g", "rgba_b") for a in anchors))

anchors, _ = anchors_of("a { color: rgb(var(--r), 0, 0); }")
check("rgb with nested var() is NOT per-arg anchored",
      all(a.channel is not None or a.source in ("rgb", "rgba") for a in anchors)
      and not any(a.source.endswith(("_r", "_g", "_b")) and "rgb" in a.source
                  for a in anchors),
      f"anchors={[(a.source, a.channel) for a in anchors]}")

# ---------------------------------------------------------------------------
print("named colours")
# ---------------------------------------------------------------------------
anchors, _ = anchors_of("a { color: teal; }")
check("named colour anchored once", len(anchors) == 1)
if anchors:
    check("named colour all-three channel", anchors[0].channel is None)
    check("teal target", abs(anchors[0].g - 128 / 255) < 1e-9
          and abs(anchors[0].b - 128 / 255) < 1e-9 and anchors[0].r == 0.0)

# ---------------------------------------------------------------------------
print("context filtering (the P0 selector bug of the original project)")
# ---------------------------------------------------------------------------
anchors, _ = anchors_of(".red { margin: 0; }")
check("class selector '.red' NOT anchored", len(anchors) == 0,
      f"got {[(a.source, a.position) for a in anchors]}")
anchors, _ = anchors_of("#facade { margin: 0; }")
check("id selector '#facade' NOT anchored", len(anchors) == 0)

# ---------------------------------------------------------------------------
print("NO_HSL_ANCHORS — the emergence guarantee")
# ---------------------------------------------------------------------------
css = """
a { color: hsl(217, 50%, 50%); }
b { color: hsla(340deg, 30%, 40%, 0.5); }
c { background: linear-gradient(hsl(10, 20%, 30%), hsl(200, 80%, 70%)); }
"""
anchors, hsls = anchors_of(css)
check("hsl/hsla produce ZERO anchors", len(anchors) == 0,
      f"got {[(a.source, a.position) for a in anchors]}")
check("hsl values ARE parsed for analysis", len(hsls) == 4, f"got {len(hsls)}")
if len(hsls) >= 1:
    check("hsl H/S/L parsed", hsls[0].h == 217.0 and hsls[0].s == 50.0
          and hsls[0].l == 50.0)
if len(hsls) >= 2:
    check("hsla deg unit parsed", hsls[1].h == 340.0)

# Mixed file: rgb anchored, hsl not, positions must not collide
css = "a { color: rgb(1, 2, 3); border-color: hsl(100, 10%, 20%); }"
anchors, hsls = anchors_of(css)
check("mixed: rgb anchored, hsl not",
      len(anchors) == 3 and len(hsls) == 1
      and all(a.source.startswith("rgb_") for a in anchors))
if anchors and hsls:
    stream = walk_css(css)
    hsl_span_texts = {"hsl", "100", "10", "20"}
    anchored_texts = {stream[a.position][0] for a in anchors}
    check("no anchor lands on an hsl token",
          anchored_texts.isdisjoint(hsl_span_texts), str(anchored_texts))

# ---------------------------------------------------------------------------
print("hue unit handling")
# ---------------------------------------------------------------------------
_, hsls = anchors_of("a{color:hsl(0.5turn, 10%, 20%)}")
check("turn unit -> degrees", hsls and abs(hsls[0].h - 180.0) < 1e-6)
_, hsls = anchors_of("a{color:hsl(200grad, 10%, 20%)}")
check("grad unit -> degrees", hsls and abs(hsls[0].h - 180.0) < 1e-6)

# ---------------------------------------------------------------------------
n = len(PASS) + len(FAIL)
print(f"\n{len(PASS)}/{n} passed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    raise SystemExit(1)
