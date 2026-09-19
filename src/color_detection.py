"""
Color detection on CSS token streams.

For each color expression in a token stream, emit a ColorAnchor that names
the single sequence position where R, G, B values should be anchored during
training, plus the normalized (r, g, b) target in [0, 1].

CRITICAL: anchors only fire when tokens appear in VALUE context. A class
selector ".red {}" must NOT be anchored to RGB(1,0,0) just because the
identifier "red" matches a named color. The walker tags each token with
its context (CTX_SELECTOR, CTX_VALUE, etc.); we filter on that.

Anchor position policy:
  - Hex color "#fff", "#a3f9c2" (in value context): the hex token itself.
  - Named color "red" (in value context): the ident token.
  - rgb()/rgba() function: the closing ')' token, after the model has read
    the components via causal attention.
  - hsl()/hsla(): DETECTED but excluded from anchoring. Hue accepts bare
    numbers, percentages, AND <angle> dimensions (deg, rad, turn, grad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from css_walker import CTX_VALUE


# ---------------------------------------------------------------------------
# Named CSS colors (147 standard)
# ---------------------------------------------------------------------------

NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "aliceblue": (240, 248, 255), "antiquewhite": (250, 235, 215),
    "aqua": (0, 255, 255), "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255), "beige": (245, 245, 220),
    "bisque": (255, 228, 196), "black": (0, 0, 0),
    "blanchedalmond": (255, 235, 205), "blue": (0, 0, 255),
    "blueviolet": (138, 43, 226), "brown": (165, 42, 42),
    "burlywood": (222, 184, 135), "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0), "chocolate": (210, 105, 30),
    "coral": (255, 127, 80), "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220), "crimson": (220, 20, 60),
    "cyan": (0, 255, 255), "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139), "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169),
    "darkgreen": (0, 100, 0), "darkkhaki": (189, 183, 107),
    "darkmagenta": (139, 0, 139), "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0), "darkorchid": (153, 50, 204),
    "darkred": (139, 0, 0), "darksalmon": (233, 150, 122),
    "darkseagreen": (143, 188, 143), "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79), "darkslategrey": (47, 79, 79),
    "darkturquoise": (0, 206, 209), "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147), "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105), "dimgrey": (105, 105, 105),
    "dodgerblue": (30, 144, 255), "firebrick": (178, 34, 34),
    "floralwhite": (255, 250, 240), "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255), "gainsboro": (220, 220, 220),
    "ghostwhite": (248, 248, 255), "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "green": (0, 128, 0),
    "greenyellow": (173, 255, 47), "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180), "indianred": (205, 92, 92),
    "indigo": (75, 0, 130), "ivory": (255, 255, 240),
    "khaki": (240, 230, 140), "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245), "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205), "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128), "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
    "lightgreen": (144, 238, 144), "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122), "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250), "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153), "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224), "lime": (0, 255, 0),
    "limegreen": (50, 205, 50), "linen": (250, 240, 230),
    "magenta": (255, 0, 255), "maroon": (128, 0, 0),
    "mediumaquamarine": (102, 205, 170), "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211), "mediumpurple": (147, 112, 219),
    "mediumseagreen": (60, 179, 113), "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154), "mediumturquoise": (72, 209, 204),
    "mediumvioletred": (199, 21, 133), "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250), "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181), "navajowhite": (255, 222, 173),
    "navy": (0, 0, 128), "oldlace": (253, 245, 230),
    "olive": (128, 128, 0), "olivedrab": (107, 142, 35),
    "orange": (255, 165, 0), "orangered": (255, 69, 0),
    "orchid": (218, 112, 214), "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152), "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147), "papayawhip": (255, 239, 213),
    "peachpuff": (255, 218, 185), "peru": (205, 133, 63),
    "pink": (255, 192, 203), "plum": (221, 160, 221),
    "powderblue": (176, 224, 230), "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153), "red": (255, 0, 0),
    "rosybrown": (188, 143, 143), "royalblue": (65, 105, 225),
    "saddlebrown": (139, 69, 19), "salmon": (250, 128, 114),
    "sandybrown": (244, 164, 96), "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238), "sienna": (160, 82, 45),
    "silver": (192, 192, 192), "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205), "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144), "snow": (255, 250, 250),
    "springgreen": (0, 255, 127), "steelblue": (70, 130, 180),
    "tan": (210, 180, 140), "teal": (0, 128, 128),
    "thistle": (216, 191, 216), "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208), "violet": (238, 130, 238),
    "wheat": (245, 222, 179), "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245), "yellow": (255, 255, 0),
    "yellowgreen": (154, 205, 50),
}

HEX_CHARS = set("0123456789abcdefABCDEF")


@dataclass
class ColorAnchor:
    position: int
    r: float
    g: float
    b: float
    source: str            # "hex_r/g/b" | "named" | "rgb_r/g/b" | "rgba_r/g/b"
                           # | "rgb" | "rgba" (close-paren fallback)
    channel: int | None = None    # 0/1/2 to anchor a single dim; None for all three


@dataclass
class HslDetection:
    position: int          # closing ')' of the hsl()/hsla() function
    source: str            # "hsl" | "hsla"
    h: float | None        # raw H value in degrees (0-360), None if unparseable
    s: float | None        # raw S percentage (0-100)
    l: float | None        # raw L percentage (0-100)


# ---------------------------------------------------------------------------
# Token helpers (accepts 2- or 3-tuple tokens; uses first two elements)
# ---------------------------------------------------------------------------

def _tok_text(tok) -> str:
    return tok[0]


def _tok_type(tok) -> str:
    return tok[1]


def _tok_context(tok) -> str:
    """Return the context tag if present (3-tuple); fall back to CTX_VALUE
    for legacy 2-tuples so callers using old streams still detect colors."""
    return tok[2] if len(tok) > 2 else CTX_VALUE


# ---------------------------------------------------------------------------
# Hex parsing
# ---------------------------------------------------------------------------

def parse_hex(token: str) -> tuple[int, int, int] | None:
    if not token.startswith("#"):
        return None
    body = token[1:]
    if not body or any(c not in HEX_CHARS for c in body):
        return None
    n = len(body)
    if n == 3:
        r, g, b = body[0] * 2, body[1] * 2, body[2] * 2
    elif n == 4:
        r, g, b = body[0] * 2, body[1] * 2, body[2] * 2  # alpha ignored
    elif n == 6:
        r, g, b = body[0:2], body[2:4], body[4:6]
    elif n == 8:
        r, g, b = body[0:2], body[2:4], body[4:6]  # alpha ignored
    else:
        return None
    try:
        return int(r, 16), int(g, 16), int(b, 16)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# rgb()/hsl() argument parsing
# ---------------------------------------------------------------------------

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _parse_rgb_component(text: str) -> float | None:
    """Parse an rgb() component. Accepts '255', '100%', '1.0'. Returns 0-255."""
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0 * 255.0
        except ValueError:
            return None
    if _is_number(text):
        return float(text)
    return None


def _parse_hue(text: str) -> float | None:
    """Parse an hsl() hue. Accepts bare number, percentage, or <angle>
    dimension (deg, rad, turn, grad). Returns degrees in [0, 360)."""
    # Strip a trailing unit if present (DimensionToken text like "240deg")
    # NOTE: "grad" must be checked BEFORE "rad" — "200grad".endswith("rad")
    # is True, and the rad branch then fails to parse "200g", silently
    # dropping every grad-unit hue. (Bug inherited from the original
    # project; caught by tests/test_detection.py.)
    if text.endswith("deg"):
        try:
            v = float(text[:-3])
        except ValueError:
            return None
    elif text.endswith("grad"):
        try:
            v = float(text[:-4]) * 0.9  # 400 grad == 360 deg
        except ValueError:
            return None
    elif text.endswith("rad"):
        try:
            v = float(text[:-3]) * 180.0 / math.pi
        except ValueError:
            return None
    elif text.endswith("turn"):
        try:
            v = float(text[:-4]) * 360.0
        except ValueError:
            return None
    elif text.endswith("%"):
        try:
            v = float(text[:-1]) * 3.6  # 100% == 360 deg in some specs
        except ValueError:
            return None
    elif _is_number(text):
        v = float(text)
    else:
        return None
    return v % 360.0


def _parse_sl(text: str) -> float | None:
    """Parse hsl() saturation/lightness. Accepts '50%' or bare number."""
    if text.endswith("%"):
        try:
            return float(text[:-1])
        except ValueError:
            return None
    if _is_number(text):
        return float(text)
    return None


def _find_function_close(tokens: list, start: int) -> int | None:
    """Given start = index of a 'func(' token, return index of matching ')'.
    Walks balancing parens. Returns None if unbalanced."""
    depth = 1
    for i in range(start + 1, len(tokens)):
        text = _tok_text(tokens[i])
        if i > start and (text == "(" or text.endswith("(")):
            depth += 1
        elif text == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _collect_function_args(tokens: list, start: int, end: int) -> list[str]:
    """Collect comma-separated arg groups between `(` (at `start`) and `)` (at `end`).

    Within each comma-bounded group, concatenates contiguous non-separator
    token texts so split components fuse back into one logical arg:
      '240' + 'deg'        -> '240deg'
      '50'  + '%'          -> '50%'
      '255' (just a number) -> '255'
    Stops at the first `/` (modern alpha separator). Skips paren literals.
    """
    args: list[str] = []
    parts: list[str] = []
    for i in range(start + 1, end):
        text = _tok_text(tokens[i])
        if text == ",":
            if parts:
                args.append("".join(parts))
                parts = []
            continue
        if text == "/":
            break
        if text in ("(", ")"):
            continue
        parts.append(text)
    if parts:
        args.append("".join(parts))
    return args


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_hex_pair(s: str) -> bool:
    return len(s) == 2 and all(c in HEX_CHARS for c in s)


def _rgb_arg_positions(
    tokens: list, open_idx: int, close_idx: int,
) -> list[tuple[int, float]] | None:
    """Locate the three rgb()/rgba() channel arguments as token positions.

    Returns [(stream_pos, value_0_1), ...] for the R, G, B argument number
    tokens, or None when the call can't be anchored per-argument:
      - a nested function (var(), calc(), ...) makes argument alignment
        ambiguous — skip the whole call rather than risk anchoring the
        wrong token;
      - fewer than three numeric arguments found.

    Handles both comma syntax `rgb(255, 0, 0)` and modern space syntax
    `rgb(255 0 0 / 50%)`: we simply take the first number token of each
    argument group. Percent components ('100' followed by a '%' token)
    are scaled 0-100 -> 0-255 before normalizing. A fourth numeric token
    (alpha in rgba) is ignored — there is no alpha dim in this scheme.
    """
    positions: list[tuple[int, float]] = []
    for i in range(open_idx + 1, close_idx):
        text = _tok_text(tokens[i])
        ttype = _tok_type(tokens[i])
        if ttype == "func":
            return None          # nested var()/calc() — bail out
        if text == "/":
            break                # modern alpha separator — channels are done
        # With whitespace stripped by the walker and nested functions
        # excluded, every number token inside rgb()/rgba() is a channel
        # argument — true for both comma and space syntax. Multi-number
        # arguments can only arise inside nested functions, which bail
        # above.
        if ttype == "number" and _is_number(text):
            value = float(text)
            # Percent form: number token followed by a '%' literal
            if i + 1 < close_idx and _tok_text(tokens[i + 1]) == "%":
                value = value / 100.0 * 255.0
            value = max(0.0, min(255.0, value)) / 255.0
            positions.append((i, value))
            if len(positions) == 3:
                return positions
    return None


def detect_colors(
    tokens: list,
) -> tuple[list[ColorAnchor], list[HslDetection]]:
    """Scan a token stream, return (color_anchors, hsl_detections).

    Anchors are only emitted when tokens carry CTX_VALUE context (selector
    tokens are filtered out so ".red {}" is NOT mistakenly anchored).

    Hex colors are anchored per-channel: a `#` + three digit-pair sequence
    produces three ColorAnchor records, one per (R, G, B) channel, each
    pinned to the position of its own digit-pair token. This gives the
    model a structural gradient signal — dim 0 should encode the first
    pair, dim 1 the second, dim 2 the third — instead of one combined
    anchor at a single point.

    rgb()/rgba() calls are anchored per-argument — dim 0 at the R argument
    token, dim 1 at G, dim 2 at B — mirroring the hex digit-pair scheme.
    Calls with nested functions fall back to a single all-three anchor at
    the closing `)` when the values still parse, else are skipped.

    hsl()/hsla() are NEVER anchored. Their parsed values are returned in
    the separate `hsls` list for post-hoc analysis only; nothing from that
    list reaches the training loss. Every emergence claim in this project
    rests on that separation — do not merge these lists.
    """
    anchors: list[ColorAnchor] = []
    hsls: list[HslDetection] = []

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        text = _tok_text(tok)
        ttype = _tok_type(tok)
        ctx = _tok_context(tok)

        if ctx != CTX_VALUE:
            i += 1
            continue

        # Hex colors: '#' literal followed by three digit-pair hash tokens
        if ttype == "literal" and text == "#":
            if (i + 3 < n
                and _tok_type(tokens[i + 1]) == "hash"
                and _tok_type(tokens[i + 2]) == "hash"
                and _tok_type(tokens[i + 3]) == "hash"
                and _is_hex_pair(_tok_text(tokens[i + 1]))
                and _is_hex_pair(_tok_text(tokens[i + 2]))
                and _is_hex_pair(_tok_text(tokens[i + 3]))):
                r = int(_tok_text(tokens[i + 1]), 16) / 255.0
                g = int(_tok_text(tokens[i + 2]), 16) / 255.0
                b = int(_tok_text(tokens[i + 3]), 16) / 255.0
                # Store the FULL colour in every record; `channel` alone
                # decides which dim the loss supervises. (Previously the
                # unsupervised slots held 0.0 placeholders — inert under the
                # loss mask, but they made anchors.bin lie about the colour.)
                anchors.append(ColorAnchor(i + 1, r, g, b, "hex_r", channel=0))
                anchors.append(ColorAnchor(i + 2, r, g, b, "hex_g", channel=1))
                anchors.append(ColorAnchor(i + 3, r, g, b, "hex_b", channel=2))
                i += 4
                continue
            i += 1
            continue

        # Legacy fused hex tokens (non-color hashes like `#header`) — skip
        # for anchoring; they're not colors. The old detection that read
        # the whole hex from a hash token is no longer needed.
        if ttype == "hash":
            i += 1
            continue

        # Named colors
        if ttype == "ident" and text.lower() in NAMED_COLORS:
            r, g, b = NAMED_COLORS[text.lower()]
            anchors.append(ColorAnchor(
                i, r / 255.0, g / 255.0, b / 255.0, "named", channel=None,
            ))
            i += 1
            continue

        # Functional notations: walker now emits func name + `(` as two tokens
        if ttype == "func" and text.lower() in ("rgb", "rgba", "hsl", "hsla"):
            name = text.lower()
            # Need the very next token to be the `(` literal
            if i + 1 >= n or _tok_text(tokens[i + 1]) != "(":
                i += 1
                continue
            paren_idx = i + 1
            close = _find_function_close(tokens, paren_idx)
            if close is None:
                i += 1
                continue
            args = _collect_function_args(tokens, paren_idx, close)

            if name in ("rgb", "rgba"):
                # Per-argument anchoring: dim 0 supervised at the R argument
                # token, dim 1 at G, dim 2 at B — the same structural scheme
                # as hex digit-pairs, and the literal reading of "constrain
                # the first three dimensions to represent the arguments of
                # rgb()". Falls back to the legacy all-three anchor at the
                # closing ')' when per-argument extraction fails (nested
                # functions, exotic forms) but the values still parse.
                arg_pos = _rgb_arg_positions(tokens, paren_idx, close)
                if arg_pos is not None:
                    (rp, r), (gp, g), (bp, b) = arg_pos
                    anchors.append(ColorAnchor(rp, r, g, b, f"{name}_r", channel=0))
                    anchors.append(ColorAnchor(gp, r, g, b, f"{name}_g", channel=1))
                    anchors.append(ColorAnchor(bp, r, g, b, f"{name}_b", channel=2))
                elif len(args) >= 3:
                    parsed_args = [_parse_rgb_component(a) for a in args[:3]]
                    if all(p is not None for p in parsed_args):
                        r, g, b = parsed_args
                        r = max(0.0, min(255.0, r)) / 255.0
                        g = max(0.0, min(255.0, g)) / 255.0
                        b = max(0.0, min(255.0, b)) / 255.0
                        anchors.append(ColorAnchor(
                            close, r, g, b, name, channel=None,
                        ))

            elif name in ("hsl", "hsla") and len(args) >= 3:
                h = _parse_hue(args[0])
                s = _parse_sl(args[1])
                l = _parse_sl(args[2])
                hsls.append(HslDetection(close, name, h, s, l))

            i = close + 1
            continue

        i += 1

    return anchors, hsls
