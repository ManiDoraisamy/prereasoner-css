"""
Bounded-vocabulary token lists for tokenizer fat-head force-inclusion.

These are CSS constructs from the spec (finite enumerations) that should
always be atomic single tokens regardless of their frequency in the corpus.
Used by stage2_build_vocab.py to inject these into the fat head, ensuring
the tokenizer never BPE-splits them mid-symbol.

Why this matters: BPE-splitting destroys the structural identity of tokens
like color values (#a3f9c2 -> nonsense merges), unit suffixes (px split
into 'p' + 'x'), or function names (`rgb(` split into BPE pieces). The
model learns the semantic correlate of an atomic token but cannot recover
meaning from BPE fragments. Force-inclusion guarantees atomicity.
"""

from __future__ import annotations

from color_detection import NAMED_COLORS

# ---------------------------------------------------------------------------
# Named colors (147 standard CSS color keywords)
# ---------------------------------------------------------------------------

NAMED_COLORS_LIST = sorted(NAMED_COLORS.keys())


# ---------------------------------------------------------------------------
# Hex digit pairs (00 .. ff) for the new structural hex tokenization
# ---------------------------------------------------------------------------

HEX_DIGIT_PAIRS = [f"{i:02x}" for i in range(256)]


# ---------------------------------------------------------------------------
# Integers 0 .. 255 (covers rgb() channels, common dim values, pcts)
# ---------------------------------------------------------------------------

INTEGERS_0_255 = [str(i) for i in range(256)]


# ---------------------------------------------------------------------------
# Common floats — alpha values, opacity, scale factors, easing constants
# ---------------------------------------------------------------------------

COMMON_FLOATS = (
    [f"0.{i}" for i in range(10)]      # 0.0 .. 0.9
    + ["1.0"]
    + ["0.25", "0.5", "0.75", "0.05", "0.15"]
    + ["1.5", "2.0", "2.5", "3.0", "5.0", "10.0"]
)


# ---------------------------------------------------------------------------
# Standard CSS property names (subset covering the common 80%+ of real CSS)
# ---------------------------------------------------------------------------

CSS_PROPERTIES = [
    # Layout / box
    "display", "position", "top", "right", "bottom", "left", "z-index",
    "float", "clear", "overflow", "overflow-x", "overflow-y", "visibility",
    "box-sizing", "object-fit", "object-position", "isolation",
    # Width / height
    "width", "height", "min-width", "min-height", "max-width", "max-height",
    "aspect-ratio", "block-size", "inline-size",
    # Margin / padding
    "margin", "margin-top", "margin-right", "margin-bottom", "margin-left",
    "margin-block", "margin-inline", "margin-block-start", "margin-block-end",
    "margin-inline-start", "margin-inline-end",
    "padding", "padding-top", "padding-right", "padding-bottom", "padding-left",
    "padding-block", "padding-inline", "padding-block-start", "padding-block-end",
    "padding-inline-start", "padding-inline-end",
    # Border
    "border", "border-width", "border-style", "border-color",
    "border-top", "border-right", "border-bottom", "border-left",
    "border-top-width", "border-right-width", "border-bottom-width", "border-left-width",
    "border-top-style", "border-right-style", "border-bottom-style", "border-left-style",
    "border-top-color", "border-right-color", "border-bottom-color", "border-left-color",
    "border-radius", "border-top-left-radius", "border-top-right-radius",
    "border-bottom-left-radius", "border-bottom-right-radius",
    "border-image", "border-image-source", "border-image-slice", "border-image-width",
    "border-image-outset", "border-image-repeat",
    "border-collapse", "border-spacing", "outline", "outline-width", "outline-style",
    "outline-color", "outline-offset",
    # Background
    "background", "background-color", "background-image", "background-repeat",
    "background-position", "background-position-x", "background-position-y",
    "background-size", "background-attachment", "background-origin", "background-clip",
    "background-blend-mode", "mix-blend-mode",
    # Typography
    "color", "font", "font-family", "font-size", "font-weight", "font-style",
    "font-variant", "font-stretch", "font-feature-settings", "font-variation-settings",
    "font-display", "font-kerning", "font-language-override", "font-optical-sizing",
    "font-size-adjust", "font-smooth", "-webkit-font-smoothing", "-moz-osx-font-smoothing",
    "line-height", "letter-spacing", "word-spacing", "text-align", "text-align-last",
    "text-decoration", "text-decoration-line", "text-decoration-color",
    "text-decoration-style", "text-decoration-thickness", "text-underline-offset",
    "text-transform", "text-indent", "text-shadow", "text-overflow", "text-wrap",
    "text-rendering", "text-orientation", "white-space", "word-break", "word-wrap",
    "overflow-wrap", "hyphens", "tab-size", "direction", "writing-mode", "unicode-bidi",
    "vertical-align", "list-style", "list-style-type", "list-style-position",
    "list-style-image", "quotes",
    # Flexbox
    "flex", "flex-direction", "flex-wrap", "flex-flow", "flex-grow", "flex-shrink",
    "flex-basis", "justify-content", "justify-items", "justify-self",
    "align-items", "align-self", "align-content", "place-content", "place-items",
    "place-self", "order", "gap", "row-gap", "column-gap",
    # Grid
    "grid", "grid-template", "grid-template-columns", "grid-template-rows",
    "grid-template-areas", "grid-auto-columns", "grid-auto-rows", "grid-auto-flow",
    "grid-column", "grid-row", "grid-area", "grid-column-start", "grid-column-end",
    "grid-row-start", "grid-row-end", "grid-gap", "grid-column-gap", "grid-row-gap",
    # Visual effects
    "opacity", "box-shadow", "filter", "backdrop-filter", "clip", "clip-path",
    "mask", "mask-image", "mask-mode", "mask-position", "mask-repeat", "mask-size",
    "mask-origin", "mask-clip", "mask-composite", "transform", "transform-origin",
    "transform-style", "transform-box", "perspective", "perspective-origin",
    "backface-visibility", "appearance", "-webkit-appearance", "-moz-appearance",
    # Transitions / animation
    "transition", "transition-property", "transition-duration",
    "transition-timing-function", "transition-delay",
    "animation", "animation-name", "animation-duration", "animation-timing-function",
    "animation-delay", "animation-iteration-count", "animation-direction",
    "animation-fill-mode", "animation-play-state", "will-change",
    # Tables
    "table-layout", "caption-side", "empty-cells",
    # User interaction
    "cursor", "pointer-events", "user-select", "-webkit-user-select",
    "-moz-user-select", "-ms-user-select", "touch-action", "resize", "scroll-behavior",
    "overscroll-behavior", "overscroll-behavior-x", "overscroll-behavior-y",
    "scroll-margin", "scroll-padding", "scroll-snap-type", "scroll-snap-align",
    "scroll-snap-stop",
    # Counters / generated content
    "content", "counter-reset", "counter-increment", "counter-set",
    # Columns
    "columns", "column-count", "column-width", "column-rule", "column-rule-width",
    "column-rule-style", "column-rule-color", "column-span", "column-fill",
    "column-gap",
    # SVG-specific
    "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "stroke-dashoffset",
    "stroke-miterlimit", "vector-effect", "shape-rendering", "image-rendering",
    # Logical / writing
    "inset", "inset-block", "inset-inline", "inset-block-start", "inset-block-end",
    "inset-inline-start", "inset-inline-end",
    # Page / print
    "page-break-before", "page-break-after", "page-break-inside",
    "break-before", "break-after", "break-inside", "orphans", "widows",
    # Misc
    "all", "contain", "content-visibility", "accent-color", "color-scheme",
    "caret-color", "tab-size", "speak", "src",
]


# ---------------------------------------------------------------------------
# At-keywords (always emitted by walker with leading @)
# ---------------------------------------------------------------------------

CSS_AT_KEYWORDS = [
    "@media", "@import", "@charset", "@namespace", "@page", "@supports",
    "@document", "@font-face", "@keyframes", "@-webkit-keyframes",
    "@-moz-keyframes", "@-o-keyframes", "@-ms-keyframes",
    "@font-feature-values", "@viewport", "@-ms-viewport",
    "@counter-style", "@property", "@container", "@layer", "@scope",
    "@starting-style", "@font-palette-values", "@position-try",
    "@-webkit-region", "@apply",
]


# ---------------------------------------------------------------------------
# Function names (WITHOUT trailing `(` — the walker now emits the paren as a
# separate literal token).
# ---------------------------------------------------------------------------

CSS_FUNCTIONS = [
    # Color functions
    "rgb", "rgba", "hsl", "hsla", "hwb", "lab", "lch", "oklab", "oklch",
    "color", "color-mix", "color-contrast", "device-cmyk",
    # Variables / calc
    "var", "calc", "clamp", "min", "max", "minmax", "env", "attr",
    # URLs
    "url", "src", "local", "format",
    # Gradients
    "linear-gradient", "radial-gradient", "conic-gradient",
    "repeating-linear-gradient", "repeating-radial-gradient", "repeating-conic-gradient",
    "-webkit-linear-gradient", "-moz-linear-gradient", "-o-linear-gradient",
    "-webkit-radial-gradient", "-moz-radial-gradient",
    # Transforms
    "translate", "translatex", "translatey", "translatez", "translate3d",
    "rotate", "rotatex", "rotatey", "rotatez", "rotate3d",
    "scale", "scalex", "scaley", "scalez", "scale3d",
    "skew", "skewx", "skewy",
    "matrix", "matrix3d", "perspective",
    # Easing
    "cubic-bezier", "steps", "linear",
    # Counters / generated content
    "counter", "counters", "leader", "target-counter", "target-text",
    # Shapes
    "polygon", "circle", "ellipse", "path", "inset", "rect", "shape",
    # Filters
    "blur", "brightness", "contrast", "drop-shadow", "grayscale", "hue-rotate",
    "invert", "opacity", "saturate", "sepia",
    # Selectors (used inside :is(), :not(), etc. and as functional pseudo-classes)
    "nth-child", "nth-last-child", "nth-of-type", "nth-last-of-type",
    "not", "is", "where", "has", "lang", "dir", "host", "host-context",
    # Other
    "alpha", "color-stop", "image", "image-set", "-webkit-image-set",
    "cross-fade", "element", "paint", "fit-content", "repeat",
    "selector", "supports",
]


# ---------------------------------------------------------------------------
# Units (now emitted as separate IDENT tokens via the dimension-split walker)
# ---------------------------------------------------------------------------

CSS_UNITS = [
    # Length absolute
    "px", "cm", "mm", "in", "pt", "pc", "Q",
    # Length relative
    "em", "rem", "ex", "ch", "lh", "rlh", "cap", "ic",
    # Viewport
    "vh", "vw", "vmin", "vmax", "vi", "vb",
    "svh", "svw", "lvh", "lvw", "dvh", "dvw",
    # Container queries
    "cqw", "cqh", "cqi", "cqb", "cqmin", "cqmax",
    # Grid
    "fr",
    # Angle
    "deg", "rad", "turn", "grad",
    # Time
    "s", "ms",
    # Frequency / resolution
    "hz", "khz", "dpi", "dpcm", "dppx", "x",
]


# ---------------------------------------------------------------------------
# CSS-wide keywords
# ---------------------------------------------------------------------------

CSS_WIDE_KEYWORDS = [
    "inherit", "initial", "unset", "revert", "revert-layer",
    "auto", "none", "normal", "default",
    "transparent", "currentcolor", "currentColor",
    "important",
    "inline", "block", "inline-block", "flex", "inline-flex", "grid",
    "inline-grid", "table", "inline-table", "table-row", "table-cell",
    "table-header-group", "table-footer-group", "table-row-group",
    "table-column", "table-column-group", "list-item", "run-in",
    "contents", "flow-root",
    "absolute", "relative", "fixed", "static", "sticky",
    "row", "row-reverse", "column", "column-reverse", "wrap", "nowrap",
    "wrap-reverse",
    "center", "stretch", "baseline", "flex-start", "flex-end",
    "start", "end", "self-start", "self-end",
    "space-between", "space-around", "space-evenly",
    "left", "right",  # used both as alignment and direction
    "top", "bottom", "middle",  # vertical-align etc.
    "solid", "dashed", "dotted", "double", "groove", "ridge", "inset", "outset",
    "hidden", "visible", "scroll",
    "uppercase", "lowercase", "capitalize",
    "bold", "bolder", "lighter",
    "italic", "oblique",
    "underline", "overline", "line-through",
    "ease", "ease-in", "ease-out", "ease-in-out", "step-start", "step-end",
    "infinite", "alternate", "reverse", "alternate-reverse",
    "forwards", "backwards", "both", "running", "paused",
    "border-box", "content-box", "padding-box", "margin-box",
    "no-repeat", "repeat-x", "repeat-y", "round", "space", "repeat",
    "cover", "contain", "fill",
    "first", "last", "first-page", "left-page", "right-page",
    "always", "avoid", "page", "column", "region", "recto", "verso",
    "balance", "balance-all",
    "pre", "pre-wrap", "pre-line", "break-word", "break-all", "keep-all",
    "manual",
]


# ---------------------------------------------------------------------------
# Pseudo-classes and pseudo-elements (without leading `:` / `::` — the walker
# emits the colon separately as a literal). Underlying ident only.
# ---------------------------------------------------------------------------

PSEUDO_CLASSES = [
    "hover", "focus", "focus-within", "focus-visible", "active", "visited",
    "link", "any-link", "target", "target-within",
    "root", "scope", "host", "host-context",
    "first-child", "last-child", "only-child", "first-of-type", "last-of-type",
    "only-of-type", "nth-child", "nth-last-child", "nth-of-type", "nth-last-of-type",
    "empty", "not", "is", "where", "has", "matches",
    "disabled", "enabled", "checked", "indeterminate", "default",
    "valid", "invalid", "required", "optional", "in-range", "out-of-range",
    "read-only", "read-write", "placeholder-shown", "user-invalid",
    "blank", "current", "past", "future", "playing", "paused", "muted",
    "modal", "fullscreen", "picture-in-picture", "popover-open",
    "defined", "lang", "dir", "autofill",
    # Pseudo-elements
    "before", "after", "first-line", "first-letter",
    "selection", "spelling-error", "grammar-error",
    "placeholder", "marker", "backdrop", "cue", "part", "slotted",
    "-webkit-scrollbar", "-webkit-scrollbar-thumb", "-webkit-scrollbar-track",
    "-webkit-input-placeholder", "-moz-placeholder", "-ms-input-placeholder",
    "-webkit-progress-bar", "-webkit-progress-value", "-moz-progress-bar",
    "-webkit-meter-bar", "-webkit-meter-optimum-value",
    "-webkit-meter-suboptimum-value", "-webkit-meter-even-less-good-value",
]


# ---------------------------------------------------------------------------
# Structural literals (mostly already in fat head by frequency, but pinned)
# ---------------------------------------------------------------------------

STRUCTURAL_LITERALS = [
    "{", "}", ";", ":", ",", "(", ")", "[", "]",
    ">", "+", "~", "*", "/", "!", "=", "^", "$", "|",
    "-", "&", "@", "#",
]


# ---------------------------------------------------------------------------
# Combined export — used by stage2_build_vocab.py
# ---------------------------------------------------------------------------

def _dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


ALL_FORCE_INCLUDE = _dedup_preserve_order(
    NAMED_COLORS_LIST
    + HEX_DIGIT_PAIRS
    + INTEGERS_0_255
    + COMMON_FLOATS
    + CSS_PROPERTIES
    + CSS_AT_KEYWORDS
    + CSS_FUNCTIONS
    + CSS_UNITS
    + CSS_WIDE_KEYWORDS
    + PSEUDO_CLASSES
    + STRUCTURAL_LITERALS
)


CATEGORY_SIZES = {
    "named_colors": len(NAMED_COLORS_LIST),
    "hex_digit_pairs": len(HEX_DIGIT_PAIRS),
    "integers_0_255": len(INTEGERS_0_255),
    "common_floats": len(COMMON_FLOATS),
    "css_properties": len(CSS_PROPERTIES),
    "css_at_keywords": len(CSS_AT_KEYWORDS),
    "css_functions": len(CSS_FUNCTIONS),
    "css_units": len(CSS_UNITS),
    "css_wide_keywords": len(CSS_WIDE_KEYWORDS),
    "pseudo_classes": len(PSEUDO_CLASSES),
    "structural_literals": len(STRUCTURAL_LITERALS),
    "_total_after_dedup": len(ALL_FORCE_INCLUDE),
}


if __name__ == "__main__":
    print("Force-include vocabulary sizes:")
    for k, v in CATEGORY_SIZES.items():
        print(f"  {k:<24s} {v:>5d}")
