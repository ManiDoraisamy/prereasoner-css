"""
CSS AST walker — converts a CSS file into a flat stream of (text, type, context)
tokens.

Token representation:
  Each token is a (text, type, context) tuple.
  - text: canonical string form (e.g., "#fff", "12px", "rgb(", "red", ":")
  - type: short tag from TOKEN_TYPES below
  - context: CTX_SELECTOR | CTX_VALUE | CTX_DECL_NAME | CTX_ATRULE_PRELUDE
             | CTX_STRUCTURAL | CTX_GENERAL

The context tag is critical: a token text like "red" or "#fff" means a CSS
color value when it appears in CTX_VALUE, but a class selector or id selector
when it appears in CTX_SELECTOR. Color anchoring must filter on context.

Strings (StringToken) and URLs (URLToken) carry a type tag that signals the
downstream tokenizer to BPE-encode them rather than treat them as fat-head
candidates, since they have unbounded vocabulary.

Joining rules:
  - Hash colors / IDs:  "#fff", "#header"  (one token, the '#' is fused)
  - Dimensions:         "12px", "1.5em"     (number+unit fused)
  - Percentages:        "50%"                (number+'%' fused)
  - At-keywords:        "@media", "@import"  (one token, '@' fused)
  - Functions:          emitted as "rgb(" then args then LITERAL ")"
"""

import tinycss2

# Token type tags
T_IDENT = "ident"
T_HASH = "hash"
T_AT = "at"
T_NUMBER = "number"
T_DIM = "dim"
T_PCT = "pct"
T_STRING = "string"      # always BPE
T_URL = "url"            # always BPE
T_FUNC = "func"          # function-name token (with trailing '(')
T_LITERAL = "literal"    # :, ;, ,, {, }, (, ), [, ], >, +, ~, *, /, !
T_UNICODE_RANGE = "urange"
T_DELIM = "delim"
T_COMMENT = "comment"
T_PARSE_ERROR = "error"

BPE_TYPES = {T_STRING, T_URL}

# Hex digit characters (lowercase form used for digit-pair tokenization)
_HEX_CHARS = set("0123456789abcdefABCDEF")


def _is_valid_hex(body: str) -> bool:
    """True if `body` is 3/4/6/8 chars of hex digits."""
    return len(body) in (3, 4, 6, 8) and all(c in _HEX_CHARS for c in body)


def _expand_hex(body: str) -> str:
    """Expand a hex color body to canonical 6-char lowercase form.

    - 3-char (#rgb)  -> 'rrggbb'   (each digit doubled)
    - 4-char (#rgba) -> 'rrggbb'   (alpha dropped)
    - 6-char         -> 'rrggbb'   (already canonical, lowercased)
    - 8-char         -> 'rrggbb'   (alpha dropped)
    """
    s = body.lower()
    if len(s) == 3:
        return s[0] * 2 + s[1] * 2 + s[2] * 2
    if len(s) == 4:
        return s[0] * 2 + s[1] * 2 + s[2] * 2  # alpha ignored
    if len(s) == 6:
        return s
    if len(s) == 8:
        return s[:6]  # alpha ignored
    return s  # shouldn't happen — caller guards with _is_valid_hex


# Context tags
CTX_SELECTOR = "selector"            # inside a QualifiedRule.prelude
CTX_VALUE = "value"                  # inside a Declaration.value
CTX_DECL_NAME = "decl_name"          # the property name in a declaration
CTX_ATRULE_PRELUDE = "atrule_prelude"  # inside an AtRule.prelude
CTX_STRUCTURAL = "structural"        # {, }, ;, : separator, !, @keyword
CTX_GENERAL = "general"              # fallback (rare)


def _walk_tokens(tokens, out, context: str):
    """Walk a list of tinycss2 component-value tokens, appending (text, type, context) to out."""
    for tok in tokens:
        t = type(tok).__name__

        if t == "WhitespaceToken":
            continue

        elif t == "IdentToken":
            out.append((tok.value, T_IDENT, context))

        elif t == "HashToken":
            # If the hash body is a valid hex color, emit as digit-pair
            # tokens: `#`, RR, GG, BB. This gives the model a structural
            # signal for hex -> RGB rather than a memorized lookup per
            # hex literal. Non-hex hashes (e.g. id selectors `#header`)
            # keep their fused form so they stay readable.
            if _is_valid_hex(tok.value):
                expanded = _expand_hex(tok.value)
                out.append(("#", T_LITERAL, context))
                out.append((expanded[0:2], T_HASH, context))   # R pair
                out.append((expanded[2:4], T_HASH, context))   # G pair
                out.append((expanded[4:6], T_HASH, context))   # B pair
            else:
                out.append(("#" + tok.value, T_HASH, context))

        elif t == "AtKeywordToken":
            out.append(("@" + tok.value, T_AT, context))

        elif t == "StringToken":
            out.append(('"' + tok.value.replace('"', '\\"') + '"', T_STRING, context))

        elif t == "NumberToken":
            out.append((tok.representation, T_NUMBER, context))

        elif t == "DimensionToken":
            # Split into number + unit so the unit stays atomic
            # via force-include (px, em, rem, deg, %, ...).
            out.append((tok.representation, T_NUMBER, context))
            out.append((tok.unit, T_IDENT, context))

        elif t == "PercentageToken":
            # Split for consistency with dimensions; '%' is in STRUCTURAL_LITERALS.
            out.append((tok.representation, T_NUMBER, context))
            out.append(("%", T_LITERAL, context))

        elif t == "URLToken":
            # Split: 'url' name + '(' + body (always BPE) + ')'
            out.append(("url", T_FUNC, context))
            out.append(("(", T_LITERAL, context))
            out.append((tok.value, T_URL, context))
            out.append((")", T_LITERAL, CTX_STRUCTURAL))

        elif t == "UnicodeRangeToken":
            out.append((tok.serialize(), T_UNICODE_RANGE, context))

        elif t == "LiteralToken":
            # Inline separators inside selector/value keep the surrounding context;
            # the lexically structural ones (parens etc.) keep it too.
            out.append((tok.value, T_LITERAL, context))

        elif t == "FunctionBlock":
            # Split: name + '(' so the function name is atomic regardless
            # of which paren follows it.
            out.append((tok.lower_name, T_FUNC, context))
            out.append(("(", T_LITERAL, context))
            _walk_tokens(tok.arguments, out, context)
            out.append((")", T_LITERAL, CTX_STRUCTURAL))

        elif t == "ParenthesesBlock":
            out.append(("(", T_LITERAL, context))
            _walk_tokens(tok.content, out, context)
            out.append((")", T_LITERAL, CTX_STRUCTURAL))

        elif t == "SquareBracketsBlock":
            out.append(("[", T_LITERAL, context))
            _walk_tokens(tok.content, out, context)
            out.append(("]", T_LITERAL, CTX_STRUCTURAL))

        elif t == "CurlyBracketsBlock":
            out.append(("{", T_LITERAL, context))
            _walk_tokens(tok.content, out, context)
            out.append(("}", T_LITERAL, CTX_STRUCTURAL))

        elif t == "Comment":
            continue

        elif t == "ParseError":
            continue

        else:
            try:
                out.append((tok.serialize(), T_DELIM, context))
            except Exception:
                pass


def _walk_declarations(declarations, out):
    """Walk a declaration list, appending tokens to out with proper contexts."""
    for decl in declarations:
        if type(decl).__name__ != "Declaration":
            continue
        out.append((decl.name, T_IDENT, CTX_DECL_NAME))
        out.append((":", T_LITERAL, CTX_STRUCTURAL))
        _walk_tokens(decl.value, out, CTX_VALUE)
        if decl.important:
            out.append(("!", T_LITERAL, CTX_STRUCTURAL))
            out.append(("important", T_IDENT, CTX_VALUE))
        out.append((";", T_LITERAL, CTX_STRUCTURAL))


def _walk_rules(rules, out):
    """Walk a list of top-level rules (QualifiedRule / AtRule)."""
    for rule in rules:
        t = type(rule).__name__

        if t == "QualifiedRule":
            _walk_tokens(rule.prelude, out, CTX_SELECTOR)
            out.append(("{", T_LITERAL, CTX_STRUCTURAL))
            decls = tinycss2.parse_declaration_list(
                rule.content, skip_whitespace=True, skip_comments=True
            )
            _walk_declarations(decls, out)
            out.append(("}", T_LITERAL, CTX_STRUCTURAL))

        elif t == "AtRule":
            out.append(("@" + rule.lower_at_keyword, T_AT, CTX_STRUCTURAL))
            _walk_tokens(rule.prelude, out, CTX_ATRULE_PRELUDE)
            if rule.content is None:
                out.append((";", T_LITERAL, CTX_STRUCTURAL))
            else:
                out.append(("{", T_LITERAL, CTX_STRUCTURAL))
                try:
                    nested_rules = tinycss2.parse_rule_list(
                        rule.content, skip_whitespace=True, skip_comments=True
                    )
                    if nested_rules and any(
                        type(r).__name__ in ("QualifiedRule", "AtRule") for r in nested_rules
                    ):
                        _walk_rules(nested_rules, out)
                    else:
                        decls = tinycss2.parse_declaration_list(
                            rule.content, skip_whitespace=True, skip_comments=True
                        )
                        _walk_declarations(decls, out)
                except Exception:
                    pass
                out.append(("}", T_LITERAL, CTX_STRUCTURAL))

        elif t == "ParseError":
            continue


def walk_css(css_text: str) -> list[tuple[str, str, str]]:
    """Parse a CSS string and return a flat list of (text, type, context) tokens.

    Whitespace and comments are stripped. Hex codes, dimensions, percentages,
    and at-keywords are emitted as single fused tokens. Function calls expand
    to "name(" + args + ")". Each token carries a context tag (CTX_*) so
    downstream code can distinguish selector tokens from value tokens.
    """
    out: list[tuple[str, str, str]] = []
    rules = tinycss2.parse_stylesheet(
        css_text, skip_whitespace=True, skip_comments=True
    )
    _walk_rules(rules, out)
    return out


def serialize_tokens(tokens) -> str:
    """Serialize a token stream back to CSS text (single-space joiner).

    Accepts both 2-tuples (legacy) and 3-tuples (current). Adjacent literals
    get tight joining; everything else is space-separated. Round-trip use only.
    """
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        text, ttype = tok[0], tok[1]
        if i == 0:
            parts.append(text)
            continue
        prev = tokens[i - 1]
        prev_text, prev_type = prev[0], prev[1]
        tight_after = prev_text in ("(", "[", "{", "@")
        tight_before = text in (")", "]", "}", ",", ";", ":")
        if prev_type == T_FUNC:
            tight_after = True
        if tight_after or tight_before:
            parts.append(text)
        else:
            parts.append(" " + text)
    return "".join(parts)
