"""Render LaTeX ``algorithm``/``algpseudocode`` listings as typeset pseudocode.

Usage in MyST Markdown::

    ```{algorithm}
    :name: alg-my-label
    \\begin{algorithm}
    \\caption{My algorithm}
    \\begin{algorithmic}[1]
    \\Require $x$
    \\State $y \\gets x^2$
    \\State \\Return $y$
    \\end{algorithmic}
    \\end{algorithm}
    ```

The TeX is parsed at build time into numbered, indented lines with bold keywords
and MathJax math, laid out like a compiled LaTeX ``algorithm`` float. Algorithms
are numbered through ``numfig`` ("Algorithm 1", "Algorithm 2", ...) and can be
cross-referenced with ``{numref}``. The original TeX is kept in a collapsible
"LaTeX source" panel; LaTeX builds emit the TeX verbatim.
"""

from __future__ import annotations

import re
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

_COMMAND = re.compile(r"\\([A-Za-z]+)\*?")

# Structural environment wrappers that carry no visible content.
_IGNORED = {"begin", "end", "label", "centering", "small", "footnotesize"}

# Block commands: (keyword before argument, keyword after argument, indent change).
_OPENERS = {
    "For": ("for", "do"),
    "ForAll": ("for all", "do"),
    "While": ("while", "do"),
    "If": ("if", "then"),
}
_CLOSERS = {
    "EndFor": "end for",
    "EndWhile": "end while",
    "EndIf": "end if",
    "EndLoop": "end loop",
    "EndFunction": "end function",
    "EndProcedure": "end procedure",
}
_HEADERS = {"Require": "Require:", "Ensure": "Ensure:", "Input": "Input:", "Output": "Output:"}
_INLINE_KEYWORDS = {
    "Return": "return",
    "And": "and",
    "Or": "or",
    "Not": "not",
    "To": "to",
    "Break": "break",
    "Continue": "continue",
}
_INLINE_CONSTANTS = {"True": "true", "False": "false", "Null": "null"}
_TEXT_STYLES = {"textbf": nodes.strong, "emph": nodes.emphasis, "textit": nodes.emphasis}
_ESCAPES = {"{": "{", "}": "}", "$": "$", "%": "%", "&": "&", "_": "_", "#": "#", ",": "\u2009", " ": " "}


class algorithm(nodes.General, nodes.Element):
    """A typeset pseudocode float."""


# ---------------------------------------------------------------- TeX parsing


def _skip_spaces(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _read_group(text: str, i: int) -> tuple[str, int]:
    """Return the contents of the ``{...}`` group starting at ``text[i]`` (after spaces)."""
    i = _skip_spaces(text, i)
    if i >= len(text) or text[i] != "{":
        return "", i
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "\\":
            continue
        if text[j] == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif text[j] == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
    raise ValueError(f"Unbalanced braces in: {text!r}")


def _find_math_end(text: str, i: int, closer: str) -> int:
    j = i
    while True:
        j = text.find(closer, j)
        if j == -1:
            raise ValueError(f"Unterminated math in: {text!r}")
        if text[j - 1] != "\\":
            return j
        j += 1


def _inline(text: str) -> list[nodes.Node]:
    """Convert a run of algorithmic text (with ``$math$`` and text macros) into nodes."""
    out: list[nodes.Node] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(nodes.Text("".join(buf)))
            buf.clear()

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "$" or text.startswith("\\(", i):
            opener, closer = ("$", "$") if ch == "$" else ("\\(", "\\)")
            end = _find_math_end(text, i + len(opener), closer)
            latex = text[i + len(opener) : end].strip()
            flush()
            out.append(nodes.math(latex, latex))
            i = end + len(closer)
            continue
        if ch == "\\":
            if i + 1 < len(text) and text[i + 1] in _ESCAPES:
                buf.append(_ESCAPES[text[i + 1]])
                i += 2
                continue
            if text.startswith("\\\\", i):
                buf.append(" ")
                i += 2
                continue
            match = _COMMAND.match(text, i)
            if match is None:
                buf.append(ch)
                i += 1
                continue
            name, i = match.group(1), match.end()
            if name in _INLINE_KEYWORDS:
                flush()
                out.append(nodes.strong(text=_INLINE_KEYWORDS[name], classes=["alg-keyword"]))
            elif name in _INLINE_CONSTANTS:
                flush()
                out.append(nodes.inline(text=_INLINE_CONSTANTS[name], classes=["alg-smallcaps"]))
            elif name == "Call":
                proc, i = _read_group(text, i)
                args, i = _read_group(text, i)
                flush()
                out.append(nodes.inline("", "", *_inline(proc), classes=["alg-smallcaps"]))
                out.extend([nodes.Text("("), *_inline(args), nodes.Text(")")])
            elif name == "Comment":
                arg, i = _read_group(text, i)
                flush()
                out.append(nodes.inline("", "", nodes.Text("\u25b7 "), *_inline(arg), classes=["alg-comment"]))
            elif name in _TEXT_STYLES:
                arg, i = _read_group(text, i)
                flush()
                out.append(_TEXT_STYLES[name]("", "", *_inline(arg)))
            elif name == "textsc":
                arg, i = _read_group(text, i)
                flush()
                out.append(nodes.inline("", "", *_inline(arg), classes=["alg-smallcaps"]))
            elif name == "texttt":
                arg, i = _read_group(text, i)
                flush()
                out.append(nodes.literal(arg, arg))
            elif name in ("quad", "qquad"):
                buf.append("\u2003" if name == "quad" else "\u2003\u2003")
            else:
                buf.append(match.group(0))
            continue
        if ch in "{}":
            i += 1
            continue
        buf.append("\u00a0" if ch == "~" else ch)
        i += 1
    flush()
    # Collapse whitespace at the edges so lines align cleanly.
    if out and isinstance(out[0], nodes.Text):
        out[0] = nodes.Text(out[0].astext().lstrip())
    if out and isinstance(out[-1], nodes.Text):
        out[-1] = nodes.Text(out[-1].astext().rstrip())
    return [n for n in out if not (isinstance(n, nodes.Text) and not n.astext())]


def _keyword(word: str) -> nodes.strong:
    return nodes.strong(text=word, classes=["alg-keyword"])


def _logical_lines(source: str) -> list[str]:
    """Split the TeX into one entry per algorithmic command, joining continuation lines."""
    lines: list[str] = []
    for raw in source.splitlines():
        line = re.sub(r"(?<!\\)%.*$", "", raw).strip()
        if not line:
            continue
        if lines and not line.startswith("\\"):
            lines[-1] += " " + line
        else:
            lines.append(line)
    return lines


def parse_algorithm(source: str) -> tuple[str | None, int, list[dict]]:
    """Parse algpseudocode TeX into ``(caption, number_every, lines)``.

    Each line is ``{"indent": int, "numbered": bool, "nodes": [...], "kind": str}``.
    """
    caption = None
    number_every = 0
    indent = 0
    rows: list[dict] = []

    def emit(content, *, numbered=True, kind="state"):
        rows.append({"indent": max(indent, 0), "numbered": numbered, "nodes": content, "kind": kind})

    for line in _logical_lines(source):
        match = _COMMAND.match(line)
        name = match.group(1) if match else None
        rest = line[match.end() :] if match else line

        if name == "begin":
            env, pos = _read_group(line, match.end())
            if env == "algorithmic":
                opt = re.match(r"\s*\[(\d+)\]", line[pos:])
                number_every = int(opt.group(1)) if opt else 0
            continue
        if name == "caption":
            caption, _ = _read_group(line, match.end())
            continue
        if name in _IGNORED:
            continue

        if name in _HEADERS:
            emit([_keyword(_HEADERS[name]), nodes.Text(" "), *_inline(rest)], numbered=False, kind="header")
        elif name == "State":
            emit(_inline(rest))
        elif name == "Statex":
            emit(_inline(rest), numbered=False)
        elif name in _OPENERS:
            cond, pos = _read_group(line, match.end())
            tail = line[pos:]
            before, after = _OPENERS[name]
            emit([_keyword(before), nodes.Text(" "), *_inline(cond), nodes.Text(" "), _keyword(after), *_inline(tail)])
            indent += 1
        elif name == "ElsIf":
            cond, pos = _read_group(line, match.end())
            tail = line[pos:]
            indent -= 1
            emit([_keyword("else if"), nodes.Text(" "), *_inline(cond), nodes.Text(" "), _keyword("then"), *_inline(tail)])
            indent += 1
        elif name == "Else":
            indent -= 1
            emit([_keyword("else"), *_inline(rest)])
            indent += 1
        elif name in ("Loop", "Repeat"):
            emit([_keyword(name.lower()), *_inline(rest)])
            indent += 1
        elif name == "Until":
            cond, pos = _read_group(line, match.end())
            tail = line[pos:]
            indent -= 1
            emit([_keyword("until"), nodes.Text(" "), *_inline(cond), *_inline(tail)])
        elif name in ("Function", "Procedure"):
            proc, pos = _read_group(line, match.end())
            args, pos = _read_group(line, pos)
            emit(
                [
                    _keyword(name.lower()),
                    nodes.Text(" "),
                    nodes.inline("", "", *_inline(proc), classes=["alg-smallcaps"]),
                    nodes.Text("("),
                    *_inline(args),
                    nodes.Text(")"),
                    *_inline(line[pos:]),
                ]
            )
            indent += 1
        elif name in _CLOSERS:
            indent -= 1
            emit([_keyword(_CLOSERS[name]), *_inline(rest)])
        else:
            # A bare line (e.g. "\Return $w$" or plain text) is treated like \State.
            emit(_inline(line))

    return caption, number_every, rows


# ---------------------------------------------------------------- directive


class AlgorithmDirective(SphinxDirective):
    has_content = True
    option_spec: ClassVar[dict] = {
        "name": directives.unchanged,
        "class": directives.class_option,
        "caption": directives.unchanged,
        "no-source": directives.flag,
    }

    def run(self) -> list[nodes.Node]:
        source = "\n".join(self.content)
        try:
            caption, number_every, rows = parse_algorithm(source)
        except ValueError as exc:
            raise self.error(f"Could not parse algorithm: {exc}") from exc
        caption = self.options.get("caption", caption)

        node = algorithm(classes=["algorithm", *self.options.get("class", [])])
        node["tex"] = source
        self.set_source_info(node)
        if "name" in self.options:
            self.add_name(node)
        else:
            self.state.document.set_id(node)

        if caption:
            node += nodes.caption(caption, "", *_inline(caption))

        body = nodes.container(classes=["alg-body"])
        counter = 0
        for row in rows:
            line = nodes.container(classes=["alg-line", f"alg-{row['kind']}", f"alg-indent-{min(row['indent'], 8)}"])
            label = ""
            if row["numbered"] and number_every:
                counter += 1
                if counter % number_every == 0:
                    label = f"{counter}:"
            if number_every:
                line += nodes.inline(text=label, classes=["alg-lineno"])
            line += nodes.inline("", "", *row["nodes"], classes=["alg-content"])
            body += line
        node += body

        if "no-source" not in self.options:
            node += nodes.raw("", '<details class="alg-source"><summary>LaTeX source</summary>', format="html")
            literal = nodes.literal_block(source, source, language="latex")
            node += literal
            node += nodes.raw("", "</details>", format="html")

        return [node]


# ---------------------------------------------------------------- writers


def visit_algorithm_html(self, node: algorithm) -> None:
    self.body.append(self.starttag(node, "div"))


def depart_algorithm_html(self, node: algorithm) -> None:
    self.body.append("</div>\n")


def visit_algorithm_latex(self, node: algorithm) -> None:
    self.body.append("\n" + node["tex"] + "\n")
    raise nodes.SkipNode


def visit_algorithm_passthrough(self, node: algorithm) -> None:
    pass


def depart_algorithm_passthrough(self, node: algorithm) -> None:
    pass


def _get_caption(node: algorithm) -> str | None:
    for child in node.children:
        if isinstance(child, nodes.caption):
            return child.astext()
    return None


def _register_numfig_format(app, config) -> None:
    config.numfig_format.setdefault("algorithm", "Algorithm %s")


def setup(app):
    passthrough = (visit_algorithm_passthrough, depart_algorithm_passthrough)
    app.add_enumerable_node(
        algorithm,
        "algorithm",
        _get_caption,
        html=(visit_algorithm_html, depart_algorithm_html),
        latex=(visit_algorithm_latex, None),
        text=passthrough,
        man=passthrough,
        texinfo=passthrough,
    )
    app.add_directive("algorithm", AlgorithmDirective)
    app.add_latex_package("algorithm")
    app.add_latex_package("algpseudocode")
    app.connect("config-inited", _register_numfig_format, priority=900)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
