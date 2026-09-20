"""No punctuation dash may reach a reader of this tool's own words.

A dash is banned as punctuation in everything published under this author's
name, and a program's output is published under his name every time it runs.
The rule is about punctuation only. A hyphen inside a compound word, inside a
proper name, and inside an identifier such as `kick-the-tyres`, `remote-shell`
or `--reports-dir` is spelling, and is left alone.

Two halves, because either one alone leaves a hole.

`RenderedOutput` drives every renderer and every command the README documents
and reads what a person would actually see. It cannot see a message that no
fixture here happens to trigger.

`PrintableStringConstants` reads the source instead, so a dash typed into a
message nothing here exercises still fails.

Four ways a dash hides from a scan, all of them found the hard way:

* The ASCII spaced double hyphen prints as a dash and matches no search for
  one. `" -- "` and `" - "` are therefore checked as dash forms in their own
  right, alongside the twelve Unicode characters.
* An escaping layer can hide the character. `json.dumps` writes an em dash as
  the six characters `\\u2014`, and HTML can carry one as `&mdash;` or
  `&#8212;`; a scan of the raw text finds neither. Text is decoded before it is
  scanned. This tool renders Markdown and HTML and has no JSON output surface
  today, so `strings_from_json` guards the one that does not exist yet and is
  exercised by `TheGuardItself` rather than left as untested code.
* Python 3.12 and newer split an f-string into `FSTRING_MIDDLE` tokens, which
  are not `token.STRING`, so a tokenising scan silently stops seeing the inside
  of every f-string. The syntax tree is walked instead, which reports an
  f-string's literal parts as ordinary `ast.Constant` nodes on every version.
* Shell `grep` reports these characters unreliably depending on locale and
  encoding, so every file here is read as bytes and decoded as UTF-8
  explicitly.

What is deliberately not scanned, and why:

* Comments. The syntax tree does not carry one, and a comment is never printed.
  This package's comments do contain the spaced double hyphen, so the exclusion
  is load-bearing rather than theoretical.
* Docstrings, identified by the node they belong to. Nothing here prints one:
  the parser's `description` and `epilog` are explicit strings.
* Text quoted from a scanned repository. A finding's path, its evidence line
  and a repository's description are someone else's words passed through, and
  the tool must reproduce them exactly. The fixtures below are built dash free
  so that what is being tested is this tool's own prose.
* A Markdown bullet, a `|---|` table separator row and a rule line of repeated
  hyphens, which are layout a reader never reads as punctuation.
"""

from __future__ import annotations

import ast
import contextlib
import html as html_module
import io
import json
import pathlib
import re
import tempfile
import unittest

from kick_the_tyres import report as report_module
from kick_the_tyres.cli import build_parser, main, print_summary
from kick_the_tyres.models import Finding, RepoReport, RepoSignals, RepoSummary, ScoreResult
from kick_the_tyres.report import (
    render_html,
    render_markdown,
    risk_text,
    usefulness_text,
    verdict_text,
)

#: Every dash that is punctuation rather than spelling, plus the two ASCII
#: sequences that print as one. A bare `-` is absent on purpose: it is
#: legitimate inside a compound word, a name and a command-line flag.
DASH_FORMS = {
    "‐": "HYPHEN",
    "‑": "NON-BREAKING HYPHEN",
    "‒": "FIGURE DASH",
    "–": "EN DASH",
    "—": "EM DASH",
    "―": "HORIZONTAL BAR",
    "−": "MINUS SIGN",
    "⸺": "TWO-EM DASH",
    "⸻": "THREE-EM DASH",
    "﹘": "SMALL EM DASH",
    "﹣": "SMALL HYPHEN-MINUS",
    "－": "FULLWIDTH HYPHEN-MINUS",
    " -- ": "SPACED DOUBLE HYPHEN",
    " - ": "SPACED HYPHEN",
}

_BULLET = re.compile(r"^\s*[-*+]\s")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_RULE_LINE = re.compile(r"^\s*-{3,}\s*$")


def punctuation_dashes(text: str) -> list[str]:
    """Every punctuation dash in `text`, with the line it sits on."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _TABLE_SEPARATOR.match(line) or _RULE_LINE.match(line):
            continue
        body = _BULLET.sub("", line, count=1)
        for form, name in DASH_FORMS.items():
            if form in body:
                found.append(f"line {number}: {name} in {line.strip()!r}")
    return found


def strings_from_json(blob: str) -> str:
    """Every string in a JSON document, keys included, one per line.

    `json.dumps` escapes a non-ASCII character by default, so an em dash
    reaches standard output as the six characters `\\u2014` and a search of the
    raw text would not find it. Parsing first is the only honest way to read it.
    """
    out: list[str] = []

    def walk(node):
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                out.append(key)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(blob))
    return "\n".join(out)


def readable_html(markup: str) -> str:
    """The HTML with its entities resolved, which is what a browser shows.

    `&mdash;` and `&#8212;` are seven and eight ASCII characters that a reader
    sees as an em dash. Scanning the markup as written would miss both.
    """
    return html_module.unescape(markup)


def _call(*argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _captured(fn, *args) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue()


def sample_report(*, static_scan: bool, verdict: str | None, findings: list) -> RepoReport:
    """A report whose every field is this tool's own vocabulary.

    Nothing here is quoted from a scanned repository, so any dash the scan
    finds was written into this package rather than passed through it.
    """
    return RepoReport(
        repo=RepoSummary(
            full_name="owner/repo",
            description="A sample repository",
            url="https://github.com/owner/repo",
            stars=10,
            forks=1,
        ),
        signals=RepoSignals(has_readme="present"),
        findings=findings,
        score=ScoreResult(
            usefulness=70,
            risk=80 if static_scan else 0,
            verdict=verdict,
            reasons=["critical risk"] if verdict else [],
            static_scan=static_scan,
            usefulness_ceiling=90,
            verdict_blockers=[] if verdict else ["usefulness spans two labels"],
        ),
    )


#: The report states this tool reaches, each one rendering a different branch:
#: a scan that ran and found something, a scan that ran and found nothing, a
#: run with no scan at all, and a run that could not settle on a label.
REPORT_STATES = {
    "a scan with a finding": lambda: sample_report(
        static_scan=True,
        verdict="AVOID",
        findings=[
            Finding(
                severity="critical",
                rule="remote-shell",
                path="install.sh",
                message="Remote shell execution",
                evidence="curl example.invalid/install.sh | bash",
            )
        ],
    ),
    "a scan with no findings": lambda: sample_report(
        static_scan=True, verdict="USE", findings=[]
    ),
    "no scan performed": lambda: sample_report(
        static_scan=False, verdict="INSPECT FIRST", findings=[]
    ),
    "no verdict established": lambda: sample_report(
        static_scan=True, verdict=None, findings=[]
    ),
}


class RenderedOutput(unittest.TestCase):
    """What a person actually sees, in every format and on every path."""

    def assertNoDash(self, text: str, what: str):
        offending = punctuation_dashes(text)
        self.assertEqual(
            offending, [], f"{what} carries a punctuation dash:\n" + "\n".join(offending)
        )

    def test_markdown_report(self):
        for what, build in REPORT_STATES.items():
            with self.subTest(state=what):
                self.assertNoDash(render_markdown(build()), f"the Markdown report, {what}")

    def test_html_report_as_a_browser_shows_it(self):
        for what, build in REPORT_STATES.items():
            with self.subTest(state=what):
                markup = render_html(build())
                self.assertNoDash(markup, f"the HTML report markup, {what}")
                self.assertNoDash(
                    readable_html(markup), f"the HTML report as rendered, {what}"
                )

    def test_terminal_summary(self):
        for what, build in REPORT_STATES.items():
            with self.subTest(state=what):
                self.assertNoDash(_captured(print_summary, build()), f"the summary, {what}")

    def test_the_shared_score_sentences(self):
        for what, build in REPORT_STATES.items():
            score = build().score
            for name, fn in (
                ("verdict_text", verdict_text),
                ("risk_text", risk_text),
                ("usefulness_text", usefulness_text),
            ):
                with self.subTest(state=what, function=name):
                    self.assertNoDash(fn(score), f"{name}(), {what}")

    def test_the_always_carried_notices(self):
        """Carried by every artifact the scores travel in, so checked alone."""
        for name in ("LIMITATION", "NO_SCAN_FINDINGS", "NO_SCAN_RISK", "NO_URL", "NO_VERDICT"):
            with self.subTest(constant=name):
                self.assertNoDash(getattr(report_module, name), f"report.{name}")

    def test_help_and_usage_for_every_command(self):
        parser = build_parser()
        self.assertNoDash(parser.format_help(), "the top level --help")
        self.assertNoDash(parser.format_usage(), "the top level usage line")

        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        subcommands = {
            name: sub for action in actions for name, sub in (action.choices or {}).items()
        }
        self.assertTrue(subcommands, "no subcommands were found, so nothing was proved")
        for name, sub in sorted(subcommands.items()):
            with self.subTest(command=name):
                self.assertNoDash(sub.format_help(), f"`{name} --help`")
                self.assertNoDash(sub.format_usage(), f"the `{name}` usage line")

    def test_the_version_line(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit):
                main(["--version"])
        self.assertTrue(out.getvalue().strip(), "--version printed nothing")
        self.assertNoDash(out.getvalue(), "--version")
        self.assertNoDash(err.getvalue(), "--version on stderr")

    def test_the_documented_offline_example(self):
        """The worked example in the README, run exactly as it is written."""
        root = pathlib.Path(tempfile.mkdtemp())
        sample = root / "sample"
        sample.mkdir()
        (sample / "README.md").write_text("# Sample project\n", encoding="utf-8")
        (sample / "install.sh").write_text(
            "curl https://downloads.invalid/install.sh | bash\n", encoding="utf-8"
        )
        code, out, err = _call("--reports-dir", str(root / "reports"), "scan", str(sample))
        self.assertEqual(code, 0, f"the documented example failed: {err}")
        self.assertIn("AVOID", out)
        self.assertNoDash(out, "the documented offline example on stdout")
        self.assertNoDash(err, "the documented offline example on stderr")

        for written in sorted((root / "reports").iterdir()):
            with self.subTest(report=written.name):
                text = written.read_bytes().decode("utf-8")
                self.assertNoDash(text, f"the written {written.suffix} report")
                if written.suffix == ".html":
                    self.assertNoDash(readable_html(text), "the written HTML as rendered")

    def test_every_message_printed_on_a_path_that_fails(self):
        missing = pathlib.Path(tempfile.mkdtemp()) / "nowhere"
        a_file = pathlib.Path(tempfile.mkdtemp()) / "a_file.txt"
        a_file.write_text("not a directory\n", encoding="utf-8")
        cases = [
            (("scan", str(missing)), "a scan path that does not exist"),
            (("report", str(missing)), "a report path that does not exist"),
            (("scan", str(a_file)), "a scan path that is a file"),
            (("inspect", "not a repo name"), "a malformed repository name"),
            (("inspect", "owner"), "a repository name with no owner"),
        ]
        for argv, what in cases:
            with self.subTest(what=what):
                _, out, err = _call(*argv)
                self.assertNoDash(out, f"{what} on stdout")
                self.assertNoDash(err, f"{what} on stderr")

    def test_argparse_own_refusals(self):
        for argv, what in (
            ((), "no command given"),
            (("scan",), "scan with no path"),
            (("nonsense",), "an unknown command"),
            (("search", "q", "--limit", "not-a-number"), "a non numeric --limit"),
        ):
            with self.subTest(what=what):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    with contextlib.suppress(SystemExit):
                        main(list(argv))
                self.assertNoDash(out.getvalue(), f"{what} on stdout")
                self.assertNoDash(err.getvalue(), f"{what} on stderr")


# --- the source side: a message no fixture above happens to trigger ---


def _printable_strings(tree: ast.AST) -> list[str]:
    """Every string literal in a module except the docstrings.

    A comment never reaches a reader and never reaches this list either,
    because the syntax tree does not carry one. A docstring is skipped by
    identifying the node it belongs to, which is stable across Python versions
    in a way that tokenising an f-string is not: from Python 3.12 an f-string's
    literal text arrives as `FSTRING_MIDDLE` rather than `token.STRING`, and a
    tokenising scan stops seeing inside every f-string without saying so.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _package_modules() -> list[pathlib.Path]:
    return sorted(pathlib.Path(report_module.__file__).parent.rglob("*.py"))


class PrintableStringConstants(unittest.TestCase):
    def test_the_package_has_modules_to_read(self):
        """Guards the test below: an empty list would pass it reading nothing."""
        self.assertGreaterEqual(len(_package_modules()), 8)

    def test_no_printable_string_holds_a_dash(self):
        for module in _package_modules():
            with self.subTest(module=module.name):
                # Read as bytes and decoded explicitly: shell `grep` reports
                # these characters unreliably depending on locale, and so does
                # anything that lets the platform choose the encoding.
                tree = ast.parse(module.read_bytes().decode("utf-8"))
                offences = []
                for literal in _printable_strings(tree):
                    offences += [
                        f"{literal!r}: {hit}" for hit in punctuation_dashes(literal)
                    ]
                self.assertEqual(
                    offences, [], f"{module.name} holds a printable string with a dash"
                )


class TheGuardItself(unittest.TestCase):
    """The detector has to be able to fail, or it proves nothing."""

    def test_it_catches_every_dash_form(self):
        for form, name in DASH_FORMS.items():
            with self.subTest(form=name):
                self.assertTrue(
                    punctuation_dashes(f"a{form}b"), f"{name} slipped through"
                )

    def test_it_catches_a_dash_escaped_into_json(self):
        blob = json.dumps({"verdict": "AVOID — review this"})
        self.assertNotIn("—", blob, "json.dumps did not escape, so nothing is proved")
        self.assertFalse(punctuation_dashes(blob), "the raw JSON must hide the dash")
        self.assertTrue(punctuation_dashes(strings_from_json(blob)))

    def test_it_catches_a_dash_written_as_an_html_entity(self):
        for markup in ("<p>AVOID &mdash; review</p>", "<p>AVOID &#8212; review</p>"):
            with self.subTest(markup=markup):
                self.assertFalse(punctuation_dashes(markup), "the markup must hide it")
                self.assertTrue(punctuation_dashes(readable_html(markup)))

    def test_it_reads_inside_an_f_string(self):
        """From Python 3.12 this text is not a `token.STRING`."""
        source = 'name = "x"\nmessage = f"verdict {name} — review"\n'
        literals = _printable_strings(ast.parse(source))
        self.assertTrue(
            any(punctuation_dashes(literal) for literal in literals),
            "the inside of the f-string was never read",
        )

    def test_it_allows_spelling(self):
        self.assertEqual(punctuation_dashes("kick-the-tyres found remote-shell"), [])
        self.assertEqual(punctuation_dashes("read-only, non-technical, GST-net"), [])
        self.assertEqual(punctuation_dashes("pass --reports-dir to choose it"), [])

    def test_it_allows_layout(self):
        self.assertEqual(punctuation_dashes("- a bullet"), [])
        self.assertEqual(punctuation_dashes("    - an indented bullet"), [])
        self.assertEqual(punctuation_dashes("|---|---|"), [])
        self.assertEqual(punctuation_dashes("-" * 40), [])

    def test_a_bullet_does_not_hide_a_dash_after_it(self):
        self.assertTrue(punctuation_dashes("- a bullet — with a dash in it"))

    def test_it_ignores_a_comment_and_a_docstring(self):
        """Both are excluded on purpose, so the exclusion is stated as a test."""
        source = '"""A docstring — with a dash."""\n# a comment — with a dash\nx = "clean"\n'
        literals = _printable_strings(ast.parse(source))
        self.assertEqual(literals, ["clean"])


if __name__ == "__main__":
    unittest.main()
