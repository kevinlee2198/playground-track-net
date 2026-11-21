"""How guard_bash.py reads a Bash command: comments, heredocs, words, simple commands.

Each line outside a heredoc body is scanned with its quoting: ``'`` and ``"``
open and close strings (one inside the other is plain text), ``$(`` opens a
command substitution (inside ``"`` too) that ``)`` closes, and ``$((`` and
``((`` open arithmetic that ``))`` closes. Comments are dropped as bash drops
them: from a word that starts with ``#``, outside strings and arithmetic, to
the end of the line. A ``<<`` (not ``<<<``) starts a heredoc only outside
strings and arithmetic: at top level, in a subshell or directly inside
``$(``, as in ``git commit -m "$(cat <<'EOF' ...``. Heredoc bodies are set
aside with their reader: a body fed to a shell or to Python is returned for
checking, and any other (a commit message, say) is data and is dropped. The
rest is split into shell words (a quoted string is one word) and into simple
commands at ``;``, ``&&``, ``||``, ``|``, ``&``, newlines and parentheses,
each with its redirection targets kept apart. Backticks and ``$(...)`` are not
followed as commands.

Standard library only: the hooks run on the system python3.
"""

from __future__ import annotations

import os
import re
import shlex

from held_out import resolve

SHELLS = {"bash", "sh", "zsh", "dash"}
PYTHON = re.compile(r"(?:^|/)python(?:\d+(?:\.\d+)*)?$")
HEREDOC = re.compile(r"<<-?[ \t]*\\?(['\"]?)(\w+)\1")
PIPED_TO = re.compile(r"\|\s*(\S+)")
PUNCTUATION = ";&|()<>\n"
SEPARATORS = set(";|()\n")
# What scan_line's stack holds: strings, command substitution, a subshell,
# arithmetic and a parenthesis inside arithmetic.
SINGLE, DOUBLE, SUBST, SUBSHELL, ARITH, ARITH_PAREN = "'", '"', "$(", "(", "((", "a("
CODE = (SUBST, SUBSHELL)  # where commands, comments and heredocs are read


def basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def split_heredocs(command: str) -> tuple[str, list[tuple[str, str]]]:
    """The command without comments and heredoc bodies, and the bodies to check.

    Each body to check comes with its reader, "shell" or "python"; bodies fed
    to anything else are data and are left out.
    """
    kept: list[str] = []
    bodies: list[tuple[str, str]] = []
    pending: list[tuple[str, str | None]] = []  # (delimiter, reader)
    body: list[str] = []
    stack: list[str] = []  # the quoting open at the end of the last line scanned
    for line in command.split("\n"):
        if not pending:
            line, markers = scan_line(line, stack)
            kept.append(line)
            for match in markers:
                reader = reader_of(line[: match.start()], line[match.end() :])
                pending.append((match.group(2), reader))
        elif line.strip() == pending[0][0]:
            if reader := pending.pop(0)[1]:
                bodies.append(("\n".join(body), reader))
            body = []
        else:
            body.append(line)
    if pending and (reader := pending[0][1]):
        bodies.append(("\n".join(body), reader))
    return "\n".join(kept), bodies


def scan_line(line: str, stack: list[str]) -> tuple[str, list[re.Match[str]]]:
    """``line`` up to its comment, and the heredoc markers on it.

    ``stack`` is the quoting open at the start of the line (see the module
    docstring; a string can span lines), and is left as it is at the end.
    """
    markers: list[re.Match[str]] = []
    i = 0
    while i < len(line):
        top = stack[-1] if stack else None
        char = line[i]
        step = 1
        if top == SINGLE:
            if char == "'":
                stack.pop()
        elif top in (ARITH, ARITH_PAREN):
            if line.startswith("))", i) and top == ARITH:
                stack.pop()
                step = 2
            elif char == "(":
                stack.append(ARITH_PAREN)
            elif char == ")" and top == ARITH_PAREN:
                stack.pop()
        elif char == "\\":
            step = 2
        elif line.startswith("$((", i) or (top != DOUBLE and line.startswith("((", i)):
            stack.append(ARITH)
            step = 3 if char == "$" else 2
        elif line.startswith("$(", i):
            stack.append(SUBST)
            step = 2
        elif top == DOUBLE:
            if char == '"':
                stack.pop()
        elif char in "'\"":
            stack.append(char)
        elif char == "(":
            stack.append(SUBSHELL)
        elif char == ")":
            if top in CODE:
                stack.pop()
        elif line.startswith("<<<", i):
            step = 3
        elif match := HEREDOC.match(line, i):
            markers.append(match)
            step = match.end() - i
        elif char == "#" and (i == 0 or line[i - 1] in " \t;&|()<>"):
            return line[:i], markers
        i += step
    return line, markers


def reader_of(before: str, after: str) -> str | None:
    """What reads a heredoc: "shell", "python" or None (data).

    ``before`` and ``after`` are the text around the ``<<`` on its line.
    """
    words = re.split(r"[;&|(]", before)[-1].split()
    piped = PIPED_TO.search(after)
    if piped:
        words.append(piped.group(1))
    for i, word in enumerate(words):
        if basename(word) in SHELLS and all(w.startswith("-") for w in words[i + 1 :]):
            return "shell"
        if PYTHON.search(word):
            return "python"
    return None


def simple_commands(command: str) -> list[tuple[list[str], list[str]]]:
    """The words of each simple command, and its redirection targets.

    Raises:
        ValueError: If shlex cannot split the command (unbalanced quotes).
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=PUNCTUATION)
    lexer.whitespace_split = True
    lexer.whitespace = " \t\r"  # newlines separate commands
    lexer.commenters = ""  # scan_line has removed them
    commands: list[tuple[list[str], list[str]]] = [([], [])]
    target_next = False
    for token in lexer:
        words, targets = commands[-1]
        if target_next:
            targets.append(token)
            target_next = False
        elif not set(token) <= set(PUNCTUATION):
            words.append(token)
        elif set(token) & SEPARATORS or token in ("&", "&&"):
            commands.append(([], []))
        else:  # a redirection such as > or 2>&1: drop it and its fd
            if words and words[-1].isdigit():
                words.pop()
            target_next = True
    return [(words, targets) for words, targets in commands if words or targets]


def cwd_after(words: list[str], cwd: str, pushed: list[str]) -> str:
    """The working directory after the simple command ``words``.

    ``cd`` and ``pushd`` move it, and ``popd`` returns to the directory the
    last ``pushd`` left; ``pushed`` is that stack, which this updates.
    """
    if not words or words[0] not in ("cd", "pushd", "popd"):
        return cwd
    if words[0] == "popd":
        return pushed.pop() if pushed else cwd
    args = [w for w in words[1:] if w == "-" or not w.startswith("-")]
    if not args:
        return os.path.expanduser("~") if words[0] == "cd" else cwd
    if words[0] == "pushd":
        pushed.append(cwd)
    return cwd if args[0] == "-" else resolve(args[0], cwd)
