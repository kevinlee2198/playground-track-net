"""What the guard hooks next to this file share: the held-out paths and the hook I/O.

The held-out data is tennis game8-game10 and the badminton Test part
(``HELD_OUT_GAMES`` and ``HELD_OUT_PART`` in training/run.py are the code's
copy; this one stays separate because the hooks cannot import the project). A path is
judged after ``os.path.normpath`` and, when relative, after joining it to the
working directory, so ``datasets//tennis/game8``, ``tennis/../tennis/game8``
and a bare ``game8`` inside ``datasets/tennis`` all count. A simple brace
list or integer range (``game{1,2,7}``, ``game{1..7}``) is expanded first, as
bash would, and each word it makes is judged. Three things are refused:

- a path into the held-out data (``datasets/tennis/game8``, ...,
  ``datasets/badminton/Test``), whatever comes before ``datasets/``;
- a wildcard (``*``, ``?``, ``[``, a ``{`` left unexpanded) or a shell
  variable or command substitution (``$``, a backtick) in the component
  directly under ``datasets``, ``datasets/tennis`` or ``datasets/badminton``,
  which could match or name the held-out data too;
- a dataset root itself (``datasets``, ``datasets/tennis``,
  ``datasets/badminton``), since searching or reading below it reaches the
  held-out data, unless the caller says roots are fine (only programs that
  walk below a directory may not take one).

Standard library only: the hooks run on the system python3.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable

HELD_OUT = re.compile(
    r"(?<![\w.-])datasets/(?:tennis/game(?:8|9|10)|badminton/Test)(?![\w.-])"
)
WILDCARD_UNDER_ROOT = re.compile(
    r"(?<![\w.-])datasets/(?:(?:tennis|badminton)/)?[^/]*[*?[{$`]"
)
DATASET_ROOT = re.compile(r"(?:^|/)datasets(?:/tennis|/badminton)?$")
BRACES = re.compile(r"\{([^{}]*)\}")
BRACE_RANGE = re.compile(r"(-?\d+)\.\.(-?\d+)")
MAX_EXPANSION = 64  # words one brace expression may make
HELD_OUT_NAMES = "tennis game8-game10, badminton Test"
READ_INSTEAD = (
    "Read the train and val data by name instead: datasets/tennis/game1 ... "
    "datasets/tennis/game7, datasets/badminton/Professional and "
    "datasets/badminton/Amateur."
)


def resolve(word: str, cwd: str | None) -> str:
    """``word`` as a normalised path, joined to ``cwd`` when it is relative."""
    path = os.path.expanduser(word)
    if cwd and not path.startswith("/"):
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


def held_out_refusal(found: str) -> str:
    return (
        f"Blocked: {found} is held-out test data ({HELD_OUT_NAMES}), which stays "
        f"unread until the model is final. {READ_INSTEAD} To search text for "
        "the name, search for game8 (or Test) without the datasets/ path."
    )


def path_problem(
    word: str,
    cwd: str | None,
    *,
    roots_ok: bool = False,
    wildcards_ok: bool = False,
) -> str | None:
    """The refusal for ``word`` as a path (see the module docstring), or None.

    A ``--flag=value`` or ``NAME=value`` word is judged by its value as well.
    """
    candidates = [word]
    if "=" in word:
        candidates.append(word.split("=", 1)[1])
    for candidate in filter(None, candidates):
        for expanded in expand_braces(candidate) or [candidate]:
            if reason := _problem(word, resolve(expanded, cwd), roots_ok, wildcards_ok):
                return reason
    return None


def _problem(word: str, path: str, roots_ok: bool, wildcards_ok: bool) -> str | None:
    if match := HELD_OUT.search(path):
        return held_out_refusal(match.group(0))
    if not wildcards_ok and WILDCARD_UNDER_ROOT.search(path):
        return (
            f"Blocked: {word} has a wildcard or a shell variable directly under a "
            f"dataset directory, so it could match or name the held-out test data "
            f"({HELD_OUT_NAMES}) too. Name the games or parts explicitly, in a "
            "brace list if you like: datasets/tennis/game{1..7}/Clip1/Label.csv or "
            f"datasets/badminton/{{Professional,Amateur}}/... {READ_INSTEAD}"
        )
    if not roots_ok and DATASET_ROOT.search(path):
        return (
            f"Blocked: {word} is a dataset directory, and searching, copying or "
            f"reading below it reaches the held-out test data ({HELD_OUT_NAMES}). "
            f"{READ_INSTEAD} (Listing a dataset directory with ls, without -R, "
            "is fine.)"
        )
    return None


def expand_braces(word: str) -> list[str] | None:
    """The words bash's brace expansion makes of ``word``, or None.

    Only a brace holding a comma list (``{a,b}``) or an integer range
    (``{1..7}``) of plain text is expanded, as many in a row as the word holds.
    None for anything else (a nested brace, a ``$`` or backtick inside one, an
    unclosed one, more than ``MAX_EXPANSION`` words): the word is then judged
    as it stands. A word without a brace is itself.
    """
    start = word.find("{")
    if start < 0:
        return [word]
    match = BRACES.match(word, start)
    if not match or any(c in match.group(1) for c in "$`"):
        return None
    items = _brace_items(match.group(1))
    tails = expand_braces(word[match.end() :])
    if items is None or tails is None or len(items) * len(tails) > MAX_EXPANSION:
        return None
    return [word[:start] + item + tail for item in items for tail in tails]


def _brace_items(text: str) -> list[str] | None:
    if bounds := BRACE_RANGE.fullmatch(text):
        first, last = int(bounds.group(1)), int(bounds.group(2))
        if abs(last - first) >= MAX_EXPANSION:
            return None
        step = 1 if last >= first else -1
        return [str(n) for n in range(first, last + step, step)]
    return text.split(",") if "," in text else None


def run_hook(name: str, check: Callable[[dict], str | None]) -> int:
    """Run a PreToolUse hook: ``check`` gets the hook input and returns a refusal.

    The hook input is Claude Code's JSON on stdin (``tool_name``,
    ``tool_input``, ``cwd``, ...). Returns the exit code: 0 lets the tool call
    run; 2 blocks it, with the refusal on stderr for the model to read. Any
    error inside the hook blocks too, naming the error, so a broken guard
    never waves a call through.
    """
    try:
        reason = check(json.load(sys.stdin))
    except Exception as err:
        reason = (
            f"Blocked: the {name} hook failed ({type(err).__name__}: {err}), so "
            "it could not check this call. Tell the owner; .claude/hooks/ holds it."
        )
    if reason is None:
        return 0
    print(reason, file=sys.stderr)
    return 2
