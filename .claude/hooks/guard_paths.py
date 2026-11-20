"""PreToolUse hook for Read, Glob and Grep: keep the held-out test data unread.

- Read: blocks a ``file_path`` into the held-out data (see held_out.py).
- Glob only lists names, so it may search a dataset root: it is blocked only
  when its ``pattern``, its ``path`` or the two joined name held-out data.
- Grep reads the files it searches: its ``path``, and ``path`` and ``glob``
  joined, may not name held-out data, a dataset root (a search there reads
  the held-out labels too) or a wildcard directly under a dataset directory.
  Its ``pattern`` is a regex, not a path, and is not checked.

Relative paths are resolved against the hook input's ``cwd``. A backstop
against an honest mistake, not a security boundary: the code checks in
training/run.py are the real guard. Standard library only (system python3).
"""

from __future__ import annotations

import os
import sys

try:
    from held_out import path_problem, run_hook
except Exception as err:  # held_out.py missing or broken: block, never crash
    print(
        f"Blocked: guard_paths.py could not import held_out.py ({err!r}).",
        file=sys.stderr,
    )
    sys.exit(2)


def check_call(tool_name: str, tool_input: dict, cwd: str | None = None) -> str | None:
    """The refusal for this Read, Glob or Grep call, or None to let it run."""
    cwd = cwd or os.getcwd()

    def text(field: str) -> str | None:
        value = tool_input.get(field)
        return value if isinstance(value, str) and value else None

    if tool_name in ("Grep", "Glob"):
        path = text("path") or "."
        extra = text("glob" if tool_name == "Grep" else "pattern")
        words = [path] if extra is None else [path, os.path.join(path, extra)]
        lenient = tool_name == "Glob"
    else:  # Read, and any other tool the matcher sends
        words = [w for w in (text("file_path"), text("path")) if w]
        lenient = True
    for word in words:
        if reason := path_problem(word, cwd, roots_ok=lenient, wildcards_ok=lenient):
            return reason
    return None


def check_payload(payload: dict) -> str | None:
    return check_call(payload["tool_name"], payload["tool_input"], payload.get("cwd"))


if __name__ == "__main__":
    sys.exit(run_hook("guard_paths.py", check_payload))
