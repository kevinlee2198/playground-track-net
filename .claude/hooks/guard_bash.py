"""PreToolUse hook for Bash: held-out data, the owner's --final, and the memory cap.

A backstop against an honest mistake, not a security boundary. A helper script
that calls evaluate with --final, or a command spelled in a way this parser
does not follow (backticks and ``$(...)`` are not run as commands), gets past
it; the real guards for the test split are the code checks in training/run.py
(``require_final``, ``require_final_for_path``). Standard library only: it
runs on the system python3.

The command is read as shell_words.py describes: comments dropped, heredoc
bodies set aside, and the rest split into simple commands. A ``cd`` or
``pushd`` moves the working directory (the hook input's ``cwd`` to begin
with) that the later simple commands' relative paths are resolved against,
and ``popd`` moves it back. A heredoc body fed to a shell, and the string
given to ``bash -c`` (``sh -c``, ...), are checked as commands; a body fed to
Python is checked for held-out paths.

Each simple command is judged on its own:

1. Held-out data (held_out.py). No word may be a path into it, or have a
   wildcard, ``$`` or backtick directly under a dataset directory (simple
   brace lists are expanded first). A dataset root is refused only as a path
   that a program walking below a directory is given: the grep family
   (``grep``, ``egrep``, ``fgrep``, ``rg``, ``ag``, ``fd``, ``git grep``),
   ``find``, ``tree``, ``tar``, ``zip``, ``cp``, ``rsync``, ``ls -R``, and
   Python (other than ``main.py extract-frames`` and ``label_vs_ball.py``,
   which take a root and refuse the held-out parts themselves). For the grep
   family, the pattern (the first word that is not an option, or the value of
   ``-e``) and the values of ``--exclude*``, ``--include*``, ``--ignore*`` and
   ``-g``/``--glob`` are not paths and are not checked at all; for ``find``,
   only the words before the first one starting with ``-`` are its paths. A
   word with whitespace in it is prose, not a path, except the code given to
   ``python -c``, whose pieces are checked like words.
2. ``main.py evaluate``, ``main.py check-data``, ``main.py infer``,
   ``prediction_offset.py`` and ``label_vs_ball.py`` never get ``--final`` or
   the test split: the owner runs those commands.
3. The memory cap: Python run through the project environment (``uv run``, an
   interpreter under a ``venv/bin/``, or a bare ``python`` after ``source
   .../bin/activate`` in the same command), ``uv run torchrun``, ``jupyter``
   and ``ipython``, a Python script file run by any interpreter, and every
   pytest run go through ``scripts/capped.sh``, or ``systemd-run`` with a
   ``MemoryMax=`` property, before the interpreter. Exempt: ``main.py
   train``, ``main.py extract-frames``, ``main.py`` with ``--help`` or ``-h``,
   the interpreter with ``--version``, ``-V``, ``--help`` or ``-h``, the hook
   scripts in .claude/hooks/, and the system python3 with ``-c``, ``-``,
   ``-m`` (other than pytest and main) or stdin, since it has no numpy, torch
   or OpenCV.
4. pytest runs serially (no ``-n``, also inside ``-qn4``, or
   ``--numprocesses``) and one run at a time: it is refused while another
   pytest process (other than ``--collect-only``) is running.

The interpreter, pytest or script is looked for past prefixes (``uv run``,
``timeout``, ``env``, ``xargs``, ...): once one has been seen, options are
skipped, and so is a word straight after an option, as that option's value.

A command that cannot be split (unbalanced quotes) gets a rougher check of the
same rules on its raw text.
"""

from __future__ import annotations

import os
import re
import shlex
import sys

try:
    from held_out import DATASET_ROOT, HELD_OUT, path_problem, resolve, run_hook
    from shell_words import (
        PYTHON,
        SHELLS,
        basename,
        cwd_after,
        simple_commands,
        split_heredocs,
    )
except Exception as err:  # a module missing or broken: block, never crash
    print(
        f"Blocked: guard_bash.py could not import its modules ({err!r}).",
        file=sys.stderr,
    )
    sys.exit(2)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAPPED = "scripts/capped.sh"  # relative to REPO

# Programs that run the command after them, matched on the basename (uv only
# as ``uv run``).
PREFIXES = {
    *("uv", "uvx", "env", "nohup", "setsid", "nice", "ionice", "time", "timeout"),
    *("exec", "command", "stdbuf", "sudo", "systemd-run", "xargs", "capped.sh"),
}
KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{"}
# Rule 1: programs that walk below a directory, so may not take a dataset root.
WALKERS = {"tree", "tar", "zip", "cp", "rsync"}
SEARCHERS = {"grep", "egrep", "fgrep", "rg", "ag", "fd"}  # a pattern, then paths
SEARCH_PATTERN = {"-e", "--regexp"}  # its value is the pattern
SEARCH_GLOBS = ("--exclude", "--include", "--ignore", "--glob")  # values: globs
SEARCH_VALUES = {"-A", "-B", "-C", "-m", "-f", "--file", "-t", "--type"}
RECURSIVE_LS = re.compile(r"-[A-Za-z1]*R[A-Za-z1]*|--recursive")
ROOT_SCRIPTS = {"label_vs_ball.py"}
ROOT_MAIN = {"extract-frames"}
# Rule 2: the owner's commands, and what to do instead.
OWNER_SCRIPTS = {
    "prediction_offset.py": "leave --final out and use the val (or train) split",
    "label_vs_ball.py": "leave --final out and name training or validation games "
    "(game1 ... game7, Professional, Amateur)",
}
OWNER_MAIN = {
    "evaluate": "leave --final out and use --split val",
    "check-data": "leave --final out (check-data then covers train and val)",
    "infer": "leave --final out and run it on a video that is not held-out data",
}
# Rule 3: exempt from the cap, and what needs it.
UNCAPPED_MAIN = {"train", "extract-frames"}
HELP = {"--help", "-h"}
INTERPRETER_INFO = {"--version", "-V", "-VV", "--help", "-h"}
PROJECT_TOOLS = {"torchrun", "jupyter", "ipython"}  # under uv run

PROJECT_PYTHON = re.compile(r"(?:^|/)\.?venv/bin/python")
PYTEST = re.compile(r"(?:^|/)(?:pytest|py\.test)$")
PARALLEL = re.compile(r"-([qvxsl]*)n(?:\d+|auto|logical)?")  # -n4, -qn 4
HOOK_SCRIPT = re.compile(r"(?:^|/)\.claude/hooks/[\w-]+\.py$")
SHELL_CODE_FLAG = re.compile(r"-[A-Za-z]*c[A-Za-z]*")  # -c, -lc, -ec
ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
NUMBER = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")  # nice -n 10, timeout 5m
CODE_PIECES = re.compile(r"[\s'\"`(),;]+")

OUT_OF_MEMORY = "running out of memory shuts down the whole WSL session"
HELD_OUT_CWD = (
    "Blocked: the working directory {cwd} is inside the held-out test data. "
    "cd to the repository root first."
)
OWNER_REFUSAL = (
    "Blocked: {what} with --final or the test split reads the held-out test data. "
    "Test-split commands are the owner's to run: {instead}, or ask the owner to "
    "run it."
)
CAP_REFUSAL = (
    "Blocked: Python in the project environment, and any Python script, runs "
    f"under the memory cap, because {OUT_OF_MEMORY}. Run this instead: "
    "{suggestion}"
)
PYTEST_REFUSAL = (
    "Blocked: pytest runs only under the memory cap and serially "
    f"({OUT_OF_MEMORY}; -n 4 holds about 8 GB). Run this instead, one run at a "
    "time: {suggestion}"
)
PYTEST_RUNNING = (
    "Blocked: another pytest run is going (pid {pid}: {cmdline}). Tests run one "
    f"at a time, since two at once can run out of memory, and {OUT_OF_MEMORY}: "
    "wait for pid {pid} to finish, then run this command again. If it looks "
    "stuck, check it with ps -o etime=,stat= -p {pid}: if it is stopped (T) or "
    "far older than the two minutes the suite takes, tell the owner. Do not "
    "kill it, and do not wait forever."
)
UNSPLIT_NOTE = " (The command could not be split into words: unbalanced quotes?)"


def check_payload(payload: dict) -> str | None:
    """The refusal for this Bash hook input, or None to let the call run."""
    return check_command(payload["tool_input"]["command"], payload.get("cwd"))


def check_command(command: str, cwd: str | None = None) -> str | None:
    """The refusal for this Bash command run in ``cwd``, or None to let it run."""
    cwd = resolve(cwd or os.getcwd(), None)
    outer, bodies = split_heredocs(command)
    try:
        commands = simple_commands(outer.replace("\\\n", " "))
    except ValueError:  # unbalanced quotes
        return rough_check(command, cwd)
    activated = False  # source .venv/bin/activate has run
    pushed: list[str] = []  # pushd's directory stack
    for words, targets in commands:
        if reason := check_paths(words, targets, cwd) or check_run(
            words, cwd, activated
        ):
            return reason
        activated = activated or activates(words)
        cwd = cwd_after(words, cwd, pushed)
    for body, reader in bodies:
        if reader == "shell":
            reason = check_command(body, cwd)
        else:
            reason = check_code(body, cwd)
        if reason:
            return reason
    return None


def activates(words: list[str]) -> bool:
    """Whether ``words`` is ``source`` (or ``.``) of a virtualenv's activate."""
    return (
        len(words) > 1
        and words[0] in ("source", ".")
        and words[1].endswith("bin/activate")
    )


def find_program(words: list[str]) -> tuple[int | None, bool]:
    """Where the program is that ``words`` runs past its prefixes, if anywhere,
    and whether ``uv run`` runs it."""
    project_env = False  # seen uv run
    prefixed = False  # seen a prefix: skip its options and their values
    value_next = False  # the word after an option may be its value
    last_prefix = ""
    for i, word in enumerate(words):
        name = basename(word)
        if PYTHON.search(word) or PYTEST.search(word) or word.endswith(".py"):
            return i, project_env
        if name in SHELLS:
            return i, project_env
        if name in PREFIXES or (name == "run" and last_prefix == "uv"):
            project_env = project_env or name == "run"
            prefixed, value_next, last_prefix = True, False, name
        elif ASSIGNMENT.match(word) or word in KEYWORDS:
            continue
        elif not prefixed:
            return i, project_env
        elif word.startswith("-"):
            value_next = "=" not in word
        elif value_next or NUMBER.match(word):
            value_next = False
        else:
            return i, project_env
    return None, project_env


# Rule 1: held-out data ------------------------------------------------------


def check_paths(words: list[str], targets: list[str], cwd: str) -> str | None:
    """The refusal for a held-out path among ``words`` and ``targets``, or None."""
    first = program_index(words)
    if HELD_OUT.search(cwd) and (
        first is None or words[first] not in ("cd", "pushd", "popd")
    ):
        return HELD_OUT_CWD.format(cwd=cwd)
    program, _ = find_program(words)
    walked, skipped, walks_cwd = path_roles(words, program)
    if walks_cwd and DATASET_ROOT.search(cwd):
        return path_problem(".", cwd)
    code = code_index(words, program)
    shell = program is not None and basename(words[program]) in SHELLS
    for i, word in enumerate(words):
        if i in skipped:
            continue
        if i == code:
            if shell:
                continue  # check_run checks it as a command
            reason = check_code(word, cwd)  # python -c
        else:
            # The program's own name is not resolved against cwd.
            where = None if i in (first, program) else cwd
            reason = check_word(word, where, roots_ok=i not in walked)
        if reason:
            return reason
    for target in targets:
        if reason := check_word(target, cwd, roots_ok=True):
            return reason
    return None


def program_index(words: list[str]) -> int | None:
    """The first word that is no assignment or keyword."""
    for i, word in enumerate(words):
        if not ASSIGNMENT.match(word) and word not in KEYWORDS:
            return i
    return None


def path_roles(words: list[str], program: int | None) -> tuple[set, set, bool]:
    """Which words are paths a walking program reads below (see rule 1), which
    are no paths at all (a search pattern or glob), and whether the program
    walks the working directory, having been given no path."""
    if program is None:
        return set(), set(), False
    name = basename(words[program])
    rest = range(program + 1, len(words))
    if name == "git" and subcommand(words[program + 1 :]) == "grep":
        return search_roles(words, words.index("grep", program + 1))
    if name in SEARCHERS:
        return search_roles(words, program)
    if name == "find":
        walked = set()
        for k in rest:
            if words[k].startswith(("-", "(", "!")):
                break
            walked.add(k)
        return walked, set(), not walked
    if (
        name in WALKERS
        or (name == "ls" and any(RECURSIVE_LS.fullmatch(words[k]) for k in rest))
        or (python_run(words, program) and not takes_root(words))
    ):
        return set(rest), set(), name == "tree" and len(rest) == 0
    return set(), set(), False


def search_roles(words: list[str], start: int) -> tuple[set, set, bool]:
    """``path_roles`` for the grep-family program at ``words[start]``."""
    walked: set[int] = set()
    skipped: set[int] = set()
    pattern = False  # the pattern has been given
    options = True  # before --
    recursive = basename(words[start]) not in ("grep", "egrep", "fgrep")
    value = None  # what the next word is: "skip" (a glob, the pattern), "keep"
    for k in range(start + 1, len(words)):
        word = words[k]
        if value:
            if value == "skip":
                skipped.add(k)
            value = None
        elif options and word == "--":
            options = False
        elif options and word.startswith("-") and word != "-":
            recursive = recursive or bool(
                re.fullmatch(r"-\w*[rR]\w*|--recursive", word)
            )
            if word in SEARCH_PATTERN or word.startswith("--regexp="):
                pattern = True
                value = "skip" if "=" not in word else None
                skipped.add(k)
            elif word == "-g" or word.startswith(SEARCH_GLOBS):
                skipped.add(k)
                value = "skip" if word == "-g" or "=" not in word else None
            elif word in SEARCH_VALUES:
                pattern = pattern or word in ("-f", "--file")  # patterns from a file
                value = "keep"
        elif not pattern:
            pattern = True
            skipped.add(k)
        else:
            walked.add(k)
    return walked, skipped, recursive and not walked


def python_run(words: list[str], program: int) -> bool:
    word = words[program]
    return bool(PYTHON.search(word) or PYTEST.search(word) or word.endswith(".py"))


def takes_root(words: list[str]) -> bool:
    """Whether the command runs a script that takes a dataset root (rule 1)."""
    for i, word in enumerate(words):
        if basename(word) in ROOT_SCRIPTS:
            return True
        is_main = basename(word) == "main.py" or (
            word == "main" and words[i - 1] == "-m"
        )
        if is_main and subcommand(words[i + 1 :]) in ROOT_MAIN:
            return True
    return False


def code_index(words: list[str], program: int | None) -> int | None:
    """Where the code is that a Python interpreter or shell gets with -c."""
    if program is None:
        return None
    if not (PYTHON.search(words[program]) or basename(words[program]) in SHELLS):
        return None
    for k in range(program + 1, len(words)):
        if SHELL_CODE_FLAG.fullmatch(words[k]):
            return k + 1
        if not words[k].startswith(("-", "+")) and words[k - 1] not in (
            *("-o", "+o", "-X", "-W"),
        ):
            return None
    return None


def check_word(word: str, cwd: str | None, roots_ok: bool) -> str | None:
    """``path_problem`` for ``word``, unless it has whitespace in it (prose)."""
    if any(char.isspace() for char in word):
        return None
    return path_problem(word, cwd, roots_ok=roots_ok)


def check_code(text: str, cwd: str) -> str | None:
    """The refusal for a held-out path in the code or commands ``text``, or None.

    Each piece between whitespace, quotes, brackets and commas is judged as a
    path; a bare ``datasets`` (no ``/``) is taken for a name, not a root.
    """
    for piece in filter(None, CODE_PIECES.split(text)):
        if reason := path_problem(piece, cwd, roots_ok="/" not in piece):
            return reason
    return None


# Rules 2 to 4: what runs ----------------------------------------------------


def check_run(words: list[str], cwd: str, activated: bool = False) -> str | None:
    """The refusal for what the simple command ``words`` runs, or None.

    ``activated``: a virtualenv was activated earlier in the command, so a
    bare ``python`` is the project environment's.
    """
    i, project_env = find_program(words)
    if i is None:
        return None
    word = words[i]
    name = basename(word)
    if PYTHON.search(word):
        in_project = (
            project_env
            or bool(PROJECT_PYTHON.search(word))
            or (activated and "/" not in word)
        )
        return check_python(words, i, in_project, cwd)
    if PYTEST.search(word):
        return check_pytest(words, i, words[i + 1 :], cwd)
    if word.endswith(".py"):  # uv run main.py, ./script.py
        return check_script(words, i, word, words[i + 1 :], cwd)
    if name in SHELLS:
        code = shell_code(words, i)
        return None if code is None else check_command(code, cwd)
    if project_env and name in PROJECT_TOOLS and not is_capped(words, i):
        return cap_refusal(words, cwd)
    return None


def shell_code(words: list[str], i: int) -> str | None:
    """The string ``bash -c`` at ``words[i]`` runs, or None if it runs a file."""
    k = code_index(words, i)
    return words[k] if k is not None and k < len(words) else None


def check_python(words: list[str], i: int, project_env: bool, cwd: str) -> str | None:
    """Judge ``words[i]``, a Python interpreter, by what it runs."""
    j = i + 1
    while j < len(words):
        arg = words[j]
        if arg in INTERPRETER_INFO:
            return None
        if arg.startswith("-m"):
            module, module_args = module_run(words, j)
            if module == "pytest":
                return check_pytest(words, i, words[module_args:], cwd)
            if module == "main":
                return check_script(words, i, "main.py", words[module_args:], cwd)
            break
        if arg == "-" or arg.startswith("-c"):
            break
        if arg in ("-X", "-W"):
            j += 2
        elif arg.startswith("-"):
            j += 1
        else:
            return check_script(words, i, arg, words[j + 1 :], cwd)
    # -c, -m, stdin or a REPL: only the project environment needs the cap.
    if project_env and not is_capped(words, i):
        return cap_refusal(words, cwd)
    return None


def module_run(args: list[str], j: int) -> tuple[str, int]:
    """The module ``args[j]`` (``-m pytest`` or ``-mpytest``) runs, and where
    the module's own arguments start."""
    if args[j] == "-m":
        return word_after(args, j), j + 2
    return args[j][2:], j + 1


def check_script(
    words: list[str], interpreter: int, script: str, args: list[str], cwd: str
) -> str | None:
    """Judge a run of the script ``script`` with arguments ``args``."""
    name = basename(script)
    command = subcommand(args)
    final = any(abbreviates(a, "--final") for a in args)
    if name == "main.py" and command in OWNER_MAIN and (final or selects_test(args)):
        return OWNER_REFUSAL.format(
            what=f"main.py {command}", instead=OWNER_MAIN[command]
        )
    if name in OWNER_SCRIPTS and (final or "test" in args):
        return OWNER_REFUSAL.format(what=name, instead=OWNER_SCRIPTS[name])
    if name == "main.py" and (command in UNCAPPED_MAIN or HELP.intersection(args)):
        return None
    if HOOK_SCRIPT.search(script) or is_capped(words, interpreter):
        return None
    return cap_refusal(words, cwd)


def check_pytest(words: list[str], i: int, args: list[str], cwd: str) -> str | None:
    """Judge a pytest run started at ``words[i]`` with arguments ``args``."""
    serial: list[str] = []
    parallel = False
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif match := PARALLEL.fullmatch(arg) or is_parallel(arg):
            parallel = True
            skip = arg.endswith("n") or (arg.startswith("--") and "=" not in arg)
            if isinstance(match, re.Match) and match.group(1):
                serial.append("-" + match.group(1))  # -qn4: keep the -q
        else:
            serial.append(arg)
    parallel = parallel or any(
        is_parallel(option)
        for word in words[:i]
        if word.startswith("PYTEST_ADDOPTS=")
        for option in word.split("=", 1)[1].split()
    )
    if parallel or not is_capped(words, i):
        suggestion = shlex.join([wrapper(cwd), "uv", "run", "pytest", *serial])
        return PYTEST_REFUSAL.format(suggestion=suggestion)
    if running := running_pytest():
        pid, cmdline = running[0]
        return PYTEST_RUNNING.format(pid=pid, cmdline=cmdline[:200])
    return None


def running_pytest(proc: str = "/proc") -> list[tuple[int, str]]:
    """(pid, command line) of each pytest process running, other than this one
    and any that only collects tests.

    Empty when ``proc`` cannot be read: an unreadable /proc never blocks.
    """
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    found = []
    for entry in entries:
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(os.path.join(proc, entry, "cmdline"), "rb") as f:
                argv = [a for a in f.read().decode(errors="replace").split("\0") if a]
        except OSError:
            continue  # gone, or not ours to read
        if runs_pytest(argv) and not {"--collect-only", "--co"}.intersection(argv):
            found.append((int(entry), " ".join(argv)))
    return sorted(found)


def runs_pytest(argv: list[str]) -> bool:
    """Whether a process with this argv is pytest (``pytest``, ``python -m pytest``)."""
    if not argv:
        return False
    if PYTEST.search(argv[0]):
        return True
    if not PYTHON.search(argv[0]):
        return False
    k = 1
    while k < len(argv):
        arg = argv[k]
        if arg.startswith("-m"):
            return module_run(argv, k)[0] == "pytest"
        if not arg.startswith("-"):  # the script: .venv/bin/pytest
            return bool(PYTEST.search(arg))
        k += 2 if arg in ("-X", "-W") else 1
    return False


def is_parallel(arg: str) -> bool:
    """Whether ``arg`` is pytest-xdist's ``-n`` (also in ``-qn4``) or
    ``--numprocesses`` (or a prefix)."""
    flag = arg.split("=", 1)[0]
    if PARALLEL.fullmatch(flag):
        return True
    return len(flag) >= len("--num") and "--numprocesses".startswith(flag)


def is_capped(words: list[str], i: int) -> bool:
    """Whether the words before ``words[i]`` put it under a memory cap."""
    before = words[:i]
    if any(w.endswith("capped.sh") for w in before):
        return True
    systemd_run = any(basename(w) == "systemd-run" for w in before)
    return systemd_run and any("MemoryMax=" in w for w in before)


def wrapper(cwd: str) -> str:
    """scripts/capped.sh by a path that works from ``cwd``."""
    return CAPPED if cwd == REPO else os.path.join(REPO, CAPPED)


def cap_refusal(words: list[str], cwd: str) -> str:
    """The cap refusal, suggesting ``words`` with the wrapper after assignments."""
    n = 0
    while n < len(words) and ASSIGNMENT.match(words[n]):
        n += 1
    suggestion = shlex.join([*words[:n], wrapper(cwd), *words[n:]])
    return CAP_REFUSAL.format(suggestion=suggestion)


def subcommand(args: list[str]) -> str | None:
    """The first argument that is not an option."""
    return next((a for a in args if not a.startswith("-")), None)


def word_after(words: list[str], j: int) -> str:
    """``words[j + 1]``, or "" if ``words[j]`` is the last word."""
    return words[j + 1] if j + 1 < len(words) else ""


def abbreviates(arg: str, flag: str) -> bool:
    """Whether argparse would read ``arg`` as ``flag`` (it accepts prefixes)."""
    return len(arg) > 2 and flag.startswith(arg)


def selects_test(args: list[str]) -> bool:
    """Whether ``args`` hold ``--split test`` or ``--split=test`` (or a prefix)."""
    for j, arg in enumerate(args):
        flag, eq, value = arg.partition("=")
        if (
            abbreviates(flag, "--split")
            and (value if eq else word_after(args, j)) == "test"
        ):
            return True
    return False


# A command shlex cannot split ------------------------------------------------


def rough_check(command: str, cwd: str) -> str | None:
    """The rules on the raw text of a command shlex cannot split."""
    if reason := check_code(command, cwd):
        return reason + UNSPLIT_NOTE
    capped = "capped.sh" in command or "MemoryMax=" in command
    pytest = re.search(r"(?<![\w.-])(?:pytest|py\.test)\b", command)
    parallel = re.search(r"\s(?:-[qvxsl]*n\s*(?:\d|auto|logical)|--num)", command)
    if pytest and (parallel or not capped):
        suggestion = f"{wrapper(cwd)} uv run pytest <files> -q"
        return PYTEST_REFUSAL.format(suggestion=suggestion) + UNSPLIT_NOTE
    owner = re.search(
        r"main\.py\s+(?:evaluate|check-data|infer)|prediction_offset\.py"
        r"|label_vs_ball\.py",
        command,
    )
    test_split = re.search(
        r"--final\b|--split[=\s]+test\b|prediction_offset\.py\s.*\stest\b", command
    )
    if owner and test_split:
        return (
            OWNER_REFUSAL.format(what=owner.group(0), instead="leave --final out")
            + UNSPLIT_NOTE
        )
    script = re.search(
        r"(?:^|[\s/])python[\d.]*\s+(?:-\S+\s+)*[^\s-]\S*\.py\b", command
    )
    project = re.search(
        r"\buv\s+run\b[^;&|]*(?:\bpython|\.py\b)|venv/bin/python", command
    )
    uncapped_main = re.search(r"main\.py\s+(?:train|extract-frames)\b", command)
    if (script or project) and not capped and not uncapped_main:
        suggestion = f"{wrapper(cwd)} <the command>"
        return CAP_REFUSAL.format(suggestion=suggestion) + UNSPLIT_NOTE
    return None


if __name__ == "__main__":
    sys.exit(run_hook("guard_bash.py", check_payload))
