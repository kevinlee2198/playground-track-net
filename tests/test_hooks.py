"""The guard hooks in .claude/hooks/: held-out paths, --final and the memory cap."""

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
HOOKS = REPO / ".claude" / "hooks"
WRAPPER = str(REPO / "scripts" / "capped.sh")  # as refusals outside REPO name it
sys.path.insert(0, str(HOOKS))
guard_bash = importlib.import_module("guard_bash")
guard_paths = importlib.import_module("guard_paths")

# The hooks must run on the system python3 (3.12 here), not only the venv's.
SYSTEM_PYTHON = (
    "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
)
CWD = "/repo"  # the working directory the commands below run in
INFER = "main.py infer --video a.mp4 --model m.pt --output o.csv"
running_pytest = guard_bash.running_pytest  # before no_other_pytest replaces it


@pytest.fixture(autouse=True)
def no_other_pytest(monkeypatch):
    """This suite is itself a pytest run: by default, pretend no other is."""
    monkeypatch.setattr(guard_bash, "running_pytest", lambda: [])


ALLOW = None  # the expected refusal of a command that runs

# Each table below is one rule of guard_bash.py (see its docstring): a command,
# and None if it runs or a phrase its refusal must contain.

# Rule 1: no word is a path into tennis game8-game10 or badminton Test (after
# expanding simple braces, normalising it and joining it to the cwd, which cd
# moves), or has a wildcard, $ or backtick directly under a dataset directory.
# A dataset root is refused only as a path given to a program that walks below
# a directory (the grep family, find, tree, tar, zip, cp, rsync, ls -R, Python
# other than extract-frames and label_vs_ball.py); a grep pattern and its
# exclude and include globs are no paths. A word with whitespace in it is
# prose, not a path.
HELD_OUT_PATHS = [
    ("ls datasets/tennis/game1 datasets/tennis/game7", ALLOW),
    ("ls datasets/tennis/game100", ALLOW),
    ("git grep -n game8 docs/", ALLOW),
    ('git commit -m "docs: datasets/badminton/Test stays held out"', ALLOW),
    ('grep -rn "datasets/tennis/game8" docs/', ALLOW),  # a pattern, not a path
    ("ls datasets/tennis/game8", "held-out test data"),
    ("ls datasets/tennis/game10/Clip1", "held-out test data"),
    ("ls datasets//tennis/game8", "held-out test data"),
    ("cat datasets/tennis/../tennis/game9/Clip1/Label.csv", "held-out test data"),
    ("cat < datasets/badminton/Test/match1/csv/1_01_01_ball.csv", "held-out test data"),
    (
        "scripts/capped.sh uv run python x.py /home/u/repo/datasets/badminton/Test/match1",
        "held-out test data",
    ),
    ("grep -n x datasets/tennis/game8/Clip1/Label.csv", "held-out test data"),
    ("cd datasets/tennis && ls game8", "held-out test data"),
    ("cd datasets/tennis && pushd game1 && popd && cat game8/x", "held-out test data"),
    ("cd datasets/tennis && pushd game1 && popd && cat game1/x", ALLOW),
    ("cd datasets && cat tennis/game8/Clip1/Label.csv", "held-out test data"),
    # Python code given to -c or fed as a heredoc is searched for paths.
    (
        "python3 -c \"open('datasets/tennis/game9/Clip1/Label.csv')\"",
        "held-out test data",
    ),
    (
        "python3 -c \"import csv; open('datasets/tennis/game9/Clip1/Label.csv')\"",
        "held-out test data",
    ),
    (
        "python3 - <<'EOF'\nprint(open('datasets/tennis/game8/Clip1/Label.csv').read())"
        "\nEOF",
        "held-out test data",
    ),
    # Wildcards, shell variables and command substitution; simple braces are
    # expanded instead.
    ("wc -l datasets/tennis/game*/*/Label.csv", "wildcard"),
    ("head -3 datasets/tennis/game[89]/Clip1/Label.csv", "wildcard"),
    ("cat datasets/badminton/*/match1/csv/1_01_01_ball.csv", "wildcard"),
    (
        "for g in 1 2 3 4 5 6 7 8 9 10; do wc -l datasets/tennis/game$g/*/Label.csv;"
        " done",
        "game{1..7}",
    ),
    (
        "for p in Professional Amateur Test; do ls datasets/badminton/$p | head; done",
        "shell variable",
    ),
    ("G=game8; cat datasets/tennis/$G/Clip1/Label.csv", "shell variable"),
    ("cat datasets/tennis/`echo game8`/Clip1/Label.csv", "shell variable"),
    ("cat datasets/tennis/$(echo game8)/Clip1/Label.csv", "shell variable"),
    ("wc -l datasets/tennis/game{1..7}/Clip1/Label.csv", ALLOW),
    ("wc -l datasets/tennis/game{1,2,7}/Clip1/Label.csv", ALLOW),
    ("ls datasets/badminton/{Professional,Amateur}/match1", ALLOW),
    ("wc -l datasets/tennis/game{7..9}/Clip1/Label.csv", "held-out test data"),
    ("wc -l datasets/tennis/game{1,8}/Clip1/Label.csv", "held-out test data"),
    ("wc -l datasets/tennis/game{1,{2,3}}/Clip1/Label.csv", "wildcard"),  # nested
    ("wc -l datasets/tennis/game{1,$N}/Clip1/Label.csv", "wildcard"),
    ("wc -l datasets/tennis/game{1..100}/Clip1/Label.csv", "wildcard"),  # too many
    # Dataset roots: fine as a name or for a program that does not walk below.
    ("ls datasets/", ALLOW),
    ("du -sh datasets/tennis", ALLOW),
    ("mkdir -p datasets/badminton", ALLOW),
    ("cd datasets/tennis && ls game1", ALLOW),
    ("uv run python main.py extract-frames --root datasets/badminton", ALLOW),
    (
        "scripts/capped.sh .venv/bin/python -I scripts/analysis/label_vs_ball.py"
        " datasets/tennis game1",
        ALLOW,
    ),
    ("grep -rn datasets docs/", ALLOW),
    ('grep -n "datasets" CLAUDE.md', ALLOW),
    ('grep -n "datasets/tennis" configs/*.yaml', ALLOW),
    ('grep -rn --exclude-dir=datasets "def main" .', ALLOW),
    ("grep -rn foo . --exclude-dir datasets --include='*.py'", ALLOW),
    ("rg -n game7 -g '!datasets' .", ALLOW),
    ("git grep -n datasets -- '*.py'", ALLOW),
    ("find . -path ./datasets -prune -o -name '*.py' -print", ALLOW),
    ("ln -s ../playground-track-net/datasets datasets", ALLOW),
    ("readlink -f datasets", ALLOW),
    ("git check-ignore -v datasets", ALLOW),
    ("git log --oneline -- datasets/", ALLOW),
    ("uv run ruff check --exclude datasets .", ALLOW),
    ("ls -la | grep datasets", ALLOW),
    ("echo datasets/ >> .gitignore", ALLOW),
    ("bash -c 'ls datasets/tennis'", ALLOW),
    ("grep -c ',1,' -r datasets/tennis", "dataset directory"),
    ("grep -e foo -r datasets/tennis", "dataset directory"),
    ("find datasets/badminton -name '*_ball.csv' | xargs wc -l", "dataset directory"),
    ("grep -rl x /home/u/repo/datasets/", "dataset directory"),
    ("cd datasets/tennis && grep -r 1280 .", "dataset directory"),
    ("cd datasets/tennis && rg 1280", "dataset directory"),  # rg walks the cwd
    ("rg foo datasets", "dataset directory"),
    ("tree datasets/tennis", "dataset directory"),
    ("ls -R datasets/tennis | head", "dataset directory"),
    ("cp -r datasets/badminton /x", "dataset directory"),
    ("tar czf /tmp/t.tgz datasets/tennis", "dataset directory"),
    ("bash -c 'grep -r x datasets/tennis'", "dataset directory"),
    (
        "scripts/capped.sh uv run python -I /s/label_speeds.py datasets/tennis game1",
        "dataset directory",
    ),
    (
        "scripts/capped.sh uv run python x.py --root=datasets/tennis",
        "dataset directory",
    ),
]

# Rule 2: main.py evaluate, check-data, infer and label, prediction_offset.py and
# label_vs_ball.py never get --final (or a prefix of it) or the test split.
OWNER_ONLY = [
    (
        "scripts/capped.sh uv run python main.py infer --video a.mp4 --model m.pt"
        " --output o.csv --final",
        "owner",
    ),
    ("scripts/capped.sh uv run python -m main evaluate --split test", "--split val"),
    (
        "scripts/capped.sh uv run python main.py check-data --out /tmp/o --final",
        "covers",
    ),
    ('git commit -m "docs: evaluate --split test needs --final"', ALLOW),
    (
        "scripts/capped.sh uv run python main.py evaluate --config configs/default.yaml"
        " --weights checkpoints/tennis_v2_noflip/best.pt --split val",
        ALLOW,
    ),
    (
        "scripts/capped.sh uv run python main.py check-data --config c.yaml --out /tmp/o",
        ALLOW,
    ),
    (
        "scripts/capped.sh uv run python main.py evaluate --config c.yaml --weights w.pt"
        " --split test",
        "owner",
    ),
    (
        "scripts/capped.sh uv run python main.py evaluate --config c.yaml --weights w.pt"
        " --split test --final",
        "owner",
    ),
    (
        "scripts/capped.sh uv run python main.py evaluate --weights w.pt --split=test",
        "owner",
    ),
    (
        "scripts/capped.sh uv run python main.py evaluate --weights w.pt --sp test",
        "owner",
    ),
    (
        "scripts/capped.sh uv run python main.py check-data --config configs/badminton_v2.yaml"
        " --out /tmp/o --final",
        "owner",
    ),
    ("scripts/capped.sh uv run python main.py check-data --out /tmp/o --fin", "owner"),
    (
        "scripts/capped.sh uv run python main.py label --video a.mp4 --out l.csv --final",
        "label a video that is not held-out data",
    ),
    ("scripts/capped.sh uv run python main.py label --video a.mp4 --fi", "owner"),
    ("scripts/capped.sh uv run python main.py label --video a.mp4 --out l.csv", ALLOW),
    (
        "PYTHONPATH=. scripts/capped.sh uv run python scripts/analysis/prediction_offset.py"
        " configs/default.yaml w.pt test 5",
        "owner",
    ),
    (
        "scripts/capped.sh .venv/bin/python scripts/analysis/label_vs_ball.py"
        " datasets/tennis game8 --final",
        "owner",
    ),
]

# Rule 3: Python in the project environment (also a bare python after
# activating a virtualenv, and uv run torchrun, jupyter and ipython), any
# Python script and every pytest run go through scripts/capped.sh or
# systemd-run with MemoryMax=. Exempt: main.py train, extract-frames and
# --help, the interpreter's --version and --help, the hook scripts, and the
# system python3 with -c, -m or stdin (it has no numpy, torch or OpenCV).
MEMORY_CAP = [
    ("uv run python --version", ALLOW),
    (".venv/bin/python -V", ALLOW),
    ("uv run python --help", ALLOW),
    ('uv run python -c "import torch" --help', "memory cap"),  # -c runs first
    (
        'source .venv/bin/activate && python -c "from training.run import x"',
        "memory cap",
    ),
    (". .venv/bin/activate; python main.py infer --video a.mp4", "memory cap"),
    ("uv run torchrun main.py", "memory cap"),
    ("uv run jupyter lab", "memory cap"),
    ("uv run --with ipython ipython", "memory cap"),
    ("uv run tensorboard --logdir runs", ALLOW),
    ("python -X dev -m pytest tests/", "memory cap"),
    ("uv run python -m main infer --video a.mp4", "memory cap"),
    ("uv run python main.py label --video a.mp4 --out l.csv", "memory cap"),
    ("uv run ruff check --fix .", ALLOW),
    ("uv run ruff format .", ALLOW),
    ("uv run ty check", ALLOW),
    ("uv sync", ALLOW),
    ("uv run ruff check tests/test_x.py", ALLOW),
    ('git grep -n "main.py infer" docs/', ALLOW),
    ("grep -n pytest docs/x.md", ALLOW),
    ('git commit -m "test: run pytest serially"', ALLOW),
    ("uv run python main.py train --config configs/tennis_v2_noflip.yaml", ALLOW),
    (
        "setsid nohup uv run python main.py train --config configs/badminton_v2.yaml"
        " > /tmp/x.log 2>&1 &",
        ALLOW,
    ),
    ("uv run python main.py --help", ALLOW),
    ("uv run python main.py infer -h", ALLOW),
    ("python3 .claude/hooks/guard_bash.py < payload.json", ALLOW),
    ('python3 -c "print(1)"', ALLOW),
    ("python3 - < script.py", ALLOW),
    ("python3 - <<'EOF'\nprint(1)\nEOF", ALLOW),
    ("python3 -m json.tool x.json", ALLOW),
    ('scripts/capped.sh uv run python -c "import cv2"', ALLOW),
    ("scripts/capped.sh uv run pytest tests/ -q", ALLOW),
    ("CAP=7G scripts/capped.sh uv run pytest tests/test_hooks.py -q", ALLOW),
    ("scripts/capped.sh uv run pytest tests/ -q > /tmp/log 2>&1 &", ALLOW),
    ("scripts/capped.sh uv run pytest tests/ -q 2>&1 | tail -5", ALLOW),
    (
        "systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0"
        " uv run pytest tests/ -q",
        ALLOW,
    ),
    (
        "cd /home/u/repo && PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=5G"
        " -p MemorySwapMax=0 uv run python /tmp/s/raw_detect.py clip.mp4 best.pt"
        " out.csv",
        ALLOW,
    ),
    (
        "for m in a b; do PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=5G"
        " -p MemorySwapMax=0 uv run python /tmp/s/x.py $m; done",
        ALLOW,
    ),
    ("cd /repo && uv run python -m pytest tests/ -q", "memory cap"),
    (".venv/bin/pytest tests/", "memory cap"),
    (f"uv run python {INFER}", "memory cap"),
    (f"uv run {INFER}", "memory cap"),
    (f"uv run ruff check . && uv run python {INFER}", "memory cap"),
    (
        "cd /repo && uv run python main.py evaluate --config c.yaml --weights w.pt"
        " --split val",
        "memory cap",
    ),
    (
        "PYTHONPATH=. uv run python scripts/analysis/val_errors.py configs/default.yaml w.pt",
        "memory cap",
    ),
    (
        ".venv/bin/python -I scripts/analysis/label_vs_ball.py datasets/tennis game1",
        "memory cap",
    ),
    ("systemd-run --user --scope uv run python x.py", "memory cap"),  # no MemoryMax=
    ("uv run python -m json.tool x.json", "memory cap"),
    ('uv run python -c "import cv2"', "memory cap"),
    ('.venv/bin/python -c "import torch"', "memory cap"),
    ("uv run python - <<'EOF'\nimport torch\nEOF", "memory cap"),
    ("uv run python", "memory cap"),
    ("python3 scripts/x.py", "memory cap"),
    # Prefixes whose options take a value.
    ("uv run --with matplotlib python scripts/plot.py", "memory cap"),
    ("uv run --extra dev pytest tests/", "memory cap"),
    ("uv run --directory /repo pytest tests/", "memory cap"),
    ("timeout -s INT 600 uv run pytest tests/", "memory cap"),
    ("env -u X uv run python x.py", "memory cap"),
    (f"/usr/bin/time -v uv run python {INFER}", "memory cap"),
    ("ls tests/*.py | xargs -n 1 uv run pytest", "memory cap"),
]

# Rule 4: pytest runs serially: no -n or --numprocesses (or a prefix of it),
# in the arguments or in PYTEST_ADDOPTS. (One run at a time: see
# test_bash_guard_blocks_pytest_while_another_runs.)
SERIAL_PYTEST = [
    ("scripts/capped.sh uv run pytest tests/ -q -n 4", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -q --numprocesses=auto", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -q --numprocesses 4", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -q --num=2", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -q --nump 3", "serially"),
    ("PYTEST_ADDOPTS='-n 4' scripts/capped.sh uv run pytest tests/", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -n4", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -qn4", "serially"),
    ("scripts/capped.sh uv run pytest tests/ -qn 4", "serially"),
]

# How a command is read: comments end at the end of their line (# inside a
# word is no comment); a heredoc body is commands when fed to a shell, and data
# otherwise; bash -c and sh -c run their string as a command; a backslash
# newline joins lines; unbalanced quotes get the rough check on the raw text.
READING = [
    ("ls docs  # a comment naming uv run pytest tests/", ALLOW),
    (f"ls docs  # first\nuv run python {INFER}", "memory cap"),
    ("echo a#b; uv run pytest tests/", "memory cap"),
    (
        "git commit -F - <<'EOF'\ntest: uv run pytest -n 4, main.py evaluate --final"
        "\nEOF",
        ALLOW,
    ),
    ("git commit -F - <<'EOF'\ndocs: datasets/tennis/game8 stays held out\nEOF", ALLOW),
    ("git commit -m \"$(cat <<'EOF'\nRun pytest serially\nEOF\n)\"", ALLOW),
    (
        "git commit -m \"$(cat <<'EOF'\nfix: an \"odd quote\n"
        'datasets/tennis/game8 stays held out\nEOF\n)"',
        ALLOW,
    ),
    (
        "git commit -m \"$(cat <<'EOF'\nx\nEOF\n)\" && uv run pytest tests/",
        "memory cap",
    ),
    # A << inside quotes or arithmetic starts no heredoc: the next line runs.
    ('echo "a <<EOF"\nuv run pytest tests/', "memory cap"),
    ('git commit -m "x <<EOF"\ncat datasets/tennis/game8/a', "held-out test data"),
    ('grep -n "<<EOF" tests/test_hooks.py\nuv run pytest tests/ -n 4', "serially"),
    ("echo $((1<<4))\nuv run pytest tests/", "memory cap"),
    ("cat <<< 'x'\nuv run pytest tests/", "memory cap"),
    ("bash <<'EOF'\nuv run pytest tests/\nEOF", "memory cap"),
    ("cat <<'EOF' | bash\nuv run pytest tests/\nEOF", "memory cap"),
    ('bash -c "uv run pytest tests/"', "memory cap"),
    (f"sh -lc 'cd /repo && uv run python {INFER}'", "memory cap"),
    ("bash -c 'cat datasets/tennis/game8/Clip1/Label.csv'", "held-out test data"),
    ("uv run python x.py \\\n  --flag", "memory cap"),
    ("uv run pytest tests/ -q 'oops", "could not be split"),
    ("uv run python main.py infer --video 'a.mp4", "could not be split"),
    ("uv run main.py infer --video 'a.mp4", "could not be split"),
    ("scripts/capped.sh uv run pytest tests/ -n 4 'x", "could not be split"),
    (
        "scripts/capped.sh uv run python main.py evaluate --split test 'x",
        "could not be split",
    ),
    (
        "scripts/capped.sh uv run python main.py label --video a.mp4 --final 'x",
        "could not be split",
    ),
    ("cat datasets/tennis/game8/Clip1/Label.csv 'x", "could not be split"),
]

# Blocked commands and the end of their refusal: the command to run instead.
SUGGESTIONS = [
    ("uv run pytest tests/ -q -n 4", "scripts/capped.sh uv run pytest tests/ -q"),
    (
        "uv run pytest tests/test_models.py -v",
        "scripts/capped.sh uv run pytest tests/test_models.py -v",
    ),
    ("python -m pytest tests/ -n4 2>&1", "scripts/capped.sh uv run pytest tests/"),
    (
        "uv run pytest tests/ --numprocesses 4 -q",
        "scripts/capped.sh uv run pytest tests/ -q",
    ),
    (
        "PYTHONPATH=. uv run python x.py 'a b'",
        "PYTHONPATH=. scripts/capped.sh uv run python x.py 'a b'",
    ),
]


@pytest.mark.parametrize(
    ("command", "expected"),
    [*HELD_OUT_PATHS, *OWNER_ONLY, *MEMORY_CAP, *SERIAL_PYTEST, *READING],
)
def test_bash_guard(command, expected):
    refusal = guard_bash.check_command(command, CWD)
    if expected is ALLOW:
        assert refusal is None, refusal
    else:
        assert refusal is not None and refusal.startswith("Blocked")
        assert expected in refusal


@pytest.mark.parametrize(("command", "suggestion"), SUGGESTIONS)
def test_bash_guard_ends_with_the_command_to_run_instead(command, suggestion):
    refusal = guard_bash.check_command(command, str(REPO))
    assert "Run this instead" in refusal
    assert refusal.endswith(suggestion)


@pytest.mark.parametrize(
    ("command", "suggestion"),
    [
        ("uv run pytest tests/ -q", ": scripts/capped.sh uv run pytest tests/ -q"),
        (
            "cd scripts/analysis && uv run python val_errors.py c.yaml w.pt",
            f": {WRAPPER} uv run python val_errors.py c.yaml w.pt",
        ),
    ],
)
def test_bash_guard_names_the_wrapper_by_a_path_that_works_there(command, suggestion):
    """From the repository root the wrapper is scripts/capped.sh; elsewhere its
    absolute path."""
    refusal = guard_bash.check_command(command, str(REPO))
    assert refusal.endswith(suggestion)


@pytest.mark.parametrize(
    ("command", "cwd", "blocked"),
    [
        ("cat game8/Clip1/Label.csv", "/repo/datasets/tennis", True),
        ("cat tennis/game9/Clip1/Label.csv", "/repo/datasets", True),
        ("ls Test", "/repo/datasets/badminton", True),
        ("ls ../badminton/Test", "/repo/datasets/tennis", True),
        ("grep -r 1280 .", "/repo/datasets/tennis", True),
        ("ls", "/repo/datasets/tennis/game8/Clip1", True),
        ("cd /repo", "/repo/datasets/tennis/game8", False),
        ("cat game1/Clip1/Label.csv", "/repo/datasets/tennis", False),
        ("cat game8/notes.txt", "/repo", False),  # not under datasets/tennis
        ("ls .", "/repo/datasets/tennis", False),
    ],
)
def test_bash_guard_resolves_relative_paths_against_cwd(command, cwd, blocked):
    refusal = guard_bash.check_command(command, cwd)
    assert (refusal is not None) == blocked, refusal


def test_bash_guard_blocks_pytest_while_another_runs(monkeypatch):
    other = [(4242, "/r/.venv/bin/python3 /r/.venv/bin/pytest tests/ -q")]
    monkeypatch.setattr(guard_bash, "running_pytest", lambda: other)
    refusal = guard_bash.check_command("scripts/capped.sh uv run pytest tests/ -q", CWD)
    assert refusal is not None and "pid 4242" in refusal
    assert "wait for pid 4242 to finish" in refusal
    assert "ps -o etime=,stat= -p 4242" in refusal and "Do not kill it" in refusal
    # Only pytest waits for pytest.
    assert guard_bash.check_command("uv run ruff check .", CWD) is None


def _fake_proc(tmp_path: Path, processes: dict[str, list[str]]) -> Path:
    for pid, argv in processes.items():
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv))
    return tmp_path


def test_running_pytest_reads_proc(tmp_path):
    proc = _fake_proc(
        tmp_path,
        {
            "101": ["/r/.venv/bin/python3", "/r/.venv/bin/pytest", "tests/", ""],
            "102": ["python3", "-m", "pytest", "tests/"],
            "103": ["/r/.venv/bin/pytest", "tests/"],
            "104": ["bash", "-c", "uv run pytest tests/"],  # a shell, not pytest
            "105": ["uv", "run", "pytest", "tests/"],  # runs pytest as a child
            "106": ["python3", "scripts/x.py", "pytest"],
            "107": [],  # a kernel thread
            "109": ["python3", "-X", "dev", "-m", "pytest", "tests/"],
            "110": ["/r/.venv/bin/pytest", "--collect-only", "-q"],  # only collects
            "111": ["/r/.venv/bin/pytest", "--co", "tests/"],
        },
    )
    (proc / "self").mkdir()  # not a pid
    (proc / "108").mkdir()  # no cmdline: gone, or not ours to read
    assert [pid for pid, _ in running_pytest(str(proc))] == [101, 102, 103, 109]


def test_running_pytest_does_not_block_without_proc(tmp_path):
    assert running_pytest(str(tmp_path / "missing")) == []


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "blocked"),
    [
        ("Read", {"file_path": "/x/datasets/tennis/game9/Clip1/Label.csv"}, True),
        ("Read", {"file_path": "/x/datasets//tennis/game9/Clip1/Label.csv"}, True),
        ("Read", {"file_path": "/x/datasets/tennis/game7/Clip1/Label.csv"}, False),
        ("Read", {"file_path": "/x/docs/data-sources.md"}, False),
        ("Glob", {"pattern": "datasets/badminton/Test/**/*.csv"}, True),
        ("Glob", {"pattern": "**/*.csv", "path": "/x/datasets/badminton/Test"}, True),
        ("Glob", {"pattern": "game8/**", "path": "/x/datasets/tennis"}, True),
        # Glob only lists names: a dataset root, and wildcards in it, are fine.
        ("Glob", {"pattern": "**/*.csv", "path": "/x/datasets/tennis"}, False),
        ("Glob", {"pattern": "datasets/tennis/game*/Clip1"}, False),
        # Grep's pattern is a regex, not a path.
        ("Grep", {"pattern": "datasets/badminton/Test", "path": "/r/docs"}, False),
        ("Grep", {"pattern": "game8", "path": "/x/docs"}, False),
        ("Grep", {"pattern": "1280", "path": "/x/datasets/tennis"}, True),
        ("Grep", {"pattern": "1280", "path": "datasets/"}, True),
        ("Grep", {"pattern": "1280", "path": "/x/datasets/tennis/game*"}, True),
        ("Grep", {"pattern": "1280", "path": "/x/datasets/tennis/game1"}, False),
        ("Grep", {"pattern": "x", "glob": "datasets/tennis/game10/*/Label.csv"}, True),
        (
            "Grep",
            {"pattern": "x", "path": "/x/datasets/tennis/game1", "glob": "*.csv"},
            False,
        ),
    ],
)
def test_path_guard(tool_name, tool_input, blocked):
    refusal = guard_paths.check_call(tool_name, tool_input, CWD)
    assert (refusal is not None) == blocked, refusal


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "cwd", "blocked"),
    [
        ("Grep", {"pattern": "x", "path": "game8"}, "/x/datasets/tennis", True),
        ("Grep", {"pattern": "x"}, "/x/datasets/tennis", True),  # searches the cwd
        ("Grep", {"pattern": "x"}, "/x/datasets/tennis/game1", False),
        ("Glob", {"pattern": "Test/**"}, "/x/datasets/badminton", True),
        ("Glob", {"pattern": "**/*.csv"}, "/x/datasets/badminton", False),
    ],
)
def test_path_guard_resolves_relative_paths_against_cwd(
    tool_name, tool_input, cwd, blocked
):
    refusal = guard_paths.check_call(tool_name, tool_input, cwd)
    assert (refusal is not None) == blocked, refusal


def _payload(tool_name: str, tool_input: dict) -> str:
    return json.dumps({"tool_name": tool_name, "tool_input": tool_input, "cwd": CWD})


def _run_hook(script: str, stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [SYSTEM_PYTHON, str(HOOKS / script)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


# The scripts as Claude Code runs them: exit 0 runs the call, exit 2 blocks it
# with the refusal on stderr. A broken payload blocks too, naming the failure.
HOOK_RUNS = [
    ("guard_bash.py", _payload("Bash", {"command": "uv run ruff check ."}), 0, ""),
    (
        "guard_bash.py",
        _payload("Bash", {"command": "uv run pytest tests/ -q -n 4"}),
        2,
        f"Run this instead, one run at a time: {WRAPPER} uv run pytest tests/ -q",
    ),
    (
        "guard_bash.py",
        json.dumps(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "ls game8"},
                "cwd": "/x/datasets/tennis",
            }
        ),
        2,
        "held-out test data",
    ),
    ("guard_paths.py", _payload("Read", {"file_path": "/x/docs/a.md"}), 0, ""),
    (
        "guard_paths.py",
        _payload("Read", {"file_path": "/x/datasets/tennis/game8/a"}),
        2,
        "held-out test data",
    ),
    (
        "guard_paths.py",
        _payload("Glob", {"pattern": "**/*.csv", "path": "/x/datasets/tennis"}),
        0,
        "",
    ),
    *[
        (script, stdin, 2, f"the {script} hook failed")
        for script in ("guard_bash.py", "guard_paths.py")
        for stdin in ("not json", json.dumps({"tool_name": "Bash"}))
    ],
]


@pytest.mark.parametrize(("script", "stdin", "returncode", "stderr"), HOOK_RUNS)
def test_hook_script_exit_code_and_stderr(script, stdin, returncode, stderr):
    result = _run_hook(script, stdin)
    assert result.returncode == returncode
    if stderr:
        assert stderr in result.stderr
        assert "Traceback" not in result.stderr
    else:
        assert result.stderr == ""


def test_hook_script_sees_this_pytest_run():
    """The real /proc: the hook finds this suite's own pytest process."""
    command = "scripts/capped.sh uv run pytest tests/test_x.py -q"
    result = _run_hook("guard_bash.py", _payload("Bash", {"command": command}))
    assert result.returncode == 2
    assert "another pytest run is going" in result.stderr
    argv = Path(f"/proc/{os.getpid()}/cmdline").read_text().split("\0")
    assert guard_bash.runs_pytest([a for a in argv if a])


def test_bash_hook_is_fast():
    start = time.perf_counter()
    for _ in range(1000):
        guard_bash.check_command(f"cd /repo && scripts/capped.sh uv run python {INFER}")
    assert (time.perf_counter() - start) / 1000 < 0.005  # well under 5 ms each
