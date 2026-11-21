"""The test split is held out: evaluate, check-data and the analysis scripts
reach it only with ``--final``."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import main
from data import Clip
from tests.test_pipeline import _config_for, _default_weights, _fast_training_config
from training.run import (
    HeldOutSplitError,
    require_held_out_in_test,
    run_evaluation,
    run_training,
    write_label_overlays,
)

REPO = Path(__file__).parents[1]


def _run_main(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    main.main()


def _config_file(tennis_root, tmp_path) -> Path:
    path = tmp_path / "config.yaml"
    config = _config_for(tennis_root) | {"num_workers": 0, "pin_memory": False}
    path.write_text(yaml.safe_dump(config))
    return path


def _forbid_loading(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("loaded something before refusing the test split")

    monkeypatch.setattr("training.run.load_config", fail)
    monkeypatch.setattr("training.run.load_model", fail)
    monkeypatch.setattr("training.run._build_datasets", fail)


def test_evaluate_defaults_to_val_and_not_final():
    args = main.build_parser().parse_args(["evaluate", "--weights", "w.pt"])
    assert (args.split, args.final) == ("val", False)


def test_evaluate_refuses_test_split_without_final(monkeypatch, tmp_path, capsys):
    _forbid_loading(monkeypatch)
    args = ["evaluate", "--weights", str(tmp_path / "w.pt"), "--split", "test"]
    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, *args)
    message = str(exit_info.value.code)
    assert "held out until the model is final" in message
    assert "--final" in message
    assert capsys.readouterr().out == ""
    assert list(tmp_path.iterdir()) == []


def test_run_evaluation_refuses_test_split_without_final(monkeypatch, tmp_path):
    _forbid_loading(monkeypatch)
    with pytest.raises(HeldOutSplitError, match="--final"):
        run_evaluation({}, tmp_path / "w.pt", "test")


def test_evaluate_runs_on_test_split_with_final(
    tennis_root, tmp_path, monkeypatch, capsys
):
    config_path = _config_file(tennis_root, tmp_path)
    weights = _default_weights(tmp_path)
    args = ["evaluate", "--config", str(config_path), "--weights", str(weights)]
    _run_main(monkeypatch, *args, "--split", "test", "--final")
    result = json.loads(capsys.readouterr().out)
    assert (result["split"], result["frames"]) == ("test", 6)


@pytest.mark.parametrize("final", [False, True])
def test_check_data_builds_test_split_only_with_final(
    tennis_root, tmp_path, monkeypatch, caplog, final
):
    out = tmp_path / "overlays"
    config_path = _config_file(tennis_root, tmp_path)
    args = ["check-data", "--config", str(config_path), "--out", str(out)]
    if final:
        args.append("--final")
    with caplog.at_level("INFO", logger="training.run"):
        _run_main(monkeypatch, *args)
    splits = {p.name.split("_")[0] for p in out.iterdir()}
    assert splits == ({"train", "val", "test"} if final else {"train", "val"})
    assert ("test: 1 clips" in caplog.text) == final  # logged when the split is built
    skipped = "test: held out until the model is final, skipped"
    assert (skipped in caplog.text) == (not final)


class _ReachedInference(Exception):
    """Raised in place of running inference: the held-out check let infer run."""


def _infer(monkeypatch, video: Path, *extra: str) -> None:
    def reached(args):
        raise _ReachedInference

    # run_inference is the first thing to open the video or load the model.
    monkeypatch.setattr("main.run_inference", reached)
    args = ["infer", "--video", str(video), "--model", "m.pt", "--output", "o.csv"]
    _run_main(monkeypatch, *args, *extra)


@pytest.mark.parametrize(
    "video",
    ["tennis/game8/Clip1", "tennis/game10", "badminton/Test/match1/video/a.mp4"],
)
def test_infer_refuses_held_out_video_without_final(monkeypatch, tmp_path, video):
    with pytest.raises(SystemExit) as exit_info:
        _infer(monkeypatch, tmp_path / "datasets" / video)
    message = str(exit_info.value.code)
    assert message.startswith("infer: ")
    assert "held-out test data" in message and "--final" in message


def test_infer_reads_held_out_video_with_final(monkeypatch, tmp_path):
    with pytest.raises(_ReachedInference):
        _infer(monkeypatch, tmp_path / "datasets/tennis/game9/Clip1", "--final")


@pytest.mark.parametrize(
    "video", ["tennis/game1/Clip1", "badminton/Professional/match1/video/a.mp4"]
)
def test_infer_runs_on_other_videos(monkeypatch, tmp_path, video):
    with pytest.raises(_ReachedInference):
        _infer(monkeypatch, tmp_path / "datasets" / video)


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, f"scripts/analysis/{script}", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )


def test_prediction_offset_refuses_test_split_without_final(tmp_path):
    result = _run_script(
        "prediction_offset.py",
        "configs/default.yaml",
        str(tmp_path / "w.pt"),
        "test",
        "5",
    )
    assert result.returncode != 0
    assert "held out until the model is final" in result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "game", ["game8", "game9", "game10", "game8/", "Test", "Test/match1", "x/../game9"]
)
def test_label_vs_ball_refuses_held_out_games(tmp_path, game):
    result = _run_script("label_vs_ball.py", str(tmp_path), "game1", game)
    assert result.returncode != 0
    assert f"{game} is in the held-out test split" in result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert result.stdout == ""


def test_label_vs_ball_reads_held_out_game_with_final(tmp_path):
    clip = tmp_path / "game8" / "Clip1"
    clip.mkdir(parents=True)
    (clip / "Label.csv").write_text("file name,visibility,x-coordinate,y-coordinate\n")
    result = _run_script("label_vs_ball.py", str(tmp_path), "game8", "--final")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "game8: no frame measured\n"


def _clip(group: str) -> Clip:
    return Clip(Path("frames"), Path("Label.csv"), group)


@pytest.mark.parametrize(
    ("split", "group", "built", "refused"),
    [
        ("train", "game8", ("train", "val"), True),
        ("val", "game10", ("val",), True),
        ("val", "Test/match1", ("val",), True),
        ("train", "Test/match2", ("train",), True),
        ("train", "game8", ("val",), False),  # train is not being built
        ("test", "game9", ("test",), False),  # where it belongs
        ("train", "game1", ("train",), False),
        ("train", "Professional/match1", ("train",), False),
    ],
)
def test_held_out_groups_stay_out_of_train_and_val(split, group, built, refused):
    splits = {"train": [_clip("game1")], "val": [], "test": []}
    splits[split].append(_clip(group))
    if refused:
        with pytest.raises(HeldOutSplitError, match=rf"{split} split holds .*{group}"):
            require_held_out_in_test(splits, built, final=False)
    else:
        require_held_out_in_test(splits, built, final=False)
    require_held_out_in_test(splits, built, final=True)  # the owner's call


def _with_game8_in_train(tennis_root) -> None:
    """A held-out game the config leaves out of test_groups: split_clips puts it
    in train."""
    shutil.copytree(tennis_root / "game3", tennis_root / "game8")


def test_training_refuses_held_out_game_in_train(tennis_root, tmp_path):
    _with_game8_in_train(tennis_root)
    config = _fast_training_config(tennis_root, tmp_path)
    with pytest.raises(HeldOutSplitError, match=r"train split holds .*\(game8\)"):
        run_training(config)
    assert not (tmp_path / "checkpoints").exists()


def test_evaluation_refuses_held_out_game_in_val(tennis_root, tmp_path):
    _with_game8_in_train(tennis_root)
    config = _config_for(tennis_root, val_groups=["game8"])
    with pytest.raises(HeldOutSplitError, match=r"val split holds .*\(game8\)"):
        run_evaluation(config, _default_weights(tmp_path), "val")


@pytest.mark.parametrize("final", [False, True])
def test_check_data_refuses_held_out_game_in_train_without_final(
    tennis_root, tmp_path, final
):
    _with_game8_in_train(tennis_root)
    config = _config_for(tennis_root)
    out = tmp_path / "overlays"
    if final:
        paths = write_label_overlays(config, out, 3, final=True)
        assert [p.name.split("_")[0] for p in paths] == ["train", "val", "test"]
    else:
        with pytest.raises(HeldOutSplitError, match="train split holds"):
            write_label_overlays(config, out, 3)
        assert not out.exists()


def test_prediction_offset_refuses_held_out_game_in_train(tennis_root, tmp_path):
    _with_game8_in_train(tennis_root)
    config_path = _config_file(tennis_root, tmp_path)
    weights = str(tmp_path / "w.pt")  # never loaded
    result = _run_script(
        "prediction_offset.py", str(config_path), weights, "train", "5"
    )
    assert result.returncode != 0
    assert "train split holds held-out test data (game8)" in result.stderr
    assert "Traceback" not in result.stderr, result.stderr
