"""The hand-labelling tool: frame preparation, the label store, the HTTP API, the
``label`` subcommand and the page's JavaScript (through Node)."""

import errno
import http.client
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from data.labels import load_labels
from labeling import HandLabel, LabelServer, LabelStore, prepare_frames
from labeling import frames as frames_module
from labeling import store as store_module
from labeling.frames import record_differences
from labeling.server import RUN_HEADER
from labeling.store import check_label, exclusive_lock
from tests.conftest import CLIP_SIZE, write_moving_square_clip

WIDTH, HEIGHT = CLIP_SIZE  # 160x90
HEADER = "Frame,Visibility,X,Y\n"
SOURCE = {"kind": "test", "name": "clip"}  # a source record for stores made directly
STATIC = Path(__file__).parents[1] / "labeling" / "static"


def decoded_frames(video: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return frames
        frames.append(frame)


def mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


@pytest.fixture
def video(tmp_path: Path) -> Path:
    return write_moving_square_clip(tmp_path / "clip.mp4", 10)


# Frame preparation


def test_video_is_decoded_in_order_to_numbered_jpegs(video: Path, tmp_path: Path):
    frames = prepare_frames(video, tmp_path / "f")
    reference = decoded_frames(video)
    assert len(reference) == 10
    assert [p.name for p in frames.paths] == [f"{i:04d}.jpg" for i in range(10)]
    assert (frames.width, frames.height) == (WIDTH, HEIGHT)
    for i, path in enumerate(frames.paths):
        image = cv2.imread(str(path))
        assert image is not None and image.shape == (HEIGHT, WIDTH, 3)
        # Frame i is the i-th frame that decodes: closest to it, and close (JPEG
        # of the noisy scene costs about 4 grey levels on average)
        distances = [mean_difference(image, ref) for ref in reference]
        assert int(np.argmin(distances)) == i
        assert distances[i] < 6.0
    assert not (tmp_path / "f.partial").exists()
    stat_ = video.stat()
    assert frames.source == {
        "kind": "video",
        "name": "clip.mp4",
        "bytes": stat_.st_size,
        "mtime_ns": stat_.st_mtime_ns,
        "frames": 10,
    }
    assert json.loads((tmp_path / "f" / "source.json").read_text()) == frames.source


def test_finished_extraction_is_reused_without_decoding(
    video: Path, tmp_path: Path, monkeypatch
):
    first = prepare_frames(video, tmp_path / "f")
    mtimes = [p.stat().st_mtime_ns for p in first.paths]

    def no_decode(_):
        raise AssertionError("decoded again")

    monkeypatch.setattr(frames_module, "_decode", no_decode)
    again = prepare_frames(video, tmp_path / "f")
    assert again == first
    assert [p.stat().st_mtime_ns for p in again.paths] == mtimes


def test_leftover_partial_extraction_is_redone(video: Path, tmp_path: Path):
    partial = tmp_path / "f.partial"
    partial.mkdir()
    (partial / "0000.jpg").write_bytes(b"half written")
    (partial / "junk.txt").write_text("from an interrupted run")
    frames = prepare_frames(video, tmp_path / "f")
    assert len(frames.paths) == 10
    assert not partial.exists()
    assert sorted(p.name for p in (tmp_path / "f").iterdir()) == sorted(
        [f"{i:04d}.jpg" for i in range(10)] + ["source.json"]
    )


def test_interrupted_extraction_leaves_no_frames_dir(
    video: Path, tmp_path: Path, monkeypatch
):
    real_imwrite = cv2.imwrite
    calls = []

    def failing_imwrite(path, image, params):
        calls.append(path)
        if len(calls) == 5:
            raise KeyboardInterrupt
        return real_imwrite(path, image, params)

    monkeypatch.setattr(frames_module.cv2, "imwrite", failing_imwrite)
    with pytest.raises(KeyboardInterrupt):
        prepare_frames(video, tmp_path / "f")
    assert not (tmp_path / "f").exists()
    assert not (tmp_path / "f.partial").exists()
    monkeypatch.setattr(frames_module.cv2, "imwrite", real_imwrite)
    assert len(prepare_frames(video, tmp_path / "f").paths) == 10


def test_frames_of_another_video_are_refused(video: Path, tmp_path: Path):
    prepare_frames(video, tmp_path / "f")
    other = write_moving_square_clip(tmp_path / "other" / "clip.mp4", 14)
    before = sorted((tmp_path / "f").iterdir())
    with pytest.raises(FileExistsError, match="another video.*bytes was"):
        prepare_frames(other, tmp_path / "f")
    assert sorted((tmp_path / "f").iterdir()) == before


def test_frames_of_the_video_before_it_changed_are_refused(video: Path, tmp_path: Path):
    prepare_frames(video, tmp_path / "f")
    os.utime(video, ns=(0, video.stat().st_mtime_ns + 10**9))
    with pytest.raises(FileExistsError, match="mtime_ns was"):
        prepare_frames(video, tmp_path / "f")


@pytest.mark.parametrize("damage", ["missing frame", "empty frame", "no manifest"])
def test_damaged_frames_dir_is_refused_and_left_alone(
    video: Path, tmp_path: Path, damage: str
):
    frames = prepare_frames(video, tmp_path / "f")
    if damage == "missing frame":
        frames.paths[4].unlink()
    elif damage == "empty frame":
        frames.paths[4].write_bytes(b"")
    else:
        (tmp_path / "f" / "source.json").unlink()
    before = sorted((tmp_path / "f").iterdir())
    with pytest.raises(FileExistsError, match="--frames-dir"):
        prepare_frames(video, tmp_path / "f")
    assert sorted((tmp_path / "f").iterdir()) == before


def test_video_that_decodes_nothing_is_an_error(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        prepare_frames(bad, tmp_path / "f")
    assert not (tmp_path / "f").exists()
    assert not (tmp_path / "f.partial").exists()


def test_directory_is_served_as_it_is_in_name_order(tmp_path: Path):
    source = write_moving_square_clip(tmp_path / "pngs", 5)
    (source / "notes.txt").write_text("not a frame")
    frames = prepare_frames(source, None)
    assert frames.paths == [source / f"{i:04d}.png" for i in range(5)]
    assert (frames.width, frames.height) == (WIDTH, HEIGHT)
    assert frames.source == {
        "kind": "directory",
        "path": str(source.resolve()),
        "count": 5,
        "files": [[p.name, p.stat().st_size] for p in frames.paths],
    }
    with pytest.raises(ValueError, match="frames-dir"):
        prepare_frames(source, tmp_path / "f")


def rename_frames(directory: Path, names: list[str]) -> None:
    for path, name in zip(sorted(directory.glob("*.png")), names, strict=True):
        path.rename(directory / name)


@pytest.mark.parametrize(
    "names",
    [
        ["1.png", "2.png", "3.png", "10.png", "11.png"],  # name order 1, 10, 11, 2, 3
        ["01.png", "1.png", "2.png", "3.png", "4.png"],  # 01 and 1 are one number
    ],
)
def test_numbered_names_out_of_numeric_order_are_refused(tmp_path: Path, names):
    source = write_moving_square_clip(tmp_path / "pngs", 5)
    rename_frames(source, names)
    with pytest.raises(ValueError, match="zero-pad"):
        prepare_frames(source, None)


@pytest.mark.parametrize(
    "names",
    [
        ["0.png", "1.png", "2.png", "3.png", "9.png"],
        ["a.png", "b.png", "c.png", "d.png", "e.png"],
        ["1.png", "10.png", "2.png", "3.png", "x.png"],  # not all numbers: name order
    ],
)
def test_other_names_are_taken_in_name_order(tmp_path: Path, names):
    source = write_moving_square_clip(tmp_path / "pngs", 5)
    rename_frames(source, names)
    assert [p.name for p in prepare_frames(source, None).paths] == sorted(names)


def test_record_differences_names_each_key_and_the_first_file():
    then = {"kind": "directory", "path": "/a", "files": [["0.png", 1], ["1.png", 2]]}
    assert record_differences(then, then) == []
    now = {"kind": "directory", "path": "/b", "files": [["0.png", 1], ["1.png", 3]]}
    assert record_differences(then, now) == [
        "path was '/a', is '/b'",
        "files[1] was ['1.png', 2], is ['1.png', 3]",
    ]
    longer = {**then, "files": [*then["files"], ["2.png", 4]]}
    assert record_differences(then, longer) == [
        "files[2] was 'nothing', is ['2.png', 4]"
    ]
    assert record_differences(then, {"kind": "video", "name": "a.mp4"}) == [
        "kind was 'directory', is 'video'"
    ]


# Label store


def make_store(tmp_path: Path, name: str = "labels.csv", source=SOURCE) -> LabelStore:
    return LabelStore(tmp_path / name, 10, WIDTH, HEIGHT, source)


def test_new_store_writes_header_only_and_its_source_record(tmp_path: Path):
    store = make_store(tmp_path, "sub/dir/labels.csv")
    assert (tmp_path / "sub/dir/labels.csv").read_text() == HEADER
    record = tmp_path / "sub/dir/labels.csv.source.json"
    assert json.loads(record.read_text()) == SOURCE
    assert store.labels == {}
    assert load_labels(store.path) == []


def test_each_visibility_round_trips_through_load_labels(tmp_path: Path):
    store = make_store(tmp_path)
    store.set(7, HandLabel(3, None, None))
    store.set(2, HandLabel(1, 10.126, 20))
    store.set(5, HandLabel(0, None, None))
    store.set(3, HandLabel(2, 0, 89))
    assert store.path.read_text() == (
        HEADER + "2,1,10.13,20.00\n3,2,0.00,89.00\n5,0,,\n7,3,,\n"
    )
    loaded = load_labels(store.path)
    assert [(lb.frame, lb.visible, lb.x, lb.y) for lb in loaded] == [
        (2, True, 10.13, 20.0),
        (3, True, 0.0, 89.0),
        (5, False, 0.0, 0.0),
        (7, False, 0.0, 0.0),
    ]
    assert store.labels[2] == HandLabel(1, 10.13, 20.0)


def test_store_resumes_and_replaces_and_deletes(tmp_path: Path):
    store = make_store(tmp_path)
    store.set(4, HandLabel(1, 5.5, 6.25))
    store.set(1, HandLabel(0, None, None))
    resumed = make_store(tmp_path)
    assert resumed.labels == store.labels
    resumed.set(4, HandLabel(2, 7, 8))
    resumed.delete(1)
    resumed.delete(9)  # not labelled: nothing to do
    assert resumed.path.read_text() == HEADER + "4,2,7.00,8.00\n"


def test_store_writes_through_a_temporary_file_and_os_replace(
    tmp_path: Path, monkeypatch
):
    store = make_store(tmp_path)
    store.set(0, HandLabel(1, 1, 1))
    before = store.path.read_text()
    events = []
    real_replace, real_sync = os.replace, store_module._sync_directory

    def spy_replace(src, dst):
        events.append(("replace", Path(src).parent, Path(dst)))
        return real_replace(src, dst)

    def spy_sync(directory):
        events.append(("sync", directory))
        return real_sync(directory)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(store_module, "_sync_directory", spy_sync)
    store.set(1, HandLabel(1, 2, 2))
    assert events == [("replace", tmp_path, store.path), ("sync", tmp_path)]

    def failing(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError, match="disk full"):
        store.set(2, HandLabel(1, 3, 3))
    with pytest.raises(OSError, match="disk full"):
        store.delete(0)
    assert store.path.read_text() == before + "1,1,2.00,2.00\n"
    assert sorted(store.labels) == [0, 1]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["labels.csv", "labels.csv.source.json"]  # no temp left


def test_new_csv_is_0644_and_an_existing_one_keeps_its_mode(tmp_path: Path):
    old_umask = os.umask(0o077)  # as strict as mkstemp's own 0600
    try:
        store = make_store(tmp_path)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o644
    store.path.chmod(0o640)
    store.set(0, HandLabel(0, None, None))
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o640


@pytest.mark.parametrize(("code", "raises"), [(errno.EINVAL, False), (errno.EIO, True)])
def test_directory_sync_tolerates_only_a_file_system_that_cannot(
    tmp_path: Path, monkeypatch, code: int, raises: bool
):
    def failing_fsync(fd):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(os, "fsync", failing_fsync)
    if raises:
        with pytest.raises(OSError, match=os.strerror(code)):
            store_module._sync_directory(tmp_path)
    else:
        store_module._sync_directory(tmp_path)


def test_closed_store_refuses_changes(tmp_path: Path):
    store = make_store(tmp_path)
    store.close()
    with pytest.raises(OSError, match="stopping"):
        store.set(0, HandLabel(0, None, None))
    with pytest.raises(OSError, match="stopping"):
        store.delete(0)
    assert store.path.read_text() == HEADER


def test_close_waits_for_a_change_being_written(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    writing, finish = threading.Event(), threading.Event()
    real_replace = os.replace

    def slow_replace(src, dst):
        writing.set()
        assert finish.wait(5)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", slow_replace)
    writer = threading.Thread(target=store.set, args=(3, HandLabel(0, None, None)))
    writer.start()
    assert writing.wait(5)
    closer = threading.Thread(target=store.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()  # still waiting for the write
    finish.set()
    writer.join(5)
    closer.join(5)
    assert not closer.is_alive()
    assert store.path.read_text() == HEADER + "3,0,,\n"


# Every rule of a label; the CSV and HTTP tables below check only that each path
# applies them, plus what is particular to that path
@pytest.mark.parametrize(
    ("frame", "visibility", "x", "y", "message"),
    [
        (10, 0, None, None, "outside 0-9"),
        (-1, 0, None, None, "outside 0-9"),
        (0, 4, None, None, "visibility must be"),
        (0, -1, None, None, "visibility must be"),
        (0, None, 1, 1, "visibility must be"),
        (0, "1", 1, 1, "visibility must be"),
        (0, 1.0, 1, 1, "visibility must be"),
        (0, True, 1, 1, "visibility must be"),
        (0, 1, None, 3, "needs x"),
        (0, 2, 3, None, "needs y"),
        (0, 1, 160, 3, "needs x in 0-159"),
        (0, 1, -0.01, 3, "needs x"),
        (0, 1, 3, 89.5, "needs y in 0-89"),
        (0, 1, float("nan"), 3, "needs x"),
        (0, 1, float("inf"), 3, "needs x"),
        (0, 1, 10**400, 3, "needs x"),  # too large for a float
        (0, 1, "1", 3, "needs x"),
        (0, 1, True, 3, "needs x"),
        (0, 0, 3, None, "takes no coordinates"),
        (0, 3, None, 3, "takes no coordinates"),
    ],
)
def test_check_label_rejects(frame: int, visibility, x, y, message: str):
    with pytest.raises(ValueError, match=message):
        check_label(frame, HandLabel(visibility, x, y), 10, WIDTH, HEIGHT)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("Frame,Visibility,X\n", "line 1: header"),
        ("", "line 1: header"),
        (HEADER + "0,1,5,5\n10,0,,\n", "line 3: frame 10 is outside"),
        (HEADER + "0,1,5,5\n0,0,,\n", "line 3: duplicate frame 0"),
        (HEADER + "0,1,,\n", "line 2: visibility 1 needs x"),  # empty field: None
        (HEADER + "zero,1,5,5\n", "line 2: invalid literal"),
        (HEADER + "0,1,five,5\n", "line 2: could not convert"),
        (HEADER + "0,1,5\n", "line 2: 3 fields"),
    ],
)
def test_bad_existing_file_is_an_error_and_left_alone(
    tmp_path: Path, content: str, message: str
):
    path = tmp_path / "labels.csv"
    path.write_text(content)
    (tmp_path / "labels.csv.source.json").write_text(json.dumps(SOURCE))
    with pytest.raises(ValueError, match=message):
        make_store(tmp_path)
    assert path.read_text() == content


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (None, "has no labels.csv.source.json"),
        ("{not json", "is not JSON"),
        (json.dumps({**SOURCE, "name": "other"}), "name was 'other', is 'clip'"),
    ],
)
def test_csv_without_a_matching_source_record_is_refused(
    tmp_path: Path, record: str | None, message: str
):
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + "0,0,,\n")
    if record is not None:
        (tmp_path / "labels.csv.source.json").write_text(record)
    with pytest.raises(ValueError, match=message):
        make_store(tmp_path)
    assert path.read_text() == HEADER + "0,0,,\n"


def test_record_left_without_its_csv_is_replaced(tmp_path: Path):
    (tmp_path / "labels.csv.source.json").write_text(json.dumps({"kind": "other"}))
    make_store(tmp_path)
    assert json.loads((tmp_path / "labels.csv.source.json").read_text()) == SOURCE


def resume_on(frames_source: Path, out: Path) -> LabelStore:
    frames = prepare_frames(frames_source, None)
    return LabelStore(
        out, len(frames.paths), frames.width, frames.height, frames.source
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("moved", "path was"),
        ("frame rewritten", r"files\[2\] was"),
        ("frame added", "count was 5, is 6"),
        ("frame removed", "count was 5, is 4"),
        ("frame renamed", r"files\[4\] was \['0004.png'"),
    ],
)
def test_csv_of_a_directory_is_not_resumed_on_other_frames(
    tmp_path: Path, change: str, message: str
):
    source = write_moving_square_clip(tmp_path / "pngs", 5)
    out = tmp_path / "labels.csv"
    resume_on(source, out).set(1, HandLabel(0, None, None))
    assert resume_on(source, out).labels == {1: HandLabel(0, None, None)}
    if change == "moved":
        source = source.rename(tmp_path / "moved")
    elif change == "frame rewritten":
        cv2.imwrite(str(source / "0002.png"), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    elif change == "frame added":
        shutil.copy(source / "0004.png", source / "0005.png")
    elif change == "frame removed":
        (source / "0004.png").unlink()
    else:
        (source / "0004.png").rename(source / "0009.png")
    with pytest.raises(ValueError, match=f"labels other frames.*{message}"):
        resume_on(source, out)


# HTTP API


@pytest.fixture
def server(tmp_path: Path):
    frames = prepare_frames(write_moving_square_clip(tmp_path / "pngs", 5), None)
    store = LabelStore(
        tmp_path / "labels.csv", 5, frames.width, frames.height, frames.source
    )
    srv = LabelServer(frames, store, 0)
    # A short poll interval so that shutdown() returns at once
    thread = threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def connect(srv: LabelServer) -> http.client.HTTPConnection:
    host, port = srv.server_address[:2]
    return http.client.HTTPConnection(str(host), int(port), timeout=5)


def request(
    srv: LabelServer,
    method: str,
    path: str,
    body: object = None,
    raw: bytes | None = None,
    headers: dict | None = None,
) -> tuple[int, http.client.HTTPResponse, bytes]:
    """Send a request, with this run's RUN_HEADER unless ``headers`` say otherwise."""
    conn = connect(srv)
    try:
        if body is not None:
            raw = json.dumps(body).encode()
        headers = {RUN_HEADER: srv.version, **(headers or {})}
        conn.request(method, path, body=raw, headers=headers)
        response = conn.getresponse()
        return response.status, response, response.read()
    finally:
        conn.close()


def test_state_lists_size_and_labels(server: LabelServer):
    status, _, body = request(server, "GET", "/api/state")
    state = json.loads(body)
    assert status == 200
    assert {k: state[k] for k in ("frames", "width", "height", "labels")} == {
        "frames": 5,
        "width": WIDTH,
        "height": HEIGHT,
        "labels": {},
    }
    assert state["version"] == server.version
    request(server, "PUT", "/api/labels/3", {"visibility": 2, "x": 1.5, "y": 2})
    state = json.loads(request(server, "GET", "/api/state")[2])
    assert state["labels"] == {"3": {"visibility": 2, "x": 1.5, "y": 2.0}}


def test_page_and_scripts_are_served(server: LabelServer):
    status, response, body = request(server, "GET", "/")
    assert status == 200
    assert response.getheader("Content-Type", "").startswith("text/html")
    assert b'<script src="/label_math.js">' in body
    assert b'<script src="/page.js">' in body
    for path, word in [("/label_math.js", b"clickToLabel"), ("/page.js", b"draw()")]:
        status, response, body = request(server, "GET", path)
        assert status == 200
        assert response.getheader("Content-Type", "").startswith("text/javascript")
        assert word in body


def test_frame_is_the_image_with_a_cache_header(server: LabelServer):
    status, response, body = request(server, "GET", "/frames/2?v=abc")
    assert status == 200
    assert response.getheader("Content-Type") == "image/png"
    assert "max-age" in response.getheader("Cache-Control", "")
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (HEIGHT, WIDTH, 3)
    assert body == server.frames.paths[2].read_bytes()


def test_put_and_delete_change_the_csv(server: LabelServer):
    csv_path = server.store.path
    status, _, body = request(
        server, "PUT", "/api/labels/1", {"visibility": 1, "x": 10.004, "y": 20}
    )
    assert status == 200
    assert json.loads(body) == {"visibility": 1, "x": 10.0, "y": 20.0}
    status, _, _ = request(
        server, "PUT", "/api/labels/0", {"visibility": 0, "x": None, "y": None}
    )
    assert status == 200
    request(server, "PUT", "/api/labels/4", {"visibility": 3})
    assert csv_path.read_text() == HEADER + "0,0,,\n1,1,10.00,20.00\n4,3,,\n"
    status, _, body = request(server, "DELETE", "/api/labels/1")
    assert (status, json.loads(body)) == (200, {"deleted": 1})
    assert csv_path.read_text() == HEADER + "0,0,,\n4,3,,\n"
    assert load_labels(csv_path)[0].frame == 0


@pytest.mark.parametrize(
    ("method", "path", "body", "message"),
    [
        ("PUT", "/api/labels/5", {"visibility": 0}, "outside 0-4"),
        ("PUT", "/api/labels/0", {"visibility": 1, "x": 160, "y": 1}, "needs x"),
        ("PUT", "/api/labels/0", {"x": 1, "y": 1}, "visibility must be"),
        ("PUT", "/api/labels/0", {"visibility": True, "x": 1}, "visibility must be"),
        ("PUT", "/api/labels/0", {"visibility": 1, "x": "1", "y": 1}, "needs x"),
        ("PUT", "/api/labels/0", {"visibility": 3, "y": 1}, "no coordinates"),
        ("PUT", "/api/labels/0", {"visibility": 0, "z": 1}, "the body must be"),
        ("PUT", "/api/labels/0", [0, 1, 2], "the body must be"),
        ("PUT", "/api/labels/0", b'{"visibility": 1, "x": NaN, "y": 1}', "needs x"),
        ("PUT", "/api/labels/0", b"visibility=1", "not JSON"),
        ("PUT", "/api/labels/0", b"\xff\xfe", "not JSON"),
        ("PUT", "/api/labels/0", b"", "1-1024 bytes"),
        ("PUT", "/api/labels/0", b" " * 1025, "1-1024 bytes"),
        ("DELETE", "/api/labels/5", None, "outside 0-4"),
        ("GET", "/frames/5", None, "outside 0-4"),
    ],
)
def test_bad_requests_answer_400_and_write_nothing(
    server: LabelServer, method: str, path: str, body: object, message: str
):
    request(server, "PUT", "/api/labels/2", {"visibility": 1, "x": 3, "y": 4})
    before = server.store.path.read_bytes()
    if isinstance(body, bytes):
        status, response, text = request(server, method, path, raw=body)
    else:
        status, response, text = request(server, method, path, body)
    assert status == 400
    assert response.getheader("Content-Type", "").startswith("text/plain")
    assert message in text.decode()
    assert server.store.path.read_bytes() == before
    assert list(server.store.labels) == [2]


def test_put_without_content_length_is_400(server: LabelServer):
    conn = connect(server)
    try:
        conn.putrequest("PUT", "/api/labels/0")
        conn.putheader(RUN_HEADER, server.version)
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 400
        assert b"Content-Length" in response.read()
    finally:
        conn.close()
    assert server.store.path.read_text() == HEADER


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/nope"),
        ("GET", "/frames/abc"),
        ("GET", "/frames/-1"),
        ("GET", "/api/labels/0"),
        ("GET", "/index.html"),
        ("PUT", "/api/labels/abc"),
        ("PUT", "/api/label/0"),
        ("DELETE", "/api/labels/"),
        ("DELETE", "/frames/0"),
    ],
)
def test_unknown_routes_are_404(server: LabelServer, method: str, path: str):
    body = {"visibility": 0} if method == "PUT" else None
    status, _, _ = request(server, method, path, body)
    assert status == 404
    assert server.store.path.read_text() == HEADER


@pytest.mark.parametrize("token", [None, "", "0123abcd", "stale"])
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_change_from_another_run_is_409_and_writes_nothing(
    server: LabelServer, method: str, token: str | None
):
    request(server, "PUT", "/api/labels/2", {"visibility": 0})
    before = server.store.path.read_bytes()
    conn = connect(server)
    try:
        body = (
            json.dumps({"visibility": 1, "x": 1, "y": 1}) if method == "PUT" else None
        )
        headers = {} if token is None else {RUN_HEADER: token}
        conn.request(method, "/api/labels/2", body=body, headers=headers)
        response = conn.getresponse()
        status, text = response.status, response.read().decode()
    finally:
        conn.close()
    assert status == 409
    assert "earlier run" in text and "reload" in text
    assert server.store.path.read_bytes() == before
    assert server.store.labels == {2: HandLabel(0, None, None)}


@pytest.mark.parametrize(
    ("host", "status"),
    [
        ("localhost:{port}", 200),
        ("127.0.0.1:{port}", 200),
        ("evil.example:{port}", 400),
        ("localhost", 400),
        ("localhost:1", 400),
        (None, 400),
    ],
)
@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_requests_must_name_this_server_as_host(
    server: LabelServer, method: str, host: str | None, status: int
):
    port = server.server_address[1]
    conn = connect(server)
    try:
        path = "/" if method == "GET" else "/api/labels/0"
        conn.putrequest(method, path, skip_host=True)
        if host is not None:
            conn.putheader("Host", host.format(port=port))
        conn.putheader(RUN_HEADER, server.version)
        body = b'{"visibility": 0}' if method == "PUT" else b""
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        response = conn.getresponse()
        assert response.status == status
        text = response.read()
    finally:
        conn.close()
    if status == 400:
        assert b"Host header" in text
        assert server.store.path.read_text() == HEADER


@pytest.mark.parametrize("failure", ["os.replace", "directory not writable"])
def test_write_error_is_500_with_its_message_and_changes_nothing(
    server: LabelServer, tmp_path: Path, monkeypatch, failure: str
):
    request(server, "PUT", "/api/labels/2", {"visibility": 0})
    before = server.store.path.read_bytes()
    if failure == "os.replace":

        def full(src, dst):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(os, "replace", full)
        expected = "No space left on device"
    else:
        if os.geteuid() == 0:
            pytest.skip("root can write to a read-only directory")
        tmp_path.chmod(0o500)  # mkstemp fails
        expected = "Permission denied"
    try:
        for method, path, body in [
            ("PUT", "/api/labels/3", {"visibility": 3}),
            ("DELETE", "/api/labels/2", None),
        ]:
            status, _, text = request(server, method, path, body)
            assert status == 500
            assert expected in text.decode()
            assert server.store.labels == {2: HandLabel(0, None, None)}
    finally:
        tmp_path.chmod(0o700)
    assert server.store.path.read_bytes() == before
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_concurrent_changes_all_reach_a_valid_csv(server: LabelServer):
    errors = []

    def label(frame: int) -> None:
        for visibility in (0, 3, 1):
            body: dict[str, float] = {"visibility": visibility}
            if visibility == 1:
                body |= {"x": frame + 0.25, "y": 2 * frame}
            status, _, text = request(server, "PUT", f"/api/labels/{frame}", body)
            if status != 200:
                errors.append((frame, status, text))

    threads = [threading.Thread(target=label, args=(f,)) for f in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == []
    expected = {f: HandLabel(1, f + 0.25, 2.0 * f) for f in range(5)}
    assert server.store.labels == expected
    loaded = load_labels(server.store.path)
    assert [(lb.frame, lb.x, lb.y) for lb in loaded] == [
        (f, f + 0.25, 2.0 * f) for f in range(5)
    ]


def test_closed_store_answers_500(server: LabelServer):
    server.store.close()
    status, _, text = request(server, "PUT", "/api/labels/0", {"visibility": 0})
    assert (status, text) == (500, b"the label tool is stopping\n")
    assert server.store.path.read_text() == HEADER


# The label subcommand


def test_label_arguments_have_defaults():
    args = main.build_parser().parse_args(["label", "--video", "v.mp4", "--out", "l"])
    assert (args.video, args.out, args.frames_dir, args.port, args.final) == (
        "v.mp4",
        "l",
        None,
        8765,
        False,
    )


def run_label(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "label", *args])
    main.main()


def test_main_label_prepares_frames_serves_and_stops_on_ctrl_c(
    video: Path, tmp_path: Path, monkeypatch, capsys
):
    served = []

    def fake_serve_forever(self):
        served.append(self)
        # The lock is held while serving
        with pytest.raises(OSError, match="in use by another label run"):
            with exclusive_lock(self.store.path):
                pass
        raise KeyboardInterrupt  # as Ctrl-C does

    monkeypatch.setattr(LabelServer, "serve_forever", fake_serve_forever)
    out = tmp_path / "labels" / "clip.csv"
    run_label(monkeypatch, "--video", str(video), "--out", str(out), "--port", "0")

    (srv,) = served
    host, port = srv.server_address[:2]
    assert host == "127.0.0.1"
    assert srv.socket.fileno() == -1  # closed
    printed = capsys.readouterr().out
    assert f"http://localhost:{port}/ or http://127.0.0.1:{port}/" in printed
    assert "10 frames of 160x90, 0 labelled" in printed
    assert "Stopped" in printed
    assert out.read_text() == HEADER
    assert len(list((tmp_path / "labels" / "clip.frames").glob("*.jpg"))) == 10
    assert json.loads((tmp_path / "labels" / "clip.csv.source.json").read_text())[
        "name"
    ] == ("clip.mp4")
    with pytest.raises(OSError, match="stopping"):  # the store refuses changes now
        srv.store.set(0, HandLabel(0, None, None))
    with exclusive_lock(out):  # released
        pass


def test_main_label_passes_frames_dir_and_port(video: Path, monkeypatch):
    calls = []
    monkeypatch.setattr("labeling.run_label_tool", lambda *a: calls.append(a))
    run_label(
        monkeypatch, "--video", str(video), "--out", "o.csv", "--frames-dir", "fd"
    )
    run_label(monkeypatch, "--video", str(video), "--out", "o.csv", "--port", "9000")
    assert calls == [
        (str(video), "o.csv", "fd", 8765),
        (str(video), "o.csv", None, 9000),
    ]


def test_second_run_on_the_same_csv_stops(video: Path, tmp_path: Path, monkeypatch):
    out = tmp_path / "clip.csv"
    with exclusive_lock(out):
        with pytest.raises(SystemExit) as exit_info:
            run_label(monkeypatch, "--video", str(video), "--out", str(out))
    message = str(exit_info.value.code)
    assert message.startswith("label: ") and "in use by another label run" in message
    assert not out.exists()
    assert not (tmp_path / "clip.frames").exists()


def test_label_refuses_a_csv_of_other_frames(video: Path, tmp_path: Path, monkeypatch):
    def stop(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(LabelServer, "serve_forever", stop)
    out = tmp_path / "clip.csv"
    run_label(monkeypatch, "--video", str(video), "--out", str(out), "--port", "0")
    other = write_moving_square_clip(tmp_path / "other" / "clip.mp4", 14)
    args = ["--video", str(other), "--out", str(out), "--port", "0"]
    with pytest.raises(SystemExit) as exit_info:  # its frames directory, first
        run_label(monkeypatch, *args)
    assert "another video" in str(exit_info.value.code)
    with pytest.raises(SystemExit) as exit_info:  # then the CSV
        run_label(monkeypatch, *args, "--frames-dir", str(tmp_path / "other.frames"))
    message = str(exit_info.value.code)
    assert message.startswith(f"label: {out} labels other frames")
    assert "bytes was" in message and "frames was 10, is 14" in message
    assert out.read_text() == HEADER


@pytest.mark.parametrize(
    "video", ["tennis/game8/Clip1", "badminton/Test/match1/video/a.mp4"]
)
def test_label_refuses_held_out_video_without_final(
    monkeypatch, tmp_path: Path, video: str
):
    out = tmp_path / "out" / "labels.csv"
    with pytest.raises(SystemExit) as exit_info:
        run_label(
            monkeypatch,
            "--video",
            str(tmp_path / "datasets" / video),
            "--out",
            str(out),
        )
    message = str(exit_info.value.code)
    assert message.startswith("label: ")
    assert "held-out test data" in message and "--final" in message
    assert not (tmp_path / "out").exists()  # nothing written, not even the lock


def test_label_takes_held_out_video_with_final(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr("labeling.run_label_tool", lambda *a: calls.append(a))
    video = str(tmp_path / "datasets/tennis/game9/Clip1")
    run_label(monkeypatch, "--video", video, "--out", "o.csv", "--final")
    assert calls == [(video, "o.csv", None, 8765)]


# The page's JavaScript

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(
    NODE is None, reason="node is not installed: the page's JavaScript is not tested"
)
MATH_SCRIPT = """
const M = require(process.argv[1]);
const scales = [0.25, 0.5, 0.7111, 1, 1.5, 3];
// The centre of source pixel (10, 20) is 10.5, 20.5 source pixels from the edges
const centre = scales.map((s) => [
  M.clickToLabel(37 + 10.5 * s, 37, 160 * s, 160),
  M.clickToLabel(12 + 20.5 * s, 12, 90 * s, 90),
]);
console.log(JSON.stringify({
  centre,
  leftEdge: M.clickToLabel(37, 37, 80, 160),
  rightEdge: M.clickToLabel(117, 37, 80, 160),
  beyond: [M.clickToLabel(0, 37, 80, 160), M.clickToLabel(500, 37, 80, 160)],
  pixelEdge: M.clickToLabel(10, 0, 160, 160),
  rounded: M.clickToLabel(10.12345, 0, 160, 160),
  edge: M.toSourceEdge(50, 10, 80, 160),
  view: [M.labelToView(10, 2), M.labelToView(10, 8, 5.25)],
  clamp: [M.clamp(-1, 0, 5), M.clamp(3, 0, 5), M.clamp(9, 0, 5)],
  first: [
    M.firstUnlabelled({}, 3),
    M.firstUnlabelled({"0": {}, "2": {}}, 3),
    M.firstUnlabelled({"0": {}, "1": {}, "2": {}}, 3),
    M.firstUnlabelled({}, 0),
  ],
}));
"""
PAGE_TESTS = Path(__file__).parent / "js" / "label_page_test.js"


def run_node(*args: str) -> dict:
    assert NODE is not None
    result = subprocess.run(
        [NODE, *args], capture_output=True, text=True, timeout=60, check=True
    )
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def js() -> dict:
    if NODE is None:
        pytest.skip("node is not installed: the page's JavaScript is not tested")
    return run_node("-e", MATH_SCRIPT, str(STATIC / "label_math.js"))


def test_click_on_a_pixel_centre_labels_that_pixel_at_any_scale(js: dict):
    assert js["centre"] == [[10.0, 20.0]] * 6


def test_click_maps_through_pixel_centres_and_is_clamped(js: dict):
    assert js["leftEdge"] == 0  # u = 0 is x = -0.5, clamped to the first centre
    assert js["rightEdge"] == 159  # u = 160 is x = 159.5, clamped to the last
    assert js["beyond"] == [0, 159]
    assert js["pixelEdge"] == 9.5  # the edge between pixels 9 and 10
    assert js["rounded"] == 9.62
    assert js["edge"] == 80


def test_label_to_view_inverts_the_click_mapping(js: dict):
    assert js["view"] == [21, (10.5 - 5.25) * 8]


def test_clamp_and_first_unlabelled(js: dict):
    assert js["clamp"] == [0, 3, 5]
    assert js["first"] == [0, 1, -1, -1]


@needs_node
def test_page_script_in_a_stubbed_browser():
    """tests/js/label_page_test.js: the page's input, timing, saving and drawing
    rules, with a fake clock (see its cases for what each checks)."""
    results = run_node(str(PAGE_TESTS), str(STATIC))
    failures = "\n\n".join(f"{f['name']}:\n{f['message']}" for f in results["failed"])
    assert not results["failed"], failures
    assert len(results["passed"]) >= 30
