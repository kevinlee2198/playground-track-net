"""The labelling page's local HTTP server (standard library only)."""

from __future__ import annotations

import json
import re
import secrets
import sys
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from labeling.frames import FrameSet, prepare_frames
from labeling.store import HandLabel, LabelStore, check_frame, exclusive_lock

HOST = (
    "127.0.0.1"  # the page can change the labels, so never serve it beyond this machine
)
STATIC = Path(__file__).with_name("static")
PAGES = {  # path: (file in STATIC, media type)
    "/": ("index.html", "text/html"),
    "/label_math.js": ("label_math.js", "text/javascript"),
    "/page.js": ("page.js", "text/javascript"),
}
MAX_BODY = 1024  # bytes: a label is under 60
IMAGE_TYPES = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".bmp": "bmp"}
# Every change names the run whose page sent it, so a tab left open from an
# earlier run (perhaps of another clip) cannot write into this run's CSV
RUN_HEADER = "X-Label-Run"


class NotFound(Exception):
    """A request for a path the server does not have (a ValueError is a bad request)."""


class StaleRun(Exception):
    """A change sent by a page of another run of the server."""


class LabelServer(ThreadingHTTPServer):
    # Handler threads do not hold up Ctrl-C (a browser keeps idle connections
    # open); LabelStore.close waits for a change being written instead
    daemon_threads = True

    def __init__(self, frames: FrameSet, store: LabelStore, port: int):
        super().__init__((HOST, port), _Handler)
        self.frames = frames
        self.store = store
        # Part of frame URLs, so a browser never shows a cached frame of another
        # run, and checked on every change (RUN_HEADER)
        self.version = secrets.token_hex(8)

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], ConnectionError):  # a cancelled preload
            super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    server: LabelServer
    timeout = 30  # seconds: a client that stalls longer loses its connection

    def do_GET(self) -> None:
        self._answer(self._get)

    def do_PUT(self) -> None:
        self._answer(self._put)

    def do_DELETE(self) -> None:
        self._answer(self._delete)

    def _answer(self, action: Callable[[], None]) -> None:
        try:
            self._check_host()
            action()
        except NotFound:
            self._send(404, b"not found\n")
        except StaleRun:
            self._send(
                409,
                b"this page belongs to an earlier run of the label tool: reload it\n",
            )
        except ValueError as e:  # nothing was written
            self._send(400, f"{e}\n".encode())
        except ConnectionError:  # the client has gone: nobody to answer
            raise
        except OSError as e:  # a file could not be read or written; nothing changed
            self._send(500, f"{e}\n".encode())

    def _check_host(self) -> None:
        """Refuse a page of another site that reaches this port by DNS rebinding."""
        port = self.server.server_address[1]
        allowed = (f"localhost:{port}", f"127.0.0.1:{port}")
        if self.headers.get("Host") not in allowed:
            raise ValueError(f"the Host header must be {' or '.join(allowed)}")

    def _get(self) -> None:
        path = urlsplit(self.path).path
        frames = self.server.frames
        if path in PAGES:
            name, kind = PAGES[path]
            self._send(200, (STATIC / name).read_bytes(), f"{kind}; charset=utf-8")
        elif path == "/api/state":
            labels = self.server.store.labels  # replaced on change, never changed
            state = {
                "frames": len(frames.paths),
                "width": frames.width,
                "height": frames.height,
                "version": self.server.version,
                "labels": {str(i): vars(label) for i, label in labels.items()},
            }
            self._send_json(state)
        elif match := re.fullmatch(r"/frames/(\d+)", path):
            frame = int(match[1])
            check_frame(frame, len(frames.paths))
            file = frames.paths[frame]
            kind = f"image/{IMAGE_TYPES[file.suffix.lower()]}"
            self._send(
                200, file.read_bytes(), kind, "private, max-age=86400, immutable"
            )
        else:
            raise NotFound

    def _put(self) -> None:
        frame = self._label_frame()
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ValueError("a Content-Length header is required") from None
        if not 0 < length <= MAX_BODY:
            raise ValueError(f"the body must be 1-{MAX_BODY} bytes, not {length}")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError as e:  # bad UTF-8 or bad JSON
            raise ValueError(f"the body is not JSON: {e}") from None
        if not isinstance(body, dict) or not set(body) <= {"visibility", "x", "y"}:
            raise ValueError('the body must be {"visibility": v, "x": x, "y": y}')
        label = HandLabel(body.get("visibility"), body.get("x"), body.get("y"))
        self._send_json(vars(self.server.store.set(frame, label)))

    def _delete(self) -> None:
        frame = self._label_frame()
        self.server.store.delete(frame)
        self._send_json({"deleted": frame})

    def _label_frame(self) -> int:
        """The frame a change is for, once it is known to come from this run."""
        match = re.fullmatch(r"/api/labels/(\d+)", urlsplit(self.path).path)
        if match is None:
            raise NotFound
        if self.headers.get(RUN_HEADER) != self.server.version:
            raise StaleRun
        return int(match[1])

    def _send_json(self, value: object) -> None:
        self._send(200, json.dumps(value).encode(), "application/json")

    def _send(
        self,
        status: int,
        body: bytes,
        kind: str = "text/plain; charset=utf-8",
        cache: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if isinstance(code, int) and code >= 400:  # errors only, not every frame
            super().log_request(code, size)


def run_label_tool(
    video: str | Path,
    out: str | Path,
    frames_dir: str | Path | None = None,
    port: int = 8765,
) -> None:
    """Prepare the frames, load or create the label CSV, and serve the page
    until Ctrl-C, holding ``<out>.lock`` throughout. Every change is on disk
    when the page is told it is saved, so stopping loses nothing."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out):
        if frames_dir is None and not Path(video).is_dir():
            frames_dir = out.with_suffix(".frames")
        frames = prepare_frames(video, frames_dir)
        store = LabelStore(
            out, len(frames.paths), frames.width, frames.height, frames.source
        )
        server = LabelServer(frames, store, port)
        port = server.server_address[1]
        print(
            f"{len(frames.paths)} frames of {frames.width}x{frames.height}, "
            f"{len(store.labels)} labelled, saved to {out}\n"
            f"Open http://localhost:{port}/ or http://127.0.0.1:{port}/ "
            "(Ctrl-C to stop)",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            store.close()  # waits for a change being written
        print("Stopped; every label the page reported saved is in the CSV.")
