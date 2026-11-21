"""Hand labelling of the ball, one click per frame, in a local web page."""

from labeling.frames import FrameSet, prepare_frames
from labeling.server import LabelServer, run_label_tool
from labeling.store import HandLabel, LabelStore

__all__ = [
    "FrameSet",
    "HandLabel",
    "LabelServer",
    "LabelStore",
    "prepare_frames",
    "run_label_tool",
]
