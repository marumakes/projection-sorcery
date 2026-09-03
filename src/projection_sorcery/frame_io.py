"""Shared debug-frame file I/O for the pipeline stages.

Filenames follow the convention capture.py established: frame_{index}_{timestamp}.png
Stages that write BGRA cutouts must read them back with ImageMode.WITH_ALPHA,
otherwise OpenCV silently drops the alpha channel and the cutout turns opaque.
"""

from collections.abc import Iterable, Iterator, Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

FRAME_FILENAME_PREFIX = "frame"
FRAME_GLOB = f"{FRAME_FILENAME_PREFIX}_*.png"
FILENAME_FIELD_COUNT = 3


class DebugOutput(Enum):
    SAVE = auto()
    SKIP = auto()


class ImageMode(Enum):
    COLOR = auto()  # 3-channel BGR
    WITH_ALPHA = auto()  # 4-channel BGRA


_READ_FLAGS = {
    ImageMode.COLOR: cv2.IMREAD_COLOR,
    ImageMode.WITH_ALPHA: cv2.IMREAD_UNCHANGED,
}


# Every stage's frame dataclass exposes at least these three fields.
class FrameLike(Protocol):
    image: np.ndarray
    timestamp: float
    index: int


def frame_filename(index: int, timestamp: float) -> str:
    return f"{FRAME_FILENAME_PREFIX}_{index}_{timestamp}.png"


# Write (index, timestamp, image) triples into out_dir, creating it if needed.
def save_images(items: Iterable[tuple[int, float, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, timestamp, image in items:
        filename = frame_filename(index, timestamp)
        if not cv2.imwrite(str(out_dir / filename), image):
            raise ValueError(f"imwrite failed for: {filename}")


def save_frames(frames: Sequence[FrameLike], out_dir: Path) -> None:
    save_images(((f.index, f.timestamp, f.image) for f in frames), out_dir)


# Read every frame file in in_dir, returned as (index, timestamp, image) in index order.
def load_images(in_dir: Path, mode: ImageMode) -> list[tuple[int, float, np.ndarray]]:
    files = sorted(in_dir.glob(FRAME_GLOB))
    if not files:
        raise ValueError(f"no frames in {in_dir} - run the previous stage first")

    loaded = [(*_parse_filename(file), _read_image(file, mode)) for file in files]
    loaded.sort(key=lambda item: item[0])

    return loaded


def _parse_filename(file: Path) -> tuple[int, float]:
    fields = file.stem.split("_")
    if len(fields) != FILENAME_FIELD_COUNT:
        raise ValueError(f"unexpected frame filename: {file.name}")

    (_, index, timestamp) = fields

    return (int(index), float(timestamp))


def _read_image(file: Path, mode: ImageMode) -> np.ndarray:
    image = cv2.imread(str(file), _READ_FLAGS[mode])
    if image is None:
        raise ValueError(f"imread failed for: {file}")

    return image


# Clear a debug directory of previous frames so a short run cannot be read as a long one.
def clear_debug_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return

    for file in out_dir.glob(FRAME_GLOB):
        file.unlink()


def matched_by_index(
    frames: Sequence[FrameLike],
    others: Sequence[FrameLike],
) -> Iterator[tuple[FrameLike, FrameLike]]:
    """Pair up two stages' frames on their shared index, skipping unmatched ones."""
    by_index = {other.index: other for other in others}

    for frame in frames:
        other = by_index.get(frame.index)
        if other is None:
            continue

        yield (frame, other)
