"""Stage 5: Stack the stabilised cutouts into a single afterimage frame.

Cutouts are painted oldest-first onto the frame-0 plate, so the newest lands on top
and the trail fades backwards through time behind the subject.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from projection_sorcery.config import (
    COMPOSITE_DIR,
    COMPOSITE_FILENAME,
    MAX_ALPHA,
    RAW_FRAMES_DIR,
    STABILIZED_DIR,
    TRAIL_OPACITY_NEWEST,
    TRAIL_OPACITY_OLDEST,
)
from projection_sorcery.frame_io import ImageMode, load_images
from projection_sorcery.stabilize import StabilizedFrame

BGR_CHANNELS = 3
ALPHA_CHANNEL = 3


@dataclass
class CompositeResult:
    image: np.ndarray  # BGR
    frame_count: int

# Opacity for each cutout in the trail, oldest first.
def trail_opacities(count: int) -> list[float]:
    if count == 1:
        return [TRAIL_OPACITY_NEWEST]

    opacities = []

    for x in range(count):
        t = x / (count - 1)
        opacity = (TRAIL_OPACITY_OLDEST + (TRAIL_OPACITY_NEWEST - TRAIL_OPACITY_OLDEST) * t * t)
        opacities.append(opacity)

    return opacities
    

# Even fade from oldest to newest. The plainest ramp that produces a visible trail.
def linear_opacities(count: int) -> list[float]:
    if count == 1:
        return [TRAIL_OPACITY_NEWEST]

    ramp = np.linspace(TRAIL_OPACITY_OLDEST, TRAIL_OPACITY_NEWEST, count)

    return [float(value) for value in ramp]


# Paint the trail onto the background plate, oldest first so the newest sits on top.
def composite_trail(
    background: np.ndarray,
    stabilized: list[StabilizedFrame],
) -> CompositeResult:
    if not stabilized:
        raise ValueError("no stabilized frames to composite")

    opacities = trail_opacities(len(stabilized))
    if len(opacities) != len(stabilized):
        raise ValueError(
            f"trail_opacities returned {len(opacities)} values for {len(stabilized)} frames"
        )

    canvas = background[:, :, :BGR_CHANNELS].copy()

    for frame, opacity in zip(stabilized, opacities, strict=True):
        canvas = _alpha_paste(canvas, frame.image, opacity)

    return CompositeResult(canvas, len(stabilized))


# Standard alpha-over of a BGRA cutout onto a BGR canvas, scaled by opacity.
def _alpha_paste(canvas: np.ndarray, cutout: np.ndarray, opacity: float) -> np.ndarray:
    if canvas.shape[:2] != cutout.shape[:2]:
        raise ValueError(f"size mismatch: canvas {canvas.shape[:2]} vs cutout {cutout.shape[:2]}")

    alpha = (cutout[:, :, ALPHA_CHANNEL].astype(np.float32) / MAX_ALPHA) * opacity
    alpha = alpha[:, :, np.newaxis]

    foreground = cutout[:, :, :BGR_CHANNELS].astype(np.float32)
    blended = foreground * alpha + canvas.astype(np.float32) * (1.0 - alpha)

    return blended.astype(np.uint8)


def save_debug_composite(result: CompositeResult, out_dir=COMPOSITE_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / COMPOSITE_FILENAME

    if not cv2.imwrite(str(path), result.image):
        raise ValueError(f"imwrite failed for: {path}")

    return path


def load_stabilized_frames(in_dir=STABILIZED_DIR) -> list[StabilizedFrame]:
    loaded = load_images(in_dir, ImageMode.WITH_ALPHA)

    return [
        StabilizedFrame(image, timestamp, index, 0.0, np.eye(3))
        for index, timestamp, image in loaded
    ]


# The plate the trail is painted onto: the earliest raw frame, which is also
# the coordinate space stabilisation warped everything into.
def load_background_plate(in_dir=RAW_FRAMES_DIR) -> np.ndarray:
    loaded = load_images(in_dir, ImageMode.COLOR)

    return loaded[0][2]


def main():
    stabilized = load_stabilized_frames()
    background = load_background_plate()
    result = composite_trail(background, stabilized)
    path = save_debug_composite(result)
    print(f"Composited {result.frame_count} afterimages into {path}")


if __name__ == "__main__":
    main()
