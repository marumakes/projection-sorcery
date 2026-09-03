"""Stage 4: Cancel camera drift so every cutout shares one coordinate system.

The afterimage only reads as motion if pixel (x, y) means the same physical spot in
every frame. Any camera movement during the burst breaks that, and the trail comes out
jittery rather than as a clean arc.

Homographies are estimated from the RAW frames, because that is where the background
features live - the segmented frames have had their background alpha'd away. The
resulting transform is then applied to the BGRA cutouts.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from projection_sorcery.capture import CapturedFrame
from projection_sorcery.config import (
    RAW_FRAMES_DIR,
    SEGMENTED_DIR,
    STABILIZED_DIR,
)
from projection_sorcery.frame_io import (
    ImageMode,
    load_images,
    matched_by_index,
    save_frames,
)
from projection_sorcery.segment import SegmentedFrame

IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)
TRANSPARENT_BORDER = (0, 0, 0, 0)


@dataclass
class StabilizedFrame:
    image: np.ndarray  # BGRA cutout warped into frame-0 space
    timestamp: float
    index: int
    confidence: float
    homography: np.ndarray  # 3x3, maps this frame's pixels onto frame 0's


def estimate_homographies(raw_frames: list[CapturedFrame]) -> list[np.ndarray]:
    """Estimate the camera drift of each raw frame relative to raw_frames[0].

    TODO: fill this in.

    Sketch: detect ORB features (config.ORB_FEATURES) in frame 0 and in frame N,
    match them, and once at least config.MIN_MATCH_COUNT survive, recover the
    transform with cv2.findHomography(..., cv2.RANSAC). cv2.findTransformECC is the
    alternative if feature matching proves flaky on your background.

    Masking the person out before matching helps a lot - their limbs are the one
    part of the scene that genuinely moves, so they poison the estimate.

    Args:
        raw_frames: the burst in chronological order. Full BGR frames, background
            intact. raw_frames[0] is the reference everything warps onto.

    Returns:
        One 3x3 float64 matrix per input frame, in the same order. Element 0 must be
        the identity (frame 0 is the reference). Return IDENTITY_HOMOGRAPHY for any
        frame you cannot solve rather than None - apply_homographies treats that as
        "no drift" and the run completes instead of crashing.
    """
    raise NotImplementedError(
        "estimate_homographies is a stub - see the docstring, or use "
        "identity_homographies(raw_frames) to pass frames through unstabilised"
    )


# Escape hatch: treat the camera as perfectly still. Useful for testing the stages downstream.
def identity_homographies(raw_frames: list[CapturedFrame]) -> list[np.ndarray]:
    return [IDENTITY_HOMOGRAPHY.copy() for _ in raw_frames]


# Warp each BGRA cutout into frame-0 space. Areas pulled in from outside stay transparent.
def apply_homographies(
    segmented: list[SegmentedFrame],
    homographies: list[np.ndarray],
) -> list[StabilizedFrame]:
    if len(segmented) != len(homographies):
        raise ValueError(
            f"got {len(homographies)} homographies for {len(segmented)} frames - "
            f"they must correspond one-to-one"
        )

    stabilized = []

    for frame, homography in zip(segmented, homographies, strict=True):
        matrix = IDENTITY_HOMOGRAPHY if homography is None else homography
        height, width = frame.image.shape[:2]

        warped = cv2.warpPerspective(
            frame.image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=TRANSPARENT_BORDER,
        )

        stabilized.append(
            StabilizedFrame(warped, frame.timestamp, frame.index, frame.confidence, matrix)
        )

    return stabilized


def stabilize_frames(
    raw_frames: list[CapturedFrame],
    segmented: list[SegmentedFrame],
) -> list[StabilizedFrame]:
    return apply_homographies(segmented, estimate_homographies(raw_frames))


def save_debug_stabilized(frames: list[StabilizedFrame], out_dir=STABILIZED_DIR) -> None:
    save_frames(frames, out_dir)


def load_segmented_frames(in_dir=SEGMENTED_DIR) -> list[SegmentedFrame]:
    loaded = load_images(in_dir, ImageMode.WITH_ALPHA)

    # Confidence is not persisted to disk; it is only carried for downstream reporting.
    return [SegmentedFrame(image, timestamp, index, 0.0) for index, timestamp, image in loaded]


def load_raw_frames(in_dir=RAW_FRAMES_DIR) -> list[CapturedFrame]:
    loaded = load_images(in_dir, ImageMode.COLOR)

    return [CapturedFrame(image, timestamp, index) for index, timestamp, image in loaded]


def main():
    segmented = load_segmented_frames()
    raw_frames = load_raw_frames()

    # Only the raw frames that made it through segmentation are relevant here.
    pairs = list(matched_by_index(segmented, raw_frames))
    if len(pairs) != len(segmented):
        raise ValueError("some segmented frames have no matching raw frame - rerun capture")

    matched_raw = [raw for _, raw in pairs]

    stabilized = stabilize_frames(matched_raw, segmented)
    save_debug_stabilized(stabilized)
    print(f"Stabilized {len(stabilized)} frames onto frame {stabilized[0].index}")


if __name__ == "__main__":
    main()
