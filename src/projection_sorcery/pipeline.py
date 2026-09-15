"""Runs the whole Projection Sorcery pipeline end to end, in memory.

Each stage module still has its own main() for working on that stage alone against
debug_output/. This is the path that does a real run:

    capture -> pose -> segment -> stabilize -> composite -> anime
"""

import argparse
from enum import Enum, auto

from ultralytics import YOLO

from projection_sorcery.anime import build_client, save_anime, stylise
from projection_sorcery.camera_source import get_camera_source
from projection_sorcery.capture import capture_frames, save_debug_frames
from projection_sorcery.composite import composite_trail, save_debug_composite
from projection_sorcery.config import (
    ANIME_DIR,
    COMPOSITE_DIR,
    POSE_DEBUG_DIR,
    RAW_FRAMES_DIR,
    SEGMENTED_DIR,
    STABILIZED_DIR,
    YOLO_POSE_MODEL,
    YOLO_SEG_MODEL,
)
from projection_sorcery.frame_io import DebugOutput, clear_debug_dir, matched_by_index
from projection_sorcery.pose import detect_poses, save_debug_poses, select_trail_frames
from projection_sorcery.segment import save_debug_segmented, segment_frames
from projection_sorcery.stabilize import save_debug_stabilized, stabilize_frames

DEBUG_DIRS = (
    RAW_FRAMES_DIR,
    SEGMENTED_DIR,
    POSE_DEBUG_DIR,
    STABILIZED_DIR,
    COMPOSITE_DIR,
    ANIME_DIR,
)


class AnimeStage(Enum):
    RUN = auto()
    SKIP = auto()


def run(
    debug: DebugOutput = DebugOutput.SAVE,
    anime_stage: AnimeStage = AnimeStage.RUN,
) -> None:
    if debug is DebugOutput.SAVE:
        # Stale frames from a longer previous run would otherwise be picked up
        # by the per-stage mains and read as part of this run.
        for directory in DEBUG_DIRS:
            clear_debug_dir(directory)

    frames = _capture(debug)
    trail = _select_trail(frames, debug)
    segmented = _segment(trail, debug)
    stabilized = _stabilize(trail, segmented, debug)
    composite = _composite(trail, stabilized, debug)

    if anime_stage is AnimeStage.SKIP:
        print("Skipping the anime stage.")
        return

    stylised = stylise(composite.image, build_client())
    path = save_anime(stylised)
    print(f"Wrote stylised frame to {path}")


def _capture(debug: DebugOutput):
    camera = get_camera_source()
    camera.start()
    try:
        frames = capture_frames(camera)
    finally:
        camera.stop()

    if debug is DebugOutput.SAVE:
        save_debug_frames(frames)

    print(f"Captured {len(frames)} frames.")

    return frames


def _select_trail(frames, debug: DebugOutput):
    posed = detect_poses(frames, YOLO(YOLO_POSE_MODEL))
    trail = select_trail_frames(posed)

    if debug is DebugOutput.SAVE:
        save_debug_poses(posed)

    kept = ", ".join(str(frame.index) for frame in trail)
    print(f"Kept {len(trail)} of {len(frames)} frames for the trail: {kept}")

    return trail


def _segment(trail, debug: DebugOutput):
    segmented = segment_frames(trail, YOLO(YOLO_SEG_MODEL))
    if not segmented:
        raise ValueError("segmentation found nobody in any trail frame")

    if debug is DebugOutput.SAVE:
        save_debug_segmented(segmented)

    print(f"Segmented {len(segmented)} of {len(trail)} trail frames.")

    return segmented


def _stabilize(trail, segmented, debug: DebugOutput):
    # Segmentation can drop frames, so realign the raw frames against what survived.
    matched_raw = [raw for _, raw in matched_by_index(segmented, trail)]

    stabilized = stabilize_frames(matched_raw, segmented)

    if debug is DebugOutput.SAVE:
        save_debug_stabilized(stabilized)

    print(f"Stabilized {len(stabilized)} frames onto frame {stabilized[0].index}.")

    return stabilized


def _composite(trail, stabilized, debug: DebugOutput):
    # The newest frame is the plate the trail is painted onto, so it's the one baked into
    # the canvas at full visibility "for free" - which is correct, since it's supposed to be
    # fully opaque anyway. Using the OLDEST frame here instead (as this used to) makes the
    # oldest cutout's own alpha-paste a no-op (there's no background left to blend with under
    # its own baked-in pixels), so it always renders fully solid regardless of its assigned
    # opacity - leaving two equally-opaque figures with no way to tell which one is newest.
    #
    # This assumes the newest raw frame shares the same coordinate space as the stabilized
    # cutouts, which holds for identity homographies (a static webcam/video source) but not
    # for a camera that genuinely moved (the "gretchen" backend) - there, this plate would
    # need warping into frame 0's space the same way the cutouts are.
    result = composite_trail(trail[-1].image, stabilized)

    if debug is DebugOutput.SAVE:
        path = save_debug_composite(result)
        print(f"Composited {result.frame_count} afterimages into {path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run the Projection Sorcery pipeline.")
    parser.add_argument(
        "--skip-anime",
        action="store_true",
        help="stop after the composite, without spending an API call",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="do not write intermediate frames to debug_output/",
    )
    args = parser.parse_args()

    run(
        debug=DebugOutput.SKIP if args.no_debug else DebugOutput.SAVE,
        anime_stage=AnimeStage.SKIP if args.skip_anime else AnimeStage.RUN,
    )


if __name__ == "__main__":
    main()
