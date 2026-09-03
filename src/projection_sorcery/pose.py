"""Stage 3: Pose-based trail selection.

Runs YOLO pose over the captured burst and keeps only the frames where the subject
actually moved since the last kept frame. A burst captured at a fixed interval
bunches afterimages wherever the subject was slowest; selecting on limb travel
instead of on time spreads them evenly across the motion.

Runs before segmentation so YOLO-seg only pays for the frames that survive.
"""

from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from projection_sorcery.capture import CapturedFrame
from projection_sorcery.config import (
    COCO_CLASS_ID,
    KEYPOINT_CONF_THRESHOLD,
    MIN_TRAIL_FRAMES,
    MIN_TRAVEL_PX,
    POSE_DEBUG_DIR,
    RAW_FRAMES_DIR,
    TARGET_TRAIL_FRAMES,
    TRAVEL_KEYPOINTS,
    YOLO_POSE_MODEL,
    Keypoint,
)
from projection_sorcery.frame_io import ImageMode, load_images, save_images

TRAVEL_KEYPOINT_INDICES = [int(keypoint) for keypoint in TRAVEL_KEYPOINTS]
ALL_KEYPOINT_INDICES = [int(keypoint) for keypoint in Keypoint]


@dataclass
class PosedFrame:
    image: np.ndarray
    timestamp: float
    index: int
    keypoints: np.ndarray  # (17, 2) float32, NaN where below KEYPOINT_CONF_THRESHOLD
    overlay: np.ndarray  # skeleton render, debug artefact only
    travel: float = field(default=0.0)  # px since previous kept frame, set by selection


# Detect the most confident person in each frame. Frames with nobody in them are dropped.
def detect_poses(frames: list[CapturedFrame], model: YOLO) -> list[PosedFrame]:
    posed = []

    for frame in frames:
        results = model(frame.image, classes=[COCO_CLASS_ID], verbose=False)
        result = results[0]

        if len(result.boxes) == 0 or result.keypoints is None:
            print(f"No pose detected in frame {frame.index}")
            continue

        best_idx = int(result.boxes.conf.argmax())
        keypoints = _confident_keypoints(result, best_idx)

        posed.append(
            PosedFrame(
                image=frame.image,
                timestamp=frame.timestamp,
                index=frame.index,
                keypoints=keypoints,
                overlay=result.plot(),
            )
        )

    return posed


# Keep frame 0, then each frame whose limbs have moved MIN_TRAVEL_PX since the last keeper.
def select_trail_frames(
    posed: list[PosedFrame],
    min_travel_px: float = MIN_TRAVEL_PX,
    target_frames: int = TARGET_TRAIL_FRAMES,
) -> list[PosedFrame]:
    if not posed:
        raise ValueError("no frames with a detected pose - nobody was in shot")

    kept = [posed[0]]

    for frame in posed[1:]:
        if len(kept) == target_frames:
            break

        travel = _travel_between(kept[-1].keypoints, frame.keypoints)
        if travel < min_travel_px:
            continue

        frame.travel = travel
        kept.append(frame)

    if len(kept) < MIN_TRAIL_FRAMES:
        raise ValueError(
            f"only {len(kept)} frame(s) cleared {min_travel_px}px of travel - "
            f"move faster through the burst, lengthen CAPTURE_DURATION_S, "
            f"or lower MIN_TRAVEL_PX"
        )

    return kept


# Keypoints the model was unsure about become NaN so they cannot fake a large travel.
def _confident_keypoints(result, person_idx: int) -> np.ndarray:
    xy = result.keypoints.xy[person_idx].cpu().numpy().astype(np.float32)

    confidences = result.keypoints.conf
    if confidences is None:
        return xy

    per_keypoint = confidences[person_idx].cpu().numpy()

    return np.where(per_keypoint[:, np.newaxis] >= KEYPOINT_CONF_THRESHOLD, xy, np.nan)


# Furthest-travelled keypoint between two poses. NaN keypoints drop out of the comparison.
# Extremities are preferred, but a close webcam framing crops them out entirely, so fall
# back to whatever both poses did see - otherwise travel is always 0 and nothing is kept.
def _travel_between(previous: np.ndarray, current: np.ndarray) -> float:
    for indices in (TRAVEL_KEYPOINT_INDICES, ALL_KEYPOINT_INDICES):
        travel = _max_displacement(previous, current, indices)
        if travel is not None:
            return travel

    return 0.0


# None when the two poses share no confidently-seen keypoint at these indices.
def _max_displacement(
    previous: np.ndarray,
    current: np.ndarray,
    indices: list[int],
) -> float | None:
    deltas = current[indices] - previous[indices]
    distances = np.linalg.norm(deltas, axis=1)
    valid = distances[~np.isnan(distances)]

    if valid.size == 0:
        return None

    return float(valid.max())


def save_debug_poses(frames: list[PosedFrame], out_dir=POSE_DEBUG_DIR) -> None:
    save_images(((f.index, f.timestamp, f.overlay) for f in frames), out_dir)


def load_captured_frames(in_dir=RAW_FRAMES_DIR) -> list[CapturedFrame]:
    loaded = load_images(in_dir, ImageMode.COLOR)

    return [CapturedFrame(image, timestamp, index) for index, timestamp, image in loaded]


def main():
    pose_model = YOLO(YOLO_POSE_MODEL)
    captured_frames = load_captured_frames()
    posed_frames = detect_poses(captured_frames, pose_model)
    trail_frames = select_trail_frames(posed_frames)
    save_debug_poses(posed_frames)

    kept = ", ".join(str(frame.index) for frame in trail_frames)
    print(f"Kept {len(trail_frames)} of {len(captured_frames)} frames for the trail: {kept}")


if __name__ == "__main__":
    main()
