"""Stage 1: Capture a short burst of frames from the robot camera.

Goal: grab NUM_FRAMES frames spanning roughly CAPTURE_DURATION_S seconds,
each tagged with the timestamp it was captured at. Later stages
(stabilize, composite) rely on these frames being in chronological order.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from projection_sorcery.camera_source import CameraSource, get_camera_source
from projection_sorcery.config import (
    CAPTURE_DURATION_S,
    NUM_FRAMES,
    RAW_FRAMES_DIR,
)


@dataclass
class CapturedFrame:
    image: np.ndarray  # BGR, as returned by cv2/gretchen
    timestamp: float
    index: int


# Capture `num_frames` frames from camera, spaced out over `duration_s`.
def capture_frames(
    camera: CameraSource,
    num_frames: int = NUM_FRAMES,
    duration_s: float = CAPTURE_DURATION_S,
) -> list[CapturedFrame]:

    captured = []

    while len(captured) < num_frames:
        (ret, frame, timestamp) = camera.get_frame()
        if ret:
            captured.append(CapturedFrame(frame, timestamp, index=len(captured)))
            if len(captured) != num_frames:
                time.sleep(duration_s / num_frames)

    return captured


def save_debug_frames(frames: list[CapturedFrame], out_dir=RAW_FRAMES_DIR) -> None:
    """Write each frame to out_dir for visual debugging.

    Hints:
    - Use cv2.imwrite. Frames are already in BGR, which is what cv2.imwrite expects.
    - Name files so they sort in capture order, e.g. f"frame_{f.index}_{f.timestamp}.png".
    - Make sure out_dir exists (Path.mkdir(parents=True, exist_ok=True)).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        filename = f"frame_{frame.index}_{frame.timestamp}.png"
        pathname = f"{out_dir}/{filename}"
        ret = cv2.imwrite(pathname, frame.image)
        if not ret:
            raise ValueError(f"imwrite failed for: {filename}")
         

def main():
    # TODO(you): wire it together:
    # 1. build a camera via get_camera_source()
    # 2. call capture_frames(...)
    # 3. call save_debug_frames(...)
    # 4. print a short summary (how many frames, over what real elapsed time)
    camera = get_camera_source()
    camera.start()
    frames = capture_frames(camera)
    camera.stop()
    save_debug_frames(frames)
    print(f"{NUM_FRAMES} frames were captured over {CAPTURE_DURATION_S} seconds.")


if __name__ == "__main__":
    main()
