"""Stage 2: Person segmentation with YOLO26-seg """

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from projection_sorcery.capture import CapturedFrame
from projection_sorcery.config import (
    COCO_CLASS_ID,
    RAW_FRAMES_DIR,
    SEGMENTED_DIR,
    YOLO_SEG_MODEL,
)


@dataclass
class SegmentedFrame:
    image: np.ndarray 
    timestamp: float
    index: int
    confidence: float


def segment_frames(frames: list[CapturedFrame], model) -> list[SegmentedFrame]:
    segmented = []
   
    for frame in frames:
        results = model(frame.image, classes=[COCO_CLASS_ID], verbose=False)
        result = results[0]
        if len(result.boxes) == 0:
            print(f"Nobody detected in frame {frame.index}")
            continue
        best_conf = result.boxes.conf.max()
        best_conf_idx = result.boxes.conf.argmax()
        best_mask = result.masks.data[best_conf_idx]

        mask_array = best_mask.cpu().numpy() # convert from tensor to array

        height, width = frame.image.shape[:2] 
        mask_array = cv2.resize(mask_array, (width, height))  # resize mask to fit frame

        alpha = (mask_array * 255).astype(np.uint8)  # convert from floats to integer pixel format

        bgra_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2BGRA)
        bgra_image[:, :, 3] = alpha  # get cutout of person from image

        segmented.append(SegmentedFrame(bgra_image, frame.timestamp, frame.index, best_conf.item()))
    
    return segmented


def save_debug_segmented(frames: list[SegmentedFrame], out_dir=SEGMENTED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        filename = f"frame_{frame.index}_{frame.timestamp}.png"
        pathname = f"{out_dir}/{filename}"
        ret = cv2.imwrite(pathname, frame.image)
        if not ret:
            raise ValueError(f"imwrite failed for: {filename}")

         
def load_captured_frames(in_dir=RAW_FRAMES_DIR) -> list[CapturedFrame]:
    files = list(in_dir.glob("frame*"))
    if not files:
        raise ValueError(f"no frames in {in_dir} - run capture.py first")

    frames = []

    for file in files:
        (_, index, timestamp) = file.stem.split("_")
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError(f"imread failed for: {file}")
        frame = CapturedFrame(image, float(timestamp), int(index))
        frames.append(frame)

    frames.sort(key=lambda frame: frame.index)

    return frames    


def main():
    yolo_model = YOLO(YOLO_SEG_MODEL)
    captured_frames = load_captured_frames()
    segmented_frames = segment_frames(captured_frames, yolo_model)
    save_debug_segmented(segmented_frames)
    print(f"Segmented {len(segmented_frames)} of {(len(captured_frames))} frames")

if __name__ == "__main__":
    main()
