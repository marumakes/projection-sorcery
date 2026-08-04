"""Shared configuration for the Projection Sorcery pipeline."""

from pathlib import Path

# --- Robot device paths ---
# Mac camera index (confirmed working: integer index 0)
CAMERA_DEVICE_PATH = 0
MOTOR_DEVICE_PATH = "/dev/tty.usbserial-FT6RW6ZU"

# --- Camera source ---
CAMERA_BACKEND = "webcam"
WEBCAM_DEVICE_INDEX = 0

# --- Capture settings ---
NUM_FRAMES = 5
CAPTURE_DURATION_S = 0.3  # total time span across all NUM_FRAMES

# --- Debug output directories ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "debug_output"
RAW_FRAMES_DIR = DEBUG_OUTPUT_DIR / "raw_frames"
SEGMENTED_DIR = DEBUG_OUTPUT_DIR / "segmented"
POSE_DEBUG_DIR = DEBUG_OUTPUT_DIR / "pose_debug"
STABILIZED_DIR = DEBUG_OUTPUT_DIR / "stabilized"
COMPOSITE_DIR = DEBUG_OUTPUT_DIR / "composite"
ANIME_DIR = DEBUG_OUTPUT_DIR / "anime"


# --- Ultralytics segmentation ---
YOLO_SEG_MODEL = "yolo26n-seg.pt"
COCO_CLASS_ID = 0
