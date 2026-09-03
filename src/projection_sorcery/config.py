"""Shared configuration for the Projection Sorcery pipeline."""

from enum import IntEnum
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


# --- Ultralytics pose ---
YOLO_POSE_MODEL = "yolo26n-pose.pt"

# Keypoints below this are treated as "not seen" and ignored when measuring travel.
KEYPOINT_CONF_THRESHOLD = 0.5


# Index layout of the COCO 17-keypoint skeleton the pose model emits.
class Keypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Extremities travel furthest in a swing or step, so they are the clearest
# signal for "did this frame actually move" — torso keypoints barely shift.
TRAVEL_KEYPOINTS = (
    Keypoint.LEFT_WRIST,
    Keypoint.RIGHT_WRIST,
    Keypoint.LEFT_ANKLE,
    Keypoint.RIGHT_ANKLE,
)

# --- Trail selection ---
# A frame joins the trail only once a limb has moved this far since the last kept
# frame, so afterimages spread evenly instead of bunching where motion was slowest.
MIN_TRAVEL_PX = 25.0
TARGET_TRAIL_FRAMES = 5
MIN_TRAIL_FRAMES = 2

# --- Stabilisation ---
ORB_FEATURES = 2000
MIN_MATCH_COUNT = 10

# --- Composite ---
COMPOSITE_FILENAME = "composite.png"
MAX_ALPHA = 255.0
TRAIL_OPACITY_OLDEST = 0.15
TRAIL_OPACITY_NEWEST = 1.0

# --- Anime stylisation ---
# TODO: describe the JJK / anime look you want applied to the composited trail.
ANIME_PROMPT = ""
ANIME_MODEL = "gpt-image-1"
ANIME_IMAGE_SIZE = "1024x1024"
ANIME_FILENAME = "anime.png"
ANIME_MAX_RETRIES = 3
ANIME_RETRY_BACKOFF_S = 2.0
OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
