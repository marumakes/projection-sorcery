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
CAPTURE_DURATION_S = 0.5  # total time span across all NUM_FRAMES

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
TRAIL_OPACITY_OLDEST = 0.75
TRAIL_OPACITY_NEWEST = 1.0

# --- Anime stylisation ---
ANIME_PROMPT =  """Transform the supplied composite image into a dark, modern shōnen anime-style illustration, with a two-tier rendering split between the newest, fully opaque figure and the older, translucent afterimages.

The input image is a composite of a person moving across multiple frames. The oldest frames have the lowest opacity, while the most recent frame is fully opaque. The existing motion trail is intentional and is a central part of the image.

Treat the supplied image as the source of truth for the scene, composition, character movement, and motion trail. Stylise the image rather than creating a new composition.

### Newest figure (fully opaque position)

* Thin, clean linework
* Flat/cel-style shading with hard-edged shadow shapes — no soft airbrushed gradients or glossy highlight blooms on the skin
* Full, saturated character colours matching the original clothing and skin tones
* Cinematic lighting, but shaped as flat shadow/light regions rather than smooth gradients
* Realistic human proportions and anatomy
* Face: angular jaw and cheekbones rather than a soft rounded face shape
* Eyes: small and thin relative to typical shōnen anime, sparse eyelash linework, at most one small highlight per eye — avoid large glossy eyes with heavy lash detail or sparkle
* Expression rendered with restrained, mostly static linework rather than exaggerated eyebrow/mouth shapes, even when the pose is dynamic
* Subtle anime facial features rather than exaggerated anime or webtoon-filter proportions

### Older afterimages (translucent trail)

Render every afterimage except the newest as a monochrome blue/cyan video-ghost, not as a faded copy of the fully coloured figure.

* Strip away local colour entirely — silhouette and linework only, tinted a uniform blue/cyan
* Add video-ghosting artefacts: horizontal scanlines, faint interlacing tear, mild chromatic fringing, as if it were a delayed video projection of the figure rather than a solid body
* Keep edges soft and glowing rather than flat-filled, so each afterimage reads as a translucent projection, not a solid duplicate character
* Preserve the existing opacity progression (older = fainter) on top of this treatment

Every ghost shape you draw must trace a faint duplicate or blurred region that is already visible in the supplied input at that same image location — even where it is very faint, look closely for it rather than assuming it is absent. Do not invent, add, or substitute a decorative or generic "afterimage" motif (for example a repeating row of separate silhouette blobs) that is not directly grounded in content already present in the input.

Where a limb or body part appears as a single swept or motion-blurred streak in the input, rather than as distinct duplicated silhouettes, render that as one continuous blue/cyan streaked ghost following the same sweep path — do not break a single continuous sweep into multiple separate floating shapes.

### Background / environment

* Simplified but detailed-enough anime background
* Muted, restrained colours
* Energetic, dramatic, cinematic overall mood
* Dark modern shōnen anime aesthetic

### Preserve

Preserve the character's position, pose, movement, and overall appearance in the newest, fully opaque figure.

Preserve the number of afterimages and their relative positions. The motion direction and progression between the afterimages must remain clear.

Preserve the camera perspective and overall composition.

Preserve the general layout of the environment and the position of major objects within the scene.

Preserve the time of day, weather, and overall lighting condition shown in the input (for example: overcast daylight stays overcast daylight). Stylise the existing light into flat/cel-shaded anime lighting without changing when in the day it is or inventing a different sky, weather, or light source than what is already there.

### What may change

Change the photographic rendering into an anime illustration for the newest figure.

Change every older afterimage from a coloured figure into a monochrome blue/cyan scanline ghost as described above.

Change the real-world lighting into stylised flat/cel-shaded anime lighting, matching the same time of day, weather, and light source already present in the photo.

Stylise the background and surrounding environment so that it looks like part of the same anime illustration. Simplify photographic details where appropriate, while keeping the general layout, perspective, and major environmental elements recognisable.

Adapt colours, textures, and environmental details of the background to fit the dark shōnen anime aesthetic, aside from the afterimages, which must stay monochrome blue/cyan.

### Do not

* Add extra people
* Add extra limbs or body parts
* Change the character's pose
* Change the character's movement
* Remove any afterimages
* Add any afterimages
* Change the number of afterimages
* Reposition the afterimages
* Render any afterimage in full colour — only the newest, fully opaque figure keeps full colour
* Invent a decorative or generic afterimage pattern (e.g. a repeating row of separate silhouette shapes) that is not directly traceable to a faint duplicate or blur already present in the input
* Split a single swept/motion-blurred limb into multiple separate floating shapes — keep it as one continuous streak
* Distort or obscure the motion trail's silhouette or direction
* Invent a different time of day, weather, or sky than what is already in the input (e.g. turning overcast daylight into a sunset or night sky)
* Change the camera angle or perspective
* Crop the image
* Replace the scene with a different environment
* Add new objects that were not present in the original scene

The result should look like the original composite image has been professionally redrawn as a dark, cinematic shōnen anime illustration, where the newest figure is fully rendered in colour and every earlier afterimage is a monochrome blue/cyan scanline video-ghost trailing behind it.
"""
ANIME_MODEL = "gpt-image-2.5-sunburst"
ANIME_IMAGE_SIZE = "1024x1024"
ANIME_FILENAME = "anime.png"
ANIME_MAX_RETRIES = 3
ANIME_RETRY_BACKOFF_S = 2.0
OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
