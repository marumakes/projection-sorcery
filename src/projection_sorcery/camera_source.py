from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from projection_sorcery.config import (
    CAMERA_BACKEND,
    CAMERA_DEVICE_PATH,
    MOTOR_DEVICE_PATH,
    NUM_FRAMES,
    WEBCAM_DEVICE_INDEX,
)


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def get_frame(self) -> tuple[bool, np.ndarray, float]:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass


class GretchenCameraSource(CameraSource):
    def __init__(self, camera_device_path, motor_device_path):
        self.camera_device_path = camera_device_path
        self.motor_device_path = motor_device_path
        self._robot = None

    def start(self):
        from gretchen.robot import Robot

        self._robot = Robot(self.motor_device_path, self.camera_device_path)
        self._robot.start_camera()

    def get_frame(self):
        return self._robot.camera.getImage()

    def stop(self):
        self._robot.camera.vc.release()


class WebcamCameraSource(CameraSource):
    def __init__(self, device_index=0):
        self._device_index = device_index
        self._vcam = None

    def start(self):
        import cv2
        self._vcam = cv2.VideoCapture(self._device_index)

        # webcam captures and discards 2 frames to stabilise sensors  
        for x in range(2): 
            self._vcam.read()


    def get_frame(self):
        ret, frame = self._vcam.read()
        return (ret, frame, datetime.now().timestamp())

    def stop(self):
        self._vcam.release()


class VideoFileCameraSource(CameraSource):
    """Samples an uploaded video instead of a live camera.

    NUM_FRAMES sample points are spread evenly across the whole clip - the video's own
    length stands in for CAPTURE_DURATION_S, since an uploaded clip has no live "now" to
    pace a burst against.
    """

    def __init__(self, video_path: str, num_frames: int = NUM_FRAMES):
        self._video_path = video_path
        self._num_frames = num_frames
        self._vcam = None
        self._frame_positions: list[int] = []
        self._fps = 0.0
        self._next_frame = 0

    def start(self):
        import cv2

        self._vcam = cv2.VideoCapture(self._video_path)
        if not self._vcam.isOpened():
            raise ValueError(f"could not open video file: {self._video_path}")

        total_frames = int(self._vcam.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self._num_frames:
            raise ValueError(
                f"{self._video_path} has {total_frames} frames, "
                f"need at least {self._num_frames}"
            )

        self._fps = self._vcam.get(cv2.CAP_PROP_FPS) or 1.0
        self._frame_positions = _evenly_spaced(total_frames, self._num_frames)
        self._next_frame = 0

    def get_frame(self):
        import cv2

        if self._next_frame >= len(self._frame_positions):
            return (False, None, 0.0)

        position = self._frame_positions[self._next_frame]
        self._next_frame += 1

        self._vcam.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self._vcam.read()

        return (ret, frame, position / self._fps)

    def stop(self):
        self._vcam.release()


# NUM_FRAMES positions spanning [0, total_frames - 1], inclusive of both ends.
def _evenly_spaced(total_frames: int, num_frames: int) -> list[int]:
    if num_frames == 1:
        return [0]

    step = (total_frames - 1) / (num_frames - 1)

    return [round(i * step) for i in range(num_frames)]


def get_camera_source(video_path: str | None = None) -> CameraSource:
    if video_path is not None:
        return VideoFileCameraSource(video_path)
    elif CAMERA_BACKEND == "gretchen":
        return GretchenCameraSource(CAMERA_DEVICE_PATH, MOTOR_DEVICE_PATH)
    elif CAMERA_BACKEND == "webcam":
        return WebcamCameraSource(WEBCAM_DEVICE_INDEX)
    else:
        raise TypeError(f"Unknown CAMERA_BACKEND: {CAMERA_BACKEND!r}")
