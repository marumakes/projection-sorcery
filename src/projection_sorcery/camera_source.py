from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from projection_sorcery.config import (
    CAMERA_BACKEND,
    CAMERA_DEVICE_PATH,
    MOTOR_DEVICE_PATH,
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


def get_camera_source() -> CameraSource:
    if CAMERA_BACKEND == "gretchen":
        return GretchenCameraSource(CAMERA_DEVICE_PATH, MOTOR_DEVICE_PATH)
    elif CAMERA_BACKEND == "webcam":
        return WebcamCameraSource(WEBCAM_DEVICE_INDEX)
    else:
        raise TypeError(f"Unknown CAMERA_BACKEND: {CAMERA_BACKEND!r}")
