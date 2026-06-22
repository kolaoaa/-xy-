"""Background camera capture and YOLO inference workers."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Lock

from PyQt5.QtCore import QThread, pyqtSignal


class LatestVisionSource:
    """Thread-safe latest-frame adapter used by the sorting state machine."""

    def __init__(self):
        self._lock = Lock()
        self._frame = None
        self._detections = []

    def update_frame(self, frame) -> None:
        with self._lock:
            self._frame = self._copy_frame(frame)

    def update_detections(self, detections) -> None:
        with self._lock:
            self._detections = list(detections)

    def capture(self):
        with self._lock:
            return self._copy_frame(self._frame)

    def detect(self, frame):
        del frame
        with self._lock:
            return list(self._detections)

    @staticmethod
    def _copy_frame(frame):
        if frame is None:
            return None
        copy = getattr(frame, "copy", None)
        return copy() if callable(copy) else frame


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    camera_status = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.camera_index = int(camera_index)
        self._running = False

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.error_occurred.emit("OpenCV 未安装，请运行: pip install opencv-python")
            self.camera_status.emit(False)
            return

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.error_occurred.emit(f"无法打开摄像头 {self.camera_index}")
            self.camera_status.emit(False)
            return

        self._running = True
        self.camera_status.emit(True)
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    self.error_occurred.emit("摄像头读取失败")
                    self.msleep(100)
                    continue
                self.frame_ready.emit(frame)
                self.msleep(20)
        finally:
            capture.release()
            self.camera_status.emit(False)

    def stop(self) -> None:
        self._running = False
        self.wait(1000)


class VisionWorker(QThread):
    result_ready = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self._frames: Queue = Queue(maxsize=1)
        self._running = False

    def submit_frame(self, frame) -> None:
        try:
            self._frames.put_nowait(frame.copy())
        except Full:
            try:
                self._frames.get_nowait()
            except Empty:
                pass
            try:
                self._frames.put_nowait(frame.copy())
            except Full:
                pass

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                frame = self._frames.get(timeout=0.1)
            except Empty:
                continue
            try:
                detections = self.detector.detect(frame)
                annotated = self.detector.annotate(frame, detections)
                self.result_ready.emit(annotated, detections)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
                self.msleep(500)

    def stop(self) -> None:
        self._running = False
        self.wait(1500)
