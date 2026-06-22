"""Coordinates exclusive webcam ownership between GUI pages."""

from __future__ import annotations

from threading import Lock


class CameraOwnershipCoordinator:
    def __init__(self):
        self._lock = Lock()
        self._owner = None

    @property
    def owner(self):
        with self._lock:
            return self._owner

    def acquire(self, owner) -> bool:
        with self._lock:
            if self._owner is None or self._owner == owner:
                self._owner = owner
                return True
            return False

    def release(self, owner) -> bool:
        with self._lock:
            if self._owner != owner:
                return False
            self._owner = None
            return True

