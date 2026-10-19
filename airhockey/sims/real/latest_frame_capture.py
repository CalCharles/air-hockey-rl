import threading

import cv2


class LatestFrameCapture:
    """Drop-in for cv2.VideoCapture whose read() returns the newest frame.

    The V4L2 driver queues several frames (4 by default; CAP_PROP_BUFFERSIZE=1
    is clamped to 2 on the PS3 Eye), so a plain cap.read() after a control-step
    pause returns a frame that is ~4 frames (~130 ms at 30 fps) old. A daemon
    thread here drains the camera continuously and keeps only the latest frame.
    """

    def __init__(self, index, api_preference=cv2.CAP_V4L2):
        self._cap = cv2.VideoCapture(index, api_preference)
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
        self._ret, self._frame = False, None
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if not self._cap.isOpened():
                    break
                continue
            with self._lock:
                self._ret, self._frame = ret, frame
            self._first_frame.set()
        self._first_frame.set()

    def read(self, timeout=2.0):
        self._first_frame.wait(timeout)
        with self._lock:
            return self._ret, self._frame

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
