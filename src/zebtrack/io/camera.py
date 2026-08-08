import threading
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import structlog

from zebtrack.io.frame_source import FrameSource
from zebtrack.settings import settings

log = structlog.get_logger()


class Camera(FrameSource):
    def __init__(self):
        self._camera_index = settings.camera.index
        self.cap = cv2.VideoCapture(self._camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera at index {self._camera_index}")

        self._desired_width = settings.camera.desired_width
        self._desired_height = settings.camera.desired_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._desired_height)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Read once, here, before the reader thread exists. Querying the capture
        # backend from another thread while it is inside cap.read() can block.
        self._declared_fps = self.cap.get(cv2.CAP_PROP_FPS)
        log.info(
            "camera.initialized",
            index=self._camera_index,
            width=self.actual_width,
            height=self.actual_height,
        )

        self._lock = threading.Lock()
        self._latest_frame: Tuple[bool, np.ndarray | None] = (False, None)
        # Capture timestamp and sequence number of the frame currently held in
        # ``_latest_frame``. These travel WITH the frame; the old module-level
        # ``latency_logging.FRAME_T0`` global is gone (it was overwritten by
        # this thread on every read, including frames never consumed).
        self._latest_t0: float | None = None
        self._latest_seq: int = 0
        self._cam_seq: int = 0
        # Running estimate of the achieved camera rate, used to write the video
        # container at the true fps instead of the declared one.
        self._fps_t_first: float | None = None
        self._fps_t_last: float | None = None
        self._fps_n: int = 0
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

    def _reader_thread(self):
        """
        The main loop for the background thread that continuously reads
        frames and handles camera reconnections.
        """
        while not self._stopped.is_set():
            if not self.cap.isOpened():
                log.warning("camera.reconnect.start")
                self.cap.open(self._camera_index)
                if self.cap.isOpened():
                    log.info("camera.reconnect.success")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._desired_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._desired_height)
                else:
                    with self._lock:
                        self._latest_frame = (False, None)
                    time.sleep(2)
                    continue

            t_before = time.perf_counter()
            ret, frame = self.cap.read()
            t0 = time.perf_counter()
            # A read that takes this long is not a slow frame, it is a stalled
            # device. Say so, instead of letting the preview silently freeze on
            # the last good frame.
            if t0 - t_before > 2.0:
                log.warning("camera.read.stalled", seconds=round(t0 - t_before, 2))

            if not ret:
                self.cap.release()
                log.warning("camera.frame_read.failed")
                with self._lock:
                    self._latest_frame = (False, None)
                continue

            with self._lock:
                self._cam_seq += 1
                self._latest_frame = (ret, frame)
                self._latest_t0 = t0
                self._latest_seq = self._cam_seq
                if self._fps_t_first is None:
                    self._fps_t_first = t0
                self._fps_t_last = t0
                self._fps_n += 1
        log.info("camera.reader_thread.stopped")

    def get_frame(self) -> Tuple[bool, np.ndarray | None]:
        """
        Returns the most recent frame read by the background thread.
        """
        with self._lock:
            ret, frame = self._latest_frame
            return ret, frame.copy() if ret else None

    def get_frame_ts(self):
        """Like :meth:`get_frame`, but also returns the frame's own capture
        timestamp and the camera reader's sequence number for it.

        Returns ``(ret, frame, t_capture_perf, cam_seq)``. ``cam_seq`` counts
        camera reads, so the gap between consecutive consumed values is the
        number of camera frames skipped by the "latest frame wins" policy.
        """
        with self._lock:
            ret, frame = self._latest_frame
            t0, seq = self._latest_t0, self._latest_seq
            return ret, (frame.copy() if ret else None), t0, seq

    def measured_fps(self) -> float | None:
        """Achieved camera rate since start-up, or None if not yet estimable.

        Measured, not declared. ``settings.video_processing.fps`` is a request,
        not an observation, and on this rig the two differ by ~30%.
        """
        with self._lock:
            if self._fps_n < 2 or self._fps_t_first is None:
                return None
            span = self._fps_t_last - self._fps_t_first
            return (self._fps_n - 1) / span if span > 0 else None

    def release(self) -> None:
        """
        Signals the reader thread to stop and releases the camera resource.
        """
        self._stopped.set()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            # The reader is still inside cap.read(). Calling cap.release() now
            # would block on the same backend and hang the UI thread. The thread
            # is a daemon and the handle is freed at process exit.
            log.error("camera.release.reader_still_running")
            return
        if self.cap.isOpened():
            self.cap.release()
            log.info("camera.released")

    def get_properties(self) -> Dict[str, Any]:
        """
        Returns the actual properties of the camera feed.
        """
        return {
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self._declared_fps or settings.video_processing.fps,
            # Observed rate of the reader thread. Prefer this over "fps" for
            # anything that converts frames to milliseconds.
            "fps_measured": self.measured_fps(),
        }


if __name__ == "__main__":
    # Example usage for testing the camera module
    camera = None
    try:
        camera = Camera()
        print("Camera properties:", camera.get_properties())

        time.sleep(1)

        while True:
            ret, frame = camera.get_frame()
            if not ret:
                print("Failed to grab frame, waiting...")
                time.sleep(0.5)
                continue

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except IOError as e:
        print(e)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print("Test finished.")
