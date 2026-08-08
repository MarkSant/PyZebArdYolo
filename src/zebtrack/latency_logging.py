"""latency_logging.py — closed-loop latency instrumentation (PyZebArdYolo).

Stdlib only. Writes three files per recording session, all into the session's
output folder:

  6_Latency_<base>.csv      one row per Arduino trigger
  7_FrameLedger_<base>.csv  one row per frame handed to the video writer
  8_LatencyMeta_<base>.json session-level metadata (measured fps, drops, config)

Design notes
------------
The previous version exposed a module-level ``FRAME_T0`` that the camera reader
thread overwrote on every ``cap.read()``. Because the reader runs free and
``Camera.get_frame()`` returns only the most recent frame, ``FRAME_T0`` almost
never referred to the frame that actually produced the decision. The resulting
``frame_to_ack_ms`` column was near-uniform over one frame interval and
correlated negatively with its own serial leg. That global is gone: the capture
timestamp is now carried with the frame and passed explicitly.

The frame ledger exists because neither ``live_frame_count`` nor the video frame
index is a camera frame index — frames are skipped at ``get_frame()`` and
dropped again at both queues. The ledger makes the video<->pipeline mapping
exact and makes drops visible instead of silent.
"""

import csv
import json
import os
import threading
import time

# Deprecated. Kept only so that any stale import does not raise. Nothing in the
# application writes or reads it any more; passing it to log_trigger is a no-op.
FRAME_T0 = None

_LOCK = threading.Lock()

_TRIG_COLUMNS = [
    "event_id",
    "wall_iso",
    "frame",            # live_frame_count of the frame that produced the decision
    "cam_seq",          # camera reader sequence number of that same frame
    "roi",              # square index (1-based), or "" if unknown
    "edge",             # "enter" | "exit"
    "token",            # command byte sent to the Arduino
    "t_capture_perf",   # perf_counter immediately after cap.read() returned
    "t_decision_perf",  # perf_counter at the end of Detector.process_frame
    "t_send_perf",      # perf_counter immediately before ser.write
    "t_ack_perf",       # perf_counter immediately after ser.readline returned
    "ack_ok",           # True if readline returned a non-empty line
    "ack_text",
    "capture_to_decision_ms",
    "decision_to_send_ms",
    "serial_act_ms",
    "frame_to_ack_ms",
]

_LEDGER_COLUMNS = ["video_write_index", "frame", "cam_seq", "t_capture_perf"]


class _Session:
    """Holds the open file handles and counters for one recording session."""

    def __init__(self, folder, base_name, meta=None):
        self.folder = folder
        self.base_name = base_name
        self.meta = dict(meta or {})
        self.event_id = 0
        self.video_write_index = 0
        self.video_drops = 0
        self.analysis_drops = 0
        self.t_first_capture = None
        self.t_last_capture = None
        self.n_captures = 0

        self._tf = open(
            os.path.join(folder, f"6_Latency_{base_name}.csv"),
            "w", newline="", encoding="utf-8",
        )
        self._tw = csv.writer(self._tf)
        self._tw.writerow(_TRIG_COLUMNS)
        self._tf.flush()

        self._lf = open(
            os.path.join(folder, f"7_FrameLedger_{base_name}.csv"),
            "w", newline="", encoding="utf-8",
        )
        self._lw = csv.writer(self._lf)
        self._lw.writerow(_LEDGER_COLUMNS)
        self._lf.flush()

    def close(self):
        for f in (self._tf, self._lf):
            try:
                f.flush()
                f.close()
            except Exception:
                pass


_SESSION: "_Session | None" = None


def start_session(folder, base_name, meta=None):
    """Open the latency files for a recording session. Safe to call twice."""
    global _SESSION
    with _LOCK:
        if _SESSION is not None:
            _SESSION.close()
        try:
            os.makedirs(folder, exist_ok=True)
            _SESSION = _Session(folder, base_name, meta)
        except Exception:
            _SESSION = None
    return _SESSION is not None


def stop_session(extra_meta=None):
    """Flush and close the session, writing the metadata sidecar."""
    global _SESSION
    with _LOCK:
        s = _SESSION
        _SESSION = None
    if s is None:
        return
    meta = dict(s.meta)
    meta.update(extra_meta or {})
    meta.update(
        {
            "n_triggers": s.event_id,
            "n_video_frames_written": s.video_write_index,
            "video_queue_drops": s.video_drops,
            "analysis_queue_drops": s.analysis_drops,
            "n_captures_consumed": s.n_captures,
            "fps_measured": measured_fps_from(s),
            "t_first_capture_perf": s.t_first_capture,
            "t_last_capture_perf": s.t_last_capture,
        }
    )
    try:
        path = os.path.join(s.folder, f"8_LatencyMeta_{s.base_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, default=str)
    except Exception:
        pass
    s.close()


def measured_fps_from(s):
    """Achieved consumption rate over the session, or None if undeterminable."""
    if s is None or s.n_captures < 2:
        return None
    if s.t_first_capture is None or s.t_last_capture is None:
        return None
    span = s.t_last_capture - s.t_first_capture
    return (s.n_captures - 1) / span if span > 0 else None


def note_capture(t_capture):
    """Record a consumed frame's capture timestamp (for the fps estimate)."""
    s = _SESSION
    if s is None:
        return
    with _LOCK:
        if s.t_first_capture is None:
            s.t_first_capture = t_capture
        s.t_last_capture = t_capture
        s.n_captures += 1


def note_drop(kind):
    """Count a dropped frame. kind is 'video' or 'analysis'."""
    s = _SESSION
    if s is None:
        return
    with _LOCK:
        if kind == "video":
            s.video_drops += 1
        else:
            s.analysis_drops += 1


def log_video_frame(frame, cam_seq, t_capture):
    """One ledger row per frame accepted by the video queue.

    Returns the 0-based index this frame will occupy in the mp4, which is what
    makes the video<->pipeline mapping exact.
    """
    s = _SESSION
    if s is None:
        return None
    try:
        with _LOCK:
            idx = s.video_write_index
            s.video_write_index += 1
            s._lw.writerow([idx, frame, cam_seq, f"{t_capture:.6f}"])
            s._lf.flush()
        return idx
    except Exception:
        return None


def log_trigger(
    cmd,
    t_send,
    t_ack,
    frame_t0=None,
    *,
    ack_ok=None,
    ack_text="",
    frame=None,
    cam_seq=None,
    roi=None,
    edge=None,
    t_decision=None,
):
    """One row per Arduino trigger.

    ``frame_t0`` must be the capture timestamp of the frame that produced this
    decision, passed explicitly by the caller. The old module-level global is
    no longer consulted.
    """
    s = _SESSION
    if s is None:
        return
    try:
        with _LOCK:
            s.event_id += 1
            eid = s.event_id
            serial_ms = (t_ack - t_send) * 1000.0
            f2a = (t_ack - frame_t0) * 1000.0 if frame_t0 else ""
            c2d = (t_decision - frame_t0) * 1000.0 if (t_decision and frame_t0) else ""
            d2s = (t_send - t_decision) * 1000.0 if t_decision else ""
            s._tw.writerow(
                [
                    eid,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "" if frame is None else frame,
                    "" if cam_seq is None else cam_seq,
                    "" if roi is None else roi,
                    "" if edge is None else edge,
                    cmd,
                    f"{frame_t0:.6f}" if frame_t0 else "",
                    f"{t_decision:.6f}" if t_decision else "",
                    f"{t_send:.6f}",
                    f"{t_ack:.6f}",
                    "" if ack_ok is None else bool(ack_ok),
                    ack_text,
                    f"{c2d:.3f}" if c2d != "" else "",
                    f"{d2s:.3f}" if d2s != "" else "",
                    f"{serial_ms:.3f}",
                    f"{f2a:.3f}" if f2a != "" else "",
                ]
            )
            s._tf.flush()
    except Exception:
        pass  # never take down the loop for logging
