"""Regression tests for the closed-loop latency instrumentation.

These exist because two defects shipped undetected and were only caught by
statistical audit of the output CSV, months later:

  D1  ``reset_input_buffer()`` was called *after* ``readline()``, so every
      trigger but the first measured a buffered read of the previous command's
      ACK (~0.5 ms) instead of a serial round trip. 346 of 354 logged rows were
      physically impossible at 9600 baud.
  D2  the frame capture timestamp was a module-level global overwritten by a
      free-running camera thread, so it referred to a later frame than the one
      that produced the decision.

Both are cheap to assert and neither was covered.
"""

import csv
import json
import os
import time

import pytest

from zebtrack import latency_logging
from zebtrack.io.arduino import Arduino


class FakeSerial:
    """Minimal serial stand-in that records the order of operations.

    Models the real failure mode: the device replies asynchronously, so a line
    from the *previous* command is still sitting in the buffer unless the caller
    drains it before writing.
    """

    def __init__(self, ack=b"Red LED 1 ON\n", ack_delay_s=0.015):
        self.is_open = True
        self.calls = []
        self._buffer = [b"stale line from previous command\n"]
        self._ack = ack
        self._ack_delay = ack_delay_s

    def write(self, payload):
        self.calls.append("write")
        time.sleep(self._ack_delay)          # transit + firmware turnaround
        self._buffer.append(self._ack)
        return len(payload)

    def readline(self):
        self.calls.append("readline")
        return self._buffer.pop(0) if self._buffer else b""

    def reset_input_buffer(self):
        self.calls.append("reset_input_buffer")
        self._buffer.clear()

    def close(self):
        self.is_open = False


def _make_arduino():
    ard = Arduino(port="FAKE", baud_rate=9600)
    ard.ser = FakeSerial()
    return ard


def test_buffer_is_drained_before_write_not_after():
    """D1 regression: draining after the read yields the previous ACK."""
    ard = _make_arduino()
    ard.send_command(1)
    order = ard.ser.calls
    assert order.index("reset_input_buffer") < order.index("write"), (
        f"buffer must be drained before the write; got {order}"
    )


def test_serial_leg_reflects_a_real_round_trip(tmp_path):
    """The measured serial leg must include the device turnaround, not a
    buffered read. Sub-millisecond values are the D1 signature."""
    latency_logging.start_session(str(tmp_path), "unit")
    ard = _make_arduino()
    t0 = time.perf_counter()
    for _ in range(5):
        assert ard.send_command(1, frame_t0=t0, t_decision=t0 + 0.01, frame=10) is True
    latency_logging.stop_session()

    rows = list(csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8")))
    assert len(rows) == 5
    serial_ms = [float(r["serial_act_ms"]) for r in rows]
    assert all(s > 10.0 for s in serial_ms), (
        f"sub-millisecond serial legs indicate a stale buffered read: {serial_ms}"
    )
    assert all(r["ack_ok"] == "True" for r in rows)
    assert all(r["ack_text"] == "Red LED 1 ON" for r in rows)


def test_non_ok_ack_is_accepted():
    """The LED firmware never replies 'OK'; any non-empty line is an ack."""
    ard = _make_arduino()
    ard.ser._ack = b"Blue LED OFF\n"
    assert ard.send_command(4) is True


def test_missing_ack_is_reported_not_silently_timed_out(tmp_path):
    latency_logging.start_session(str(tmp_path), "unit")
    ard = _make_arduino()
    ard.ser._ack = b""                       # device says nothing
    assert ard.send_command(1, frame_t0=time.perf_counter()) is False
    latency_logging.stop_session()
    rows = list(csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8")))
    assert rows[0]["ack_ok"] == "False"


def test_no_module_level_frame_timestamp_is_consulted(tmp_path):
    """D2 regression: the capture time must come from the argument, never from
    module state, so that a concurrently advancing camera cannot poison it."""
    latency_logging.start_session(str(tmp_path), "unit")
    ard = _make_arduino()
    latency_logging.FRAME_T0 = time.perf_counter() + 999  # absurd global
    explicit_t0 = time.perf_counter()
    ard.send_command(1, frame_t0=explicit_t0)
    latency_logging.stop_session()
    row = next(csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8")))
    assert 0 < float(row["frame_to_ack_ms"]) < 1000, (
        "end-to-end latency was computed from the stale global, not the argument"
    )


def test_legs_sum_to_end_to_end(tmp_path):
    latency_logging.start_session(str(tmp_path), "unit")
    ard = _make_arduino()
    t_cap = time.perf_counter()
    time.sleep(0.005)
    t_dec = time.perf_counter()
    ard.send_command(1, frame_t0=t_cap, t_decision=t_dec, frame=20, cam_seq=25,
                     roi=1, edge="enter")
    latency_logging.stop_session()
    r = next(csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8")))
    total = (
        float(r["capture_to_decision_ms"])
        + float(r["decision_to_send_ms"])
        + float(r["serial_act_ms"])
    )
    assert total == pytest.approx(float(r["frame_to_ack_ms"]), abs=0.01)
    assert r["roi"] == "1" and r["edge"] == "enter" and r["frame"] == "20"


def test_frame_ledger_indexes_written_frames_contiguously(tmp_path):
    latency_logging.start_session(str(tmp_path), "unit")
    idx = [
        latency_logging.log_video_frame(
            frame=f, cam_seq=f + 3, t_capture=time.perf_counter()
        )
        for f in range(1, 6)
    ]
    latency_logging.note_drop("video")
    latency_logging.stop_session()
    assert idx == [0, 1, 2, 3, 4]
    ledger = tmp_path / "7_FrameLedger_unit.csv"
    rows = list(csv.DictReader(open(ledger, encoding="utf-8")))
    assert [int(r["video_write_index"]) for r in rows] == [0, 1, 2, 3, 4]
    meta = json.load(open(tmp_path / "8_LatencyMeta_unit.json", encoding="utf-8"))
    assert meta["video_queue_drops"] == 1
    assert meta["n_video_frames_written"] == 5


def test_measured_fps_is_recorded(tmp_path):
    latency_logging.start_session(str(tmp_path), "unit")
    t = time.perf_counter()
    for k in range(40):
        latency_logging.note_capture(t + k / 39.0)      # simulate 39 fps
    latency_logging.stop_session()
    meta = json.load(open(tmp_path / "8_LatencyMeta_unit.json", encoding="utf-8"))
    assert meta["fps_measured"] == pytest.approx(39.0, rel=1e-6)


def test_logging_never_raises_without_a_session():
    """Logging must never take down the closed loop."""
    latency_logging.stop_session()
    latency_logging.log_trigger(1, 0.0, 0.1)
    latency_logging.log_video_frame(1, 1, 0.0)
    latency_logging.note_capture(0.0)
    latency_logging.note_drop("video")


def test_session_files_land_in_the_recording_folder(tmp_path):
    folder = tmp_path / "Grupo_1"
    latency_logging.start_session(str(folder), "Grupo_1")
    latency_logging.stop_session()
    for name in ("6_Latency_Grupo_1.csv", "7_FrameLedger_Grupo_1.csv",
                 "8_LatencyMeta_Grupo_1.json"):
        assert os.path.exists(folder / name)


# --- Review follow-ups (PR #21) -------------------------------------------


def test_zero_is_a_legal_timestamp_not_a_missing_one(tmp_path):
    """perf_counter's epoch is arbitrary, so 0.0 is a reading, not an absence.

    Truthiness checks blanked the latency columns for it, which corrupts the
    measurement silently -- the failure mode this whole module exists to stop.
    """
    latency_logging.start_session(str(tmp_path), "unit")
    latency_logging.log_trigger(
        1, 0.0, 0.5, 0.0, ack_ok=True, ack_text="Red LED 1 ON", t_decision=0.0
    )
    latency_logging.stop_session()
    row = next(csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8")))
    assert row["t_capture_perf"] == "0.000000"
    assert row["t_decision_perf"] == "0.000000"
    assert row["t_send_perf"] == "0.000000"
    assert float(row["frame_to_ack_ms"]) == pytest.approx(500.0)
    assert float(row["capture_to_decision_ms"]) == pytest.approx(0.0)


def test_a_trigger_with_no_ack_still_produces_a_row(tmp_path):
    """A write timeout is a data point: the stimulus may never have fired."""
    latency_logging.start_session(str(tmp_path), "unit")
    latency_logging.log_trigger(
        7, 1.0, None, 0.5, ack_ok=False, ack_text="WRITE_TIMEOUT", roi=4, edge="enter"
    )
    latency_logging.stop_session()
    row = next(csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8")))
    assert row["ack_text"] == "WRITE_TIMEOUT"
    assert row["ack_ok"] == "False"
    assert row["t_ack_perf"] == ""
    assert row["serial_act_ms"] == ""
    assert row["frame_to_ack_ms"] == ""
    # It still counts as an attempted trigger.
    meta = json.load(open(tmp_path / "8_LatencyMeta_unit.json", encoding="utf-8"))
    assert meta["n_triggers"] == 1


def test_stop_session_does_not_race_concurrent_logging(tmp_path):
    """stop_session must quiesce logging, not close files under a live writer.

    Reading _SESSION outside the lock let a call that had already taken the
    handle write to a closed file after teardown.
    """
    import threading

    latency_logging.start_session(str(tmp_path), "unit")
    stop = threading.Event()
    errors = []

    def hammer():
        while not stop.is_set():
            try:
                latency_logging.note_capture(time.perf_counter())
                latency_logging.log_video_frame(1, 1, time.perf_counter())
                latency_logging.log_trigger(1, 0.0, 0.1, 0.0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                return

    workers = [threading.Thread(target=hammer) for _ in range(4)]
    for w in workers:
        w.start()
    time.sleep(0.2)
    latency_logging.stop_session()
    stop.set()
    for w in workers:
        w.join(timeout=2)

    assert not errors
    # The sidecar must exist and its counters must be internally consistent
    # with the rows that were actually written.
    meta = json.load(open(tmp_path / "8_LatencyMeta_unit.json", encoding="utf-8"))
    ledger = list(
        csv.DictReader(open(tmp_path / "7_FrameLedger_unit.csv", encoding="utf-8"))
    )
    triggers = list(
        csv.DictReader(open(tmp_path / "6_Latency_unit.csv", encoding="utf-8"))
    )
    assert meta["n_video_frames_written"] == len(ledger)
    assert meta["n_triggers"] == len(triggers)
