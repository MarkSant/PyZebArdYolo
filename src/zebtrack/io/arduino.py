import queue
import threading
import time
from types import TracebackType
from typing import Optional, Type

import serial
import structlog

from zebtrack import latency_logging
from zebtrack.settings import settings

log = structlog.get_logger()


class Arduino:
    """
    Manages serial communication with an Arduino device.
    """

    def __init__(self, port: str, baud_rate: int):
        """
        Initializes the Arduino controller.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.ser: Optional[serial.Serial] = None
        # Trigger dispatch runs on its own thread (see send_command_async).
        self._tx_queue: "queue.Queue" = queue.Queue(maxsize=8)
        self._tx_stop = threading.Event()
        self._tx_thread: Optional[threading.Thread] = None
        log.info("arduino.init", port=self.port, baud_rate=self.baud_rate)

    def connect(self) -> bool:
        """
        Attempts to establish a serial connection with the Arduino.
        """
        if self.ser and self.ser.is_open:
            log.info("arduino.connect.already_connected")
            return True
        try:
            # 250 ms is ~17x the 14.6 ms an ACK line takes at 9600 baud, so a
            # healthy reply always arrives, while a lost one costs 0.25 s
            # instead of stalling the closed loop for 2 s.
            # write_timeout is NOT optional here. pyserial defaults it to None,
            # which on Windows means WriteFile + GetOverlappedResult(bWait=True)
            # -- a wait with no bound. A single stalled write then blocks the
            # calling thread until the port is closed, which is exactly what
            # froze the live preview for 60 s on 2026-08-08.
            self.ser = serial.Serial(
                self.port, self.baud_rate, timeout=0.25, write_timeout=0.25
            )
            # Opening the port auto-resets the Arduino Uno (DTR toggle). Wait
            # for the board to finish booting before treating it as available.
            time.sleep(2)
            # Some sketches print a banner on boot; capture it if present, but
            # do not require it (the LED-controller firmware sends nothing).
            banner = ""
            if self.ser.in_waiting:
                banner = self.ser.readline().decode("utf-8", errors="replace").strip()
            self.ser.reset_input_buffer()
            log.info("arduino.connect.success", port=self.port, banner=banner)
            return True
        except (serial.SerialException, OSError) as e:
            log.warning(
                "arduino.connect.failed", port=self.port, exc_info=e
            )
            self.ser = None
            return False

    def __enter__(self) -> "Arduino":
        """Enter the runtime context related to this object."""
        if not self.connect():
            raise RuntimeError(f"Failed to connect to Arduino on port {self.port}")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit the runtime context and close the connection."""
        self.close()

    # ------------------------------------------------------------------
    # Asynchronous dispatch
    # ------------------------------------------------------------------
    # The closed-loop analysis thread must never touch the serial port. Even
    # with every timeout set, a USB-CDC device can stall in the driver, and a
    # stall on the analysis thread stops detection, the preview and recording
    # all at once. Triggers are queued here and written by a dedicated thread;
    # a stalled port now costs dropped triggers (counted, logged) instead of a
    # frozen application. The latency columns are unaffected: t_send and t_ack
    # are still taken around the real write, and the queue hop shows up in
    # decision_to_send_ms, where it is visible rather than hidden.

    def _ensure_dispatcher(self) -> None:
        if self._tx_thread is not None and self._tx_thread.is_alive():
            return
        self._tx_stop.clear()
        self._tx_thread = threading.Thread(
            target=self._tx_loop, name="ArduinoTxThread", daemon=True
        )
        self._tx_thread.start()

    def _tx_loop(self) -> None:
        while not self._tx_stop.is_set():
            try:
                box_number, kwargs = self._tx_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.send_command(box_number, **kwargs)
            except Exception:  # noqa: BLE001 - a dead tx thread is silent
                log.exception("arduino.tx_thread.error", command=box_number)
        log.info("arduino.tx_thread.finished")

    def send_command_async(self, box_number: int, **kwargs) -> bool:
        """Queue a trigger for the dispatcher thread. Never blocks the caller.

        Returns True if the trigger was queued, False if it was dropped because
        the port is not keeping up.
        """
        self._ensure_dispatcher()
        try:
            self._tx_queue.put_nowait((box_number, kwargs))
            return True
        except queue.Full:
            # Counted in 8_LatencyMeta as trigger_queue_drops. A stimulus the
            # port was too slow to accept must not vanish from the record.
            log.error("arduino.command.dropped_queue_full", command=box_number)
            latency_logging.note_drop("trigger")
            return False

    def stop_dispatcher(self, timeout: float = 1.0) -> None:
        """Stops the dispatcher thread, without waiting on a wedged write."""
        self._tx_stop.set()
        thread = self._tx_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.error("arduino.tx_thread.join_timeout", timeout_s=timeout)
        self._tx_thread = None

    def send_command(
        self,
        box_number: int,
        *,
        frame_t0: Optional[float] = None,
        t_decision: Optional[float] = None,
        frame: Optional[int] = None,
        cam_seq: Optional[int] = None,
        roi: Optional[int] = None,
        edge: Optional[str] = None,
    ) -> bool:
        """
        Sends a command to the Arduino and waits for an acknowledgment.

        The keyword arguments carry the identity and timing of the frame that
        produced this decision. They are optional so that existing callers and
        the module self-test keep working, but the live loop must pass them:
        without ``frame_t0`` the end-to-end latency column cannot be computed,
        and reading a stale global instead (as this method used to) silently
        measures the wrong frame.
        """
        try:
            command_num = int(box_number)
        except (ValueError, TypeError):
            log.error("arduino.command.invalid", command=box_number)
            return False

        if self.ser and self.ser.is_open:
            command = f"{command_num}\n"
            # Bound before the try: the timeout handler below reports it, and
            # the drain could in principle raise before it is assigned.
            t_send = None
            try:
                # Drain BEFORE the write. Draining after the read (as this code
                # previously did) means the next readline returns the ACK of the
                # PREVIOUS command, already buffered — an off-by-one that made
                # 97.7% of the logged serial round trips sub-millisecond and
                # therefore physically impossible at 9600 baud.
                self.ser.reset_input_buffer()
                t_send = time.perf_counter()
                self.ser.write(command.encode("utf-8"))
                log.info("arduino.command.sent", command=command_num)
            except serial.SerialTimeoutException:
                # The write did not complete within write_timeout. Report the
                # trigger as lost rather than waiting on the driver -- but still
                # record the row. A trigger the firmware may never have acted on
                # has to stay visible in 6_Latency_<base>.csv and in n_triggers,
                # otherwise the lost stimulus is indistinguishable from one that
                # was never attempted.
                log.error("arduino.command.write_timeout", command=command_num)
                latency_logging.log_trigger(
                    command_num,
                    t_send,
                    None,
                    frame_t0,
                    ack_ok=False,
                    ack_text="WRITE_TIMEOUT",
                    frame=frame,
                    cam_seq=cam_seq,
                    roi=roi,
                    edge=edge,
                    t_decision=t_decision,
                )
                return False
            except serial.SerialException as e:
                log.error("arduino.command.send_error", exc_info=e)
                return False

            try:
                response = self.ser.readline().decode("utf-8", errors="replace").strip()
                t_ack = time.perf_counter()
                ack_ok = bool(response)
                latency_logging.log_trigger(
                    command_num,
                    t_send,
                    t_ack,
                    frame_t0,
                    ack_ok=ack_ok,
                    ack_text=response,
                    frame=frame,
                    cam_seq=cam_seq,
                    roi=roi,
                    edge=edge,
                    t_decision=t_decision,
                )
                # The LED firmware replies with a human-readable line such as
                # "Red LED 1 ON" — never the literal "OK" this branch used to
                # require, so every command was previously logged as a nack and
                # returned False even though the LED did fire. Any non-empty
                # reply is an acknowledgment.
                if ack_ok:
                    log.info(
                        "arduino.command.ack", command=command_num, response=response
                    )
                    return True
                log.warning("arduino.command.no_response", command=command_num)
                return False
            except serial.SerialException as e:
                log.error("arduino.command.send_error", exc_info=e)
                return False
        else:
            log.debug("arduino.command.offline", command=command_num)
            return False

    def close(self) -> None:
        """
        Closes the serial connection.
        """
        self.stop_dispatcher()
        if self.ser and self.ser.is_open:
            self.ser.close()
            log.info("arduino.connection.closed")
        self.ser = None


def main():
    """Main function to run a test of the Arduino module."""
    # This is a test function, using print is fine here.
    print("Testing Arduino communication...")

    if not settings:
        print("Settings could not be loaded. Aborting test.")
        return

    try:
        with Arduino(
            port=settings.arduino.port, baud_rate=settings.arduino.baud_rate
        ) as arduino:
            print(f"Successfully connected to Arduino on {arduino.port}.")

            print("\nSending test commands (1 to 8)...")
            all_commands_successful = True
            for i in range(1, 9):
                print(f"Sending command: {i}...")
                if arduino.send_command(i):
                    print(f"Command {i} sent and acknowledged.")
                else:
                    print(f"Command {i} FAILED.")
                    all_commands_successful = False
                time.sleep(0.5)

            if all_commands_successful:
                print("\nAll test commands sent successfully.")
            else:
                print("\nSome test commands failed.")

    except (RuntimeError, serial.SerialException, OSError) as e:
        print(f"\nERROR: {e}")
        print("Running in offline mode. No commands will be sent.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        print("\nClosing connection (if open).")

    print("\nTest script finished.")


if __name__ == "__main__":
    main()
