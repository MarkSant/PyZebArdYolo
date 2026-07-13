"""latency_logging.py — instrumentacao de latencia do loop fechado (PyZebArdYolo).
Coloque este arquivo em src/zebtrack/ . Nao tem dependencias alem da stdlib.

Mede dois tempos por gatilho (entrada/saida de ROI -> comando ao Arduino):
  - serial_act_ms : t_ack - t_send  (transmissao serial + atuacao do Arduino/LED)
  - frame_to_ack_ms : t_ack - FRAME_T0 (captura do frame -> LED aceso = ponta-a-ponta de software)

FRAME_T0 e atualizado pela alca de captura (uma linha; ver PASSO_A_PASSO.md).
Saida: CSV definido em PYZEB_LATENCY_CSV (default ./latency_log.csv).
"""
import time, csv, os, threading

FRAME_T0 = None                      # setado na alca de captura: time.perf_counter()
_PATH = os.environ.get("PYZEB_LATENCY_CSV", "latency_log.csv")
_LOCK = threading.Lock()
_f = None
_w = None


def _ensure():
    global _f, _w
    if _w is None:
        new = (not os.path.exists(_PATH)) or os.path.getsize(_PATH) == 0
        _f = open(_PATH, "a", newline="", encoding="utf-8")
        _w = csv.writer(_f)
        if new:
            _w.writerow(["wall_iso", "cmd", "t_send_perf", "t_ack_perf",
                         "serial_act_ms", "frame_to_ack_ms"])


def log_trigger(cmd, t_send, t_ack, frame_t0=None):
    """Grava uma linha por gatilho. Chamado de dentro de Arduino.send_command()."""
    try:
        with _LOCK:
            _ensure()
            serial_ms = (t_ack - t_send) * 1000.0
            f2a = ((t_ack - frame_t0) * 1000.0) if frame_t0 else ""
            _w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), cmd,
                         f"{t_send:.6f}", f"{t_ack:.6f}", f"{serial_ms:.3f}",
                         (f"{f2a:.3f}" if f2a != "" else "")])
            _f.flush()
    except Exception:
        pass   # nunca derrubar o loop por causa de logging
