"""Human interrupt singleton + listener for real-world TD3 collection.

Two-move operator interrupt that piggybacks on the existing stop pipeline:

  Move 1 — STOP   (`s` keystroke OR `touch /tmp/airhockey_human_interrupt`):
                  sets `human_interrupt_state.active=True`. The next call to
                  `_classify_stop_event` returns `active=True`, which causes
                  PolicyRunner to truncate the current episode and the
                  orchestrator to route to the hard-reset path.

  Move 2 — RESET  (`r` keystroke OR delete the flag file):
                  clears the singleton. The reset runner's existing
                  `while stop_state.active: sleep(0.25)` wait unblocks and
                  the FSM proceeds.

Always-on. The stdin listener no-ops gracefully when stdin is not a tty
(e.g., headless / piped). The file-flag path is the fallback for tmux/ssh
sessions where keystrokes don't reach the launching terminal.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from airhockey.sims.real.multiprocessing import NonBlockingConsole


HUMAN_INTERRUPT_FLAG_PATH = "/tmp/airhockey_human_interrupt"

_KEY_STOP = "s"
_KEY_RESET = "r"


@dataclass
class _HumanInterruptState:
    """Singleton state. Thread-safe via a single lock."""

    _lock: threading.Lock
    _active: bool = False
    _reason: str = "human_interrupt"

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def set_stop(self, reason: str = "human_interrupt") -> bool:
        """Activate. Returns True if state transitioned from inactive→active."""
        with self._lock:
            transitioned = not self._active
            self._active = True
            self._reason = str(reason)
        return transitioned

    def clear(self) -> bool:
        """Deactivate. Returns True if state transitioned from active→inactive."""
        with self._lock:
            transitioned = self._active
            self._active = False
        return transitioned


human_interrupt_state = _HumanInterruptState(_lock=threading.Lock())


class HumanInterruptListener:
    """Background thread polling stdin + the file-flag.

    Use as `listener = HumanInterruptListener(); listener.start()` at
    orchestrator init, with `listener.stop()` in a `finally` at teardown.
    """

    def __init__(
        self,
        *,
        flag_path: str = HUMAN_INTERRUPT_FLAG_PATH,
        poll_interval_s: float = 0.05,
    ) -> None:
        self._flag_path = str(flag_path)
        self._poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # Clear any stale flag file from a previous run so we don't trip
        # on startup. Best-effort — a missing file is fine.
        try:
            if os.path.exists(self._flag_path):
                os.remove(self._flag_path)
        except OSError:
            pass
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="HumanInterruptListener",
            daemon=True,
        )
        self._thread.start()
        print(
            "[human_interrupt] listener started. "
            f"keys: '{_KEY_STOP}'=STOP  '{_KEY_RESET}'=RESET   "
            f"file flag: touch/rm {self._flag_path}"
        )

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    # ------------------------------------------------------------------

    def _handle_key(self, ch: str) -> None:
        if ch == _KEY_STOP:
            if human_interrupt_state.set_stop("human_interrupt_keystroke"):
                print(
                    "[human_interrupt] STOP pressed — episode will truncate; "
                    f"press '{_KEY_RESET}' (or rm {self._flag_path}) when ready to reset."
                )
        elif ch == _KEY_RESET:
            if human_interrupt_state.clear():
                print("[human_interrupt] RESET pressed — resuming reset/training.")

    def _check_flag_file(self) -> None:
        exists = os.path.exists(self._flag_path)
        if exists and not human_interrupt_state.is_active():
            if human_interrupt_state.set_stop("human_interrupt_flag_file"):
                print(
                    "[human_interrupt] STOP via flag file — episode will truncate; "
                    f"rm {self._flag_path} (or press '{_KEY_RESET}') when ready to reset."
                )
        elif (not exists) and human_interrupt_state.is_active() and (
            human_interrupt_state.reason() == "human_interrupt_flag_file"
        ):
            # File-set state cleared by removing the file; clear singleton.
            if human_interrupt_state.clear():
                print("[human_interrupt] RESET via flag-file removal — resuming.")

    def _run(self) -> None:
        # `NonBlockingConsole` requires a tty; on a non-tty stdin
        # (`old_settings is None`) `get_data()` returns False and we just
        # rely on the file-flag path.
        try:
            with NonBlockingConsole() as nbc:
                while not self._stop_event.is_set():
                    try:
                        ch = nbc.get_data()
                    except Exception:
                        ch = False
                    if isinstance(ch, str) and ch:
                        self._handle_key(ch)
                    self._check_flag_file()
                    time.sleep(self._poll_interval_s)
        except Exception as exc:
            # Listener failures must never take down training. Fall back
            # to file-flag-only polling so the operator still has a way in.
            print(f"[human_interrupt] stdin listener disabled ({exc}); file-flag still active.")
            while not self._stop_event.is_set():
                self._check_flag_file()
                time.sleep(self._poll_interval_s)
