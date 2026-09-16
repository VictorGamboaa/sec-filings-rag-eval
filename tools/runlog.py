"""Structured run logging -- Rule 2: instrumentation is not optional.

Every stage emits timing to a JSONL run log. One shared recorder exists so that
logs stay comparable across runs and across stages; if each stage invented its
own schema, the harness could not measure scaling behavior or throughput, which
is its stated purpose.

Record types, one JSON object per line:

  run_start    run id, resolved config, environment
  stage_start  stage name, start timestamp
  error        one line per failure -- Rule 2 requires errors be logged
               individually, not merely tallied
  stage_end    wall clock, item count, bytes in/out, request and token counts,
               error count
  run_end      total wall clock, per-stage summary

``stage_end`` is written even when the stage body raises, so a crashed run is
still measurable.

Usage::

    with RunLog(config) as log:
        with log.stage("fetch") as st:
            st.request()
            st.bytes_in(len(body))
            st.count()

CLI self-test::

    python -m tools.runlog --selftest
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

__all__ = ["RunLog", "StageRecorder", "new_run_id"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_run_id() -> str:
    """Timestamp-derived run id, sortable and filesystem-safe."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class StageRecorder:
    """Accumulates counters for one stage. Thread-safe.

    Counters are additive and start at zero. A stage that performs no network
    work simply never calls ``request()`` or ``tokens()``, and those totals stay
    zero -- which is meaningful rather than missing: it is what makes a
    local-vs-hosted cost comparison readable later.
    """

    name: str
    _log: RunLog
    items: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def count(self, n: int = 1) -> None:
        """Record n items processed."""
        with self._lock:
            self.items += n

    def bytes_in(self, n: int) -> None:
        """Record bytes read or downloaded."""
        with self._lock:
            self.bytes_read += n

    def bytes_out(self, n: int) -> None:
        """Record bytes persisted."""
        with self._lock:
            self.bytes_written += n

    def request(self, n: int = 1) -> None:
        """Record n network requests. Network stages only."""
        with self._lock:
            self.requests += n

    def tokens(self, prompt: int = 0, completion: int = 0) -> None:
        """Record token usage, for stages billed by token."""
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    def note(self, **kwargs: Any) -> None:
        """Attach stage-specific scalars to the stage_end record."""
        with self._lock:
            self.extra.update(kwargs)

    def error(self, exc: BaseException | str, context: Any = None) -> None:
        """Log one failure individually and increment the error count.

        Does not raise. Callers decide whether a failure is fatal; this only
        records it. Rule 2 requires both the individual record and the tally.
        """
        with self._lock:
            self.errors += 1
            ordinal = self.errors
        if isinstance(exc, BaseException):
            etype = type(exc).__name__
            emsg = str(exc)
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        else:
            etype, emsg, detail = "Error", str(exc), None
        self._log._write(
            {
                "event": "error",
                "stage": self.name,
                "error_ordinal": ordinal,
                "error_type": etype,
                "message": emsg,
                "detail": detail,
                "context": context,
            }
        )

    def _summary(self) -> dict[str, Any]:
        with self._lock:
            wall = time.perf_counter() - self._t0
            rate = round(self.items / wall, 3) if wall > 0 else None
            return {
                "stage": self.name,
                "wall_clock_sec": round(wall, 6),
                "items": self.items,
                "items_per_sec": rate,
                "bytes_in": self.bytes_read,
                "bytes_out": self.bytes_written,
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "errors": self.errors,
                **self.extra,
            }


class RunLog:
    """Writes a JSONL run log. One file per run.

    Accepts either a Config (reading ``run.id`` and ``run.log_dir``) or an
    explicit ``run_id`` / ``log_dir`` pair.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        run_id: str | None = None,
        log_dir: str | os.PathLike[str] | None = None,
        resolved_config: dict[str, Any] | None = None,
    ) -> None:
        if config is not None:
            run_id = run_id or config.get("run.id") or new_run_id()
            log_dir = log_dir or config.get("run.log_dir", "outputs/runs")
            if resolved_config is None:
                resolved_config = config.as_dict()
        self.run_id = run_id or new_run_id()
        self.log_dir = Path(log_dir or "outputs/runs")
        self.path = self.log_dir / f"{self.run_id}.jsonl"
        self._resolved_config = resolved_config
        self._fh: Any = None
        self._lock = threading.Lock()
        self._stages: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def __enter__(self) -> RunLog:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._write(
            {
                "event": "run_start",
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "config": self._resolved_config,
            }
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._write(
            {
                "event": "run_end",
                "wall_clock_sec": round(time.perf_counter() - self._t0, 6),
                "stages": self._stages,
                "total_errors": sum(s.get("errors", 0) for s in self._stages),
                "failed": exc_type is not None,
                "failure": None if exc is None else f"{exc_type.__name__}: {exc}",
            }
        )
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return False  # never suppress the exception

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecorder]:
        """Time one stage, emitting stage_start and stage_end.

        stage_end is emitted even if the body raises, so a crashed run still
        yields a measurement. The exception then propagates unchanged.
        """
        rec = StageRecorder(name=name, _log=self)
        self._write({"event": "stage_start", "stage": name})
        aborted = False
        try:
            yield rec
        except BaseException as exc:
            aborted = True
            rec.error(exc, context="stage aborted")
            raise
        finally:
            summary = rec._summary()
            summary["aborted"] = aborted
            self._stages.append(summary)
            self._write({"event": "stage_end", **summary})

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(
            {"ts": _utc_now(), "run_id": self.run_id, **record},
            ensure_ascii=False,
            default=str,
        )
        with self._lock:
            if self._fh is None:
                raise RuntimeError("RunLog used outside its context manager")
            self._fh.write(line + "\n")
            self._fh.flush()


def _selftest() -> int:
    """Synthetic two-stage run with one injected error, printed back."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        with RunLog(run_id="selftest", log_dir=td) as log:
            with log.stage("fetch") as st:
                for _ in range(3):
                    st.request()
                    st.bytes_in(1024)
                    st.count()
                st.error(
                    ValueError("simulated 429 from source"),
                    context={"cik": "0000320193"},
                )
            with log.stage("embed") as st:
                st.count(3)
                st.bytes_out(3 * 384 * 4)
                st.note(model="selftest-stub", dim=384)
        text = Path(td, "selftest.jsonl").read_text(encoding="utf-8")

    print(text.rstrip())
    records = [json.loads(line) for line in text.splitlines()]
    events = [r["event"] for r in records]
    fetch_end = next(
        r for r in records if r["event"] == "stage_end" and r["stage"] == "fetch"
    )
    embed_end = next(
        r for r in records if r["event"] == "stage_end" and r["stage"] == "embed"
    )
    run_end = records[-1]

    checks = [
        ("run_start emitted first", events[0] == "run_start"),
        ("run_end emitted last", events[-1] == "run_end"),
        ("two stages recorded", events.count("stage_end") == 2),
        ("error logged individually", events.count("error") == 1),
        ("error counted in stage_end", fetch_end["errors"] == 1),
        ("items counted", fetch_end["items"] == 3),
        ("bytes_in counted", fetch_end["bytes_in"] == 3072),
        ("requests counted", fetch_end["requests"] == 3),
        ("wall clock present", fetch_end["wall_clock_sec"] >= 0),
        ("throughput derived", fetch_end["items_per_sec"] is not None),
        ("non-network stage reports zero requests", embed_end["requests"] == 0),
        ("note() reached stage_end", embed_end.get("dim") == 384),
        ("run totals aggregated", run_end["total_errors"] == 1),
        ("stage not marked aborted", fetch_end["aborted"] is False),
    ]

    print("\n--- checks ---")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
    raise SystemExit("No action taken. Pass --selftest.")
