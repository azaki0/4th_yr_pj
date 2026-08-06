import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

_request_context = threading.local()
_latency_logs = deque(maxlen=100)
_latency_lock = threading.Lock()


@dataclass
class StageTiming:
    name: str
    start_time: float
    end_time: float = 0.0
    children: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


@dataclass
class RequestLatency:
    request_id: str
    prompt: str
    language: str
    root: StageTiming
    total_ms: float = 0.0
    ttfr_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt[:100] + ("..." if len(self.prompt) > 100 else ""),
            "language": self.language,
            "total_ms": round(self.total_ms, 2),
            "ttfr_ms": round(self.ttfr_ms, 2),
            "timestamp": self.timestamp,
            "stages": self.root.to_dict(),
        }


class TimerContext:
    def __init__(self, name: str, metadata: Optional[dict] = None):
        self.name = name
        self.metadata = metadata or {}
        self.stage: Optional[StageTiming] = None

    def __enter__(self) -> "TimerContext":
        if not hasattr(_request_context, "stack"):
            _request_context.stack = []
            _request_context.root = StageTiming(name="root", start_time=time.perf_counter())
            _request_context.request_id = str(uuid.uuid4())[:8]
            if not hasattr(_request_context, "prompt"):
                _request_context.prompt = ""
            if not hasattr(_request_context, "language"):
                _request_context.language = ""

        parent = _request_context.stack[-1] if _request_context.stack else _request_context.root
        self.stage = StageTiming(name=self.name, start_time=time.perf_counter(), metadata=self.metadata)
        parent.children.append(self.stage)
        _request_context.stack.append(self.stage)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stage:
            self.stage.end_time = time.perf_counter()
        if _request_context.stack:
            _request_context.stack.pop()


def set_request_info(prompt: str, language: str):
    if not hasattr(_request_context, "prompt"):
        _request_context.prompt = ""
        _request_context.language = ""
    _request_context.prompt = prompt
    _request_context.language = language


def record_first_response():
    if hasattr(_request_context, "first_response_time"):
        return
    _request_context.first_response_time = time.perf_counter()


def finalize_request() -> Optional[RequestLatency]:
    if not hasattr(_request_context, "root"):
        return None

    _request_context.root.end_time = time.perf_counter()

    ttfr_ms = 0.0
    if hasattr(_request_context, "first_response_time"):
        ttfr_ms = (_request_context.first_response_time - _request_context.root.start_time) * 1000

    latency = RequestLatency(
        request_id=_request_context.request_id,
        prompt=_request_context.prompt,
        language=_request_context.language,
        root=_request_context.root,
        total_ms=_request_context.root.duration_ms,
        ttfr_ms=ttfr_ms,
    )

    with _latency_lock:
        _latency_logs.append(latency)

    del _request_context.root
    del _request_context.stack
    del _request_context.request_id
    del _request_context.prompt
    del _request_context.language
    if hasattr(_request_context, "first_response_time"):
        del _request_context.first_response_time

    return latency


def get_latency_logs(limit: int = 50) -> list[dict]:
    with _latency_lock:
        return [log.to_dict() for log in list(_latency_logs)[-limit:]]


def clear_latency_logs():
    with _latency_lock:
        _latency_logs.clear()


@contextmanager
def measure(name: str, metadata: Optional[dict] = None):
    with TimerContext(name, metadata):
        yield