import json
import logging
import queue
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

WORKER_PATH = Path(__file__).resolve().with_name("voice_worker.py")

_worker = None
_worker_lock = threading.Lock()


def _start_worker():
    global _worker

    if _worker is not None and _worker.poll() is None:
        return _worker

    _worker = subprocess.Popen(
        [sys.executable, str(WORKER_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    return _worker

def _stop_worker():
    global _worker

    if _worker is not None and _worker.poll() is None:
        _worker.kill()
    _worker = None

def _readline_with_timeout(worker, timeout):
    lines = queue.Queue(maxsize=1)

    def read_line():
        try:
            if worker.stdout is None:
                lines.put(None)
            else:
                lines.put(worker.stdout.readline())
        except Exception:
            lines.put(None)

    threading.Thread(target=read_line, daemon=True).start()

    try:
        return lines.get(timeout=timeout)
    except queue.Empty:
        _stop_worker()
        raise TimeoutError("voice worker timed out")

def _worker_request(payload, timeout=None):
    with _worker_lock:
        worker = _start_worker()
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("voice worker pipes are unavailable")

        worker.stdin.write(json.dumps(payload) + "\n")
        worker.stdin.flush()

        read_timeout = timeout if timeout is not None else 60
        while True:
            line = _readline_with_timeout(worker, read_timeout)
            if not line:
                _stop_worker()
                raise RuntimeError("voice worker stopped")

            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON voice worker output: %s", line.strip())


def preload():
    result = _worker_request({"action": "preload"})
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "voice preload failed"))
    return result


def capture(timeout=30):
    try:
        result = _worker_request({"action": "capture", "timeout": timeout}, timeout=timeout + 5)
        if not result.get("ok"):
            logger.warning("Voice capture error: %s", result.get("error"))
            return None

        text = result.get("text")
        if text:
            logger.info("Captured: %s", text[:80])
        return text
    except Exception as exc:
        logger.warning("Voice capture error: %s", exc)
        return None
