import logging
import os
import socket
import requests

logger = logging.getLogger(__name__)

MOUTH_CLOSED_ANGLE = 90
MOUTH_OPEN_ANGLE = 135
ENVELOPE_WINDOW_MS = 50
SMOOTH_WINDOW = 3
MAX_ENVELOPE_FRAMES = 180
MDNS_HOSTNAME = "robot-mouth.local"
DEFAULT_PORT = 80

_esp32_url = None

def _discover():
    global _esp32_url

    if _esp32_url is not None:
        return _esp32_url

    try:
        ip = socket.gethostbyname(MDNS_HOSTNAME)
        _esp32_url = f"http://{ip}:{DEFAULT_PORT}"
        logger.info("Servo ESP32 found via mDNS at %s", _esp32_url)
        return _esp32_url
    except OSError:
        logger.debug("mDNS resolution failed for %s", MDNS_HOSTNAME)

    static_ip = os.getenv("SERVO_ESP32_IP")
    if static_ip:
        _esp32_url = f"http://{static_ip}:{DEFAULT_PORT}"
        logger.info("Servo ESP32 using static IP %s", _esp32_url)
        return _esp32_url

    logger.warning("Servo ESP32 not found")
    return None

def _send_angle(angle):
    url = _discover()
    if url is None:
        return False
    try:
        r = requests.get(f"{url}/servo?angle={int(angle)}", timeout=1)
        return r.ok
    except requests.RequestException:
        logger.debug("Servo HTTP failed")
        return False

def set_angle(angle):
    return _send_angle(angle)

def play_speech(audio, sample_rate: int) -> bool:
    url = _discover()
    if url is None:
        return False

    window_size = int(sample_rate * ENVELOPE_WINDOW_MS / 1000)
    if window_size < 1:
        window_size = 1

    num_windows = len(audio) // window_size
    if num_windows < 2:
        return False

    rms = []
    for index in range(num_windows):
        start = index * window_size
        window = audio[start:start + window_size]
        if not window:
            rms.append(0.0)
            continue
        rms.append((sum(sample * sample for sample in window) / len(window)) ** 0.5)

    max_rms = max(rms) if rms else 0
    if max_rms > 1e-10:
        rms = [value / max_rms for value in rms]

    if SMOOTH_WINDOW > 1 and len(rms) > SMOOTH_WINDOW:
        smoothed = []
        half_window = SMOOTH_WINDOW // 2
        for index in range(len(rms)):
            start = max(0, index - half_window)
            end = min(len(rms), index + half_window + 1)
            smoothed.append(sum(rms[start:end]) / (end - start))
        rms = smoothed

    angles = [
        int(max(0, min(180, MOUTH_CLOSED_ANGLE + value * (MOUTH_OPEN_ANGLE - MOUTH_CLOSED_ANGLE))))
        for value in rms
    ]

    if len(angles) > MAX_ENVELOPE_FRAMES:
        step = (len(angles) - 1) / (MAX_ENVELOPE_FRAMES - 1)
        angles = [angles[round(index * step)] for index in range(MAX_ENVELOPE_FRAMES)]
        interval_ms = max(
            ENVELOPE_WINDOW_MS,
            int(round(len(audio) * 1000 / sample_rate / MAX_ENVELOPE_FRAMES)),
        )
    else:
        interval_ms = ENVELOPE_WINDOW_MS

    try:
        r = requests.post(
            f"{url}/envelope",
            json={
                "angles": angles,
                "interval_ms": interval_ms,
                "start_delay_ms": 0,
            },
            timeout=2,
        )
        return r.ok
    except requests.RequestException:
        logger.debug("Failed to send envelope to servo")
        return False

def close_mouth():
    return _send_angle(MOUTH_CLOSED_ANGLE)
