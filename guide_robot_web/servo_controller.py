import logging
import os
import socket
import numpy as np
import requests
from latency_tracker import TimerContext

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
    with TimerContext("servo_send_angle"):
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

def play_speech(audio: np.ndarray, sample_rate: int) -> bool:
    with TimerContext("play_speech_total"):
        url = _discover()
        if url is None:
            return False

        window_size = int(sample_rate * ENVELOPE_WINDOW_MS / 1000)
        if window_size < 1:
            window_size = 1

        num_windows = len(audio) // window_size
        if num_windows < 2:
            return False

        with TimerContext("compute_audio_envelope"):
            audio_trimmed = audio[:num_windows * window_size]
            windows = audio_trimmed.reshape(num_windows, window_size)
            rms = np.sqrt(np.mean(windows ** 2, axis=1))

            max_rms = rms.max()
            if max_rms > 1e-10:
                rms = rms / max_rms

            if SMOOTH_WINDOW > 1 and len(rms) > SMOOTH_WINDOW:
                kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
                rms = np.convolve(rms, kernel, mode="same")

            angles = MOUTH_CLOSED_ANGLE + rms * (MOUTH_OPEN_ANGLE - MOUTH_CLOSED_ANGLE)
            angles = np.clip(angles, 0, 180).astype(int)

            if len(angles) > MAX_ENVELOPE_FRAMES:
                idx = np.linspace(0, len(angles) - 1, MAX_ENVELOPE_FRAMES).astype(int)
                angles = angles[idx]
                interval_ms = max(
                    ENVELOPE_WINDOW_MS,
                    int(round(len(audio) * 1000 / sample_rate / MAX_ENVELOPE_FRAMES)),
                )
            else:
                interval_ms = ENVELOPE_WINDOW_MS

        with TimerContext("servo_envelope_http"):
            try:
                r = requests.post(
                    f"{url}/envelope",
                    json={
                        "angles": angles.tolist(),
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
    with TimerContext("servo_close_mouth"):
        return _send_angle(MOUTH_CLOSED_ANGLE)