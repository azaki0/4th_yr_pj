import logging
import os

logger = logging.getLogger(__name__)

HAND_PORT = os.getenv("HAND_SERVO_PORT", "COM5")
HAND_BAUD = 9600

_serial = None

def _get_serial():
    global _serial

    if _serial is not None:
        return _serial

    try:
        import serial

        _serial = serial.Serial(HAND_PORT, HAND_BAUD, timeout=0.1)
        logger.info("servo connected on %s", HAND_PORT)
        return _serial
    except Exception as exc:
        logger.warning("servo serial unavailable on %s: %s", HAND_PORT, exc)
        return None

def _send(command):
    ser = _get_serial()
    if ser is None:
        return False
    try:
        ser.write(command)
        return True
    except Exception as e:
        logger.warning("servo serial failed: %s", e)
        try:
            ser.close()
        except Exception:
            pass
        globals()["_serial"] = None
        return False

def start_hand():
    print("controller working.start")
    return _send(b'1')

def stop_hand():
    print("controller working.stop")
    return _send(b'0')
