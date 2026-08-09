import json
import threading
from queue import Queue

_display_clients = []
_clients_lock = threading.Lock()

OUTPUT_MODE = "laptop"

def set_output_mode(mode):
    global OUTPUT_MODE
    OUTPUT_MODE = mode

def get_output_mode():
    return OUTPUT_MODE

def subscribe():
    queue = Queue()
    with _clients_lock:
        _display_clients.append(queue)
    return queue

def unsubscribe(queue):
    with _clients_lock:
        if queue in _display_clients:
            _display_clients.remove(queue)

def send_event(event_type, data=None):
    payload = json.dumps({"type": event_type, "data": data})
    with _clients_lock:
        clients = list(_display_clients)
    for queue in clients:
        queue.put(payload)

def send_text(text):
    send_event("text", text)

def reset_display():
    send_event("reset")
