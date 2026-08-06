import logging
import threading
from queue import Queue

from flask import Flask, Response, jsonify, render_template, request

from bridge import reset_display, send_event, send_text
from demo_interactions import INTERACTIONS, run_interaction

app = Flask(__name__)
interaction_queue = Queue()

@app.route("/")
def index():
    return render_template("index.html", interactions=INTERACTIONS)

@app.route("/admin")
def admin():
    return render_template("admin.html", interactions=INTERACTIONS)

@app.route("/api/interactions")
def interactions():
    return jsonify({"interactions": INTERACTIONS})

@app.route("/api/interactions/<interaction_id>/play", methods=["POST"])
def play_interaction(interaction_id):
    interaction = next((item for item in INTERACTIONS if item["id"] == interaction_id), None)
    if interaction is None:
        return jsonify({"error": "unknown interaction"}), 404

    interaction_queue.put(interaction)
    return jsonify({"queued": True, "interaction": interaction})

@app.route("/api/prompt", methods=["POST"])
def submit_prompt():
    data = request.get_json(silent=True) or {}
    interaction_id = data.get("interactionId") or data.get("prompt")
    if not interaction_id:
        return jsonify({"error": "interactionId is required"}), 400
    return play_interaction(str(interaction_id))

@app.route("/stream")
def stream():
    def generate():
        while True:
            chunk = display_queue.get()
            yield chunk + "\n"

    from bridge import display_queue

    return Response(generate(), mimetype="text/plain")

def run_demo_worker():
    while True:
        interaction = interaction_queue.get()
        try:
            run_interaction(interaction)
        except Exception as exc:
            reset_display()
            send_text(f"Demo interaction error: {exc}")
            send_event("done")
        finally:
            interaction_queue.task_done()

if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    threading.Thread(target=run_demo_worker, daemon=True).start()

    print("Guide Robot Demo Display Starting on Port 7070")
    app.run(host="0.0.0.0", port=7070, debug=False, threaded=True, use_reloader=False)
