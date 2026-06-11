import logging
import threading
from queue import Queue
from flask import Flask, Response, jsonify, render_template, request
from bridge import display_queue
from memory_store import (
    add_manual_memory,
    clear_conversations,
    populate_uni_info,
    rebuild_conversation_embeddings,
    remove_last_conversation,
    status as memory_status,
)
from remote_agent_client import get_config, handle_prompt, set_config

app = Flask(__name__)
prompt_queue = Queue()
current_mode = {"language": "en"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/api/mode", methods=["GET", "POST"])
def mode():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        language = data.get("language", "").lower()
        if language not in {"en", "mm"}:
            return jsonify({"error": "language must be en or mm"}), 400

        current_mode["language"] = language

    return jsonify(current_mode)

@app.route("/api/prompt", methods=["POST"])
def submit_prompt():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    prompt_queue.put({"language": current_mode["language"], "prompt": prompt})
    return jsonify({"queued": True, "language": current_mode["language"]})

@app.route("/api/remote", methods=["GET", "POST"])
def remote_config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        set_config(url=data.get("url"), token=data.get("token"))

    return jsonify(get_config())

@app.route("/api/admin/db", methods=["GET", "POST"])
def admin_db():
    if request.method == "GET":
        return jsonify(memory_status())

    data = request.get_json(silent=True) or {}
    command = data.get("command")

    try:
        if command == "populate_uni_info":
            result = populate_uni_info()
        elif command == "rebuild_conversations":
            result = rebuild_conversation_embeddings()
        elif command == "remove_last":
            result = remove_last_conversation()
        elif command == "clear_conversations":
            result = clear_conversations()
        elif command == "add_memory":
            result = {"id": add_manual_memory(data.get("text", ""))}
        else:
            return jsonify({"error": "unknown command"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    result["status"] = memory_status()
    return jsonify(result)

@app.route("/stream")
def stream():
    def generate():
        while True:
            chunk = display_queue.get()
            yield chunk + "\n"

    return Response(generate(), mimetype="text/plain")

def run_guide_agent():
    print("Guide Robot prompt worker ready.")

    while True:
        item = prompt_queue.get()
        try:
            handle_prompt(item["prompt"], item["language"])
        except Exception as exc:
            print(f"Prompt worker error: {exc}")
        finally:
            prompt_queue.task_done()

if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    threading.Thread(target=run_guide_agent, daemon=True).start()

    print("Display Starting on Port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
