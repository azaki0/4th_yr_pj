import logging
import threading
from flask import Flask, Response, render_template

from bridge import display_queue
import guide_agent_en as guide_agent

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    def generate():
        while True:
            chunk = display_queue.get()
            yield chunk + "\n"

    return Response(generate(), mimetype="text/plain")


def run_guide_agent():
    guide_agent.main()


if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    threading.Thread(target=run_guide_agent, daemon=True).start()

    print("--- Guide Robot Display Starting on Port 5000 ---")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
