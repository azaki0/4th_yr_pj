import logging
from flask import Flask, render_template, Response
import threading
from medical_ai.bridge import text_display_queue
import medical_ai.test_med as test_med

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        while True:
            chunk = text_display_queue.get()
            yield chunk
            
    return Response(generate(), mimetype='text/plain')

def run_medical_agent():
    test_med.text()

if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) 

    threading.Thread(target=run_medical_agent, daemon=True).start()
    
    print("Flask Server Starting on Port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)