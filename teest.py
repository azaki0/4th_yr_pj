import cv2
import time
import threading
from ultralytics import YOLO

# 1. Load the ultra-lightweight vision model
vision_model = YOLO("yolov8n.pt") 

url = "http://192.168.137.166"  

# Threaded Camera Class to eliminate network buffer lag
class FreshFrameStream:
    def __init__(self, stream_url):
        self.cap = cv2.VideoCapture(stream_url)
        self.ret, self.frame = False, None
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Kills thread automatically when main script exits
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01) # Yield to prevent CPU thrashing

    def read(self):
        with self.read_lock:
            # Returns a copy of the newest frame to prevent corruption during drawing
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        self.cap.release()

# Settings to manage detection frequency
last_detection_time = 0
COOLDOWN_SECONDS = 5  

print("Robot eyes are online. Booting background stream thread...")
stream = FreshFrameStream(url).start()
time.sleep(2.0) # Let the stream stabilize

print("Scanning for people...")

try:
    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue # Wait until a valid frame is copied from the thread

        # Run YOLO inference 
        # verbose=False keeps the console clean for your robot alerts
        results = vision_model(frame, stream=True, verbose=False)
        person_detected = False
        
        for r in results:
            for box in r.boxes:
                # Class 0 = person; Confidence > 65%
                if int(box.cls) == 0 and float(box.conf) > 0.65:
                    person_detected = True
                    
                    # Bounding box mathematics
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {float(box.conf):.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the camera window
        cv2.imshow("Robot Vision Feed", frame)

        # Action trigger with cooldown
        current_time = time.time()
        if person_detected and (current_time - last_detection_time > COOLDOWN_SECONDS):
            print(f"A person is standing in front of the robot! ({time.strftime('%H:%M:%S')})")
            
            # TODO: Trigger your main LLM wake-up routine here!
            
            last_detection_time = current_time

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean shutdown
    stream.stop()
    cv2.destroyAllWindows()
    print("Robot eyes shut down successfully.")