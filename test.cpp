#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// --- WiFi Configuration ---
const char* WIFI_SSID = "azaki";
const char* WIFI_PASSWORD = "12345678";

// --- Servo Configuration (tweak these for your hardware) ---
const int SERVO_PIN = 13;
const int MOUTH_CLOSED_ANGLE = 90;
const int MOUTH_OPEN_ANGLE = 135;
const int STEP_SIZE = 5;
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

// --- mDNS ---
const char* MDNS_HOSTNAME = "robot-mouth";

// --- Servo State ---
Servo myServo;
int currentAngle = MOUTH_CLOSED_ANGLE;

// --- WebServer ---
WebServer server(80);

// --- Envelope Playback State ---
int* envelopeAngles = nullptr;
int envelopeLength = 0;
int envelopeIndex = 0;
int envelopeIntervalMs = 50;
unsigned long lastEnvelopeStep = 0;
bool envelopePlaying = false;

void stopEnvelope();

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Configure servo
  ESP32PWM::allocateTimer(0);
  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(MOUTH_CLOSED_ANGLE);
  currentAngle = MOUTH_CLOSED_ANGLE;

  // Connect WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected. IP: ");
  Serial.println(WiFi.localIP());

  // Start mDNS
  if (MDNS.begin(MDNS_HOSTNAME)) {
    Serial.printf("mDNS responder started: http://%s.local\n", MDNS_HOSTNAME);
    MDNS.addService("http", "tcp", 80);
  } else {
    Serial.println("mDNS responder failed!");
  }

  // HTTP endpoints
  server.on("/servo", HTTP_GET, handleServo);
  server.on("/envelope", HTTP_POST, handleEnvelope);
  server.on("/stop", HTTP_GET, handleStop);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("HTTP server started");

  Serial.printf("--- Robot Mouth Servo Ready ---\n");
  Serial.printf("Closed: %d, Open: %d\n", MOUTH_CLOSED_ANGLE, MOUTH_OPEN_ANGLE);
  Serial.println("Serial: a/A=left  d/D=right");
}

void loop() {
  server.handleClient();

  // Handle envelope playback timing
  if (envelopePlaying) {
    unsigned long now = millis();
    if (now - lastEnvelopeStep >= (unsigned long)envelopeIntervalMs) {
      lastEnvelopeStep = now;

      if (envelopeIndex < envelopeLength) {
        int angle = constrain(envelopeAngles[envelopeIndex], MIN_ANGLE, MAX_ANGLE);
        myServo.write(angle);
        currentAngle = angle;
        envelopeIndex++;
      } else {
        stopEnvelope();
      }
    }
  }

  // Serial keyboard control (fallback for testing)
  if (Serial.available() > 0) {
    char key = Serial.read();
    if (key == 'a' || key == 'A') {
      stopEnvelope();
      currentAngle -= STEP_SIZE;
      if (currentAngle < MIN_ANGLE) currentAngle = MIN_ANGLE;
      myServo.write(currentAngle);
      Serial.printf("Angle: %d\n", currentAngle);
    } else if (key == 'd' || key == 'D') {
      stopEnvelope();
      currentAngle += STEP_SIZE;
      if (currentAngle > MAX_ANGLE) currentAngle = MAX_ANGLE;
      myServo.write(currentAngle);
      Serial.printf("Angle: %d\n", currentAngle);
    }
  }
}

// --- HTTP Handlers ---

void handleServo() {
  if (server.hasArg("angle")) {
    int angle = server.arg("angle").toInt();
    angle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    stopEnvelope();
    myServo.write(angle);
    currentAngle = angle;
    server.send(200, "text/plain", String("OK ") + angle);
  } else {
    server.send(400, "text/plain", "Missing ?angle=N");
  }
}

void handleEnvelope() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "No body");
    return;
  }

  StaticJsonDocument<8192> doc;
  DeserializationError error = deserializeJson(doc, server.arg("plain"));
  if (error) {
    server.send(400, "text/plain", "JSON parse error");
    return;
  }

  JsonArray arr = doc["angles"];
  if (arr.isNull() || arr.size() == 0) {
    server.send(400, "text/plain", "Missing or empty angles array");
    return;
  }

  int intervalMs = doc["interval_ms"] | 50;
  if (intervalMs < 10) intervalMs = 10;

  stopEnvelope();

  envelopeLength = arr.size();
  envelopeAngles = new int[envelopeLength];
  for (int i = 0; i < envelopeLength; i++) {
    envelopeAngles[i] = arr[i].as<int>();
  }

  envelopeIntervalMs = intervalMs;
  envelopeIndex = 0;
  lastEnvelopeStep = millis();
  envelopePlaying = true;

  server.send(200, "text/plain", String("OK ") + envelopeLength + " frames");
}

void handleStop() {
  stopEnvelope();
  server.send(200, "text/plain", "Stopped");
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

void stopEnvelope() {
  envelopePlaying = false;
  envelopeIndex = 0;
  if (envelopeAngles != nullptr) {
    delete[] envelopeAngles;
    envelopeAngles = nullptr;
  }
  envelopeLength = 0;
  myServo.write(MOUTH_CLOSED_ANGLE);
  currentAngle = MOUTH_CLOSED_ANGLE;
}
