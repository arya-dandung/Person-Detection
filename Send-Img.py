from ultralytics import YOLO
import cv2
import requests
import time
import base64
import os

# ==============================
# WAHA CONFIG
# ==============================
WAHA_URL = "http://localhost:3000/api/sendImage"
API_KEY = "63237478d6fb42e3af3b171a33dae5a6"
SESSION = "default"
CHAT_ID = "62813xxxxxx@c.us"

# ==============================
# YOLO CONFIG
# ==============================
MODEL_PATH = "yolov8n.pt"
SOURCE = 1            # 0 = webcam
CONFIDENCE = 0.5

# ==============================
# COOLDOWN (ANTI SPAM)
# ==============================
SEND_COOLDOWN = 30    # detik
last_send_time = 0

# ==============================
# LOAD MODEL
# ==============================
model = YOLO(MODEL_PATH)

detected_ids = set()

# ==============================
# START DETECTION
# ==============================
results = model.track(
    source=SOURCE,
    classes=[0],               # person only
    conf=CONFIDENCE,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    show=True
)

print("🚀 People detection started...")

for r in results:
    if r.boxes is None or r.boxes.id is None:
        continue

    frame = r.orig_img
    now = time.time()

    for track_id in r.boxes.id.tolist():
        person_id = int(track_id)

        # kirim hanya untuk ID baru
        if person_id in detected_ids:
            continue

        # cooldown global
        if now - last_send_time < SEND_COOLDOWN:
            continue

        detected_ids.add(person_id)
        last_send_time = now

        # ==============================
        # SAVE IMAGE
        # ==============================
        filename = f"person_{person_id}.jpg"
        cv2.imwrite(filename, frame)

        # ==============================
        # BASE64 ENCODE
        # ==============================
        with open(filename, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ==============================
        # WAHA PAYLOAD
        # ==============================
        payload = {
            "session": SESSION,
            "chatId": CHAT_ID,
            "file": {
                "mimetype": "image/jpeg",
                "filename": filename,
                "data": img_b64
            },
            "caption": (
                "🚨 DETEKSI ORANG BARU 🚨\n"
                f"ID Tracking: {person_id}\n"
                f"Waktu: {time.strftime('%d-%m-%Y %H:%M:%S')}"
            )
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": API_KEY
        }

        # ==============================
        # SEND TO WAHA
        # ==============================
        resp = requests.post(
            WAHA_URL,
            json=payload,
            headers=headers,
            timeout=10
        )

        print("📤 WA STATUS:", resp.status_code)
        print("📩 WA RESPONSE:", resp.text)

        os.remove(filename)

