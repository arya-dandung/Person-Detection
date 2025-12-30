from ultralytics import YOLO
import requests
import time

WAHA_URL = "http://localhost:3000/api/sendText"
API_KEY = "63237478d6fb42e3af3b171a33dae5a6"
CHAT_ID = "62813xxxxxxx@c.us"

model = YOLO("yolov8n.pt")

detected_ids = set()
last_send = 0
COOLDOWN = 30

results = model.track(
    source=1,
    classes=[0],
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    show=True
)

for r in results:
    if r.boxes is None or r.boxes.id is None:
        continue

    now = time.time()

    for tid in r.boxes.id.tolist():
        pid = int(tid)

        if pid in detected_ids:
            continue
        if now - last_send < COOLDOWN:
            continue

        detected_ids.add(pid)
        last_send = now

        payload = {
            "session": "default",
            "chatId": CHAT_ID,
            "text": (
                "🚨 ORANG TERDETEKSI 🚨\n"
                f"ID Tracking: {pid}\n"
                f"Waktu: {time.strftime('%d-%m-%Y %H:%M:%S')}"
            )
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": API_KEY
        }

        resp = requests.post(WAHA_URL, json=payload, headers=headers)
        print("WA STATUS:", resp.status_code, resp.text)
