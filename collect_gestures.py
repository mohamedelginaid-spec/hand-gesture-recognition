import csv
import uuid
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ملف البيانات بجانب ملف البرنامج
data_path = Path(__file__).with_name("gestures.csv")
session_id = uuid.uuid4().hex[:8]

labels = {0: "Open", 1: "Fist", 2: "Point"}
counts = {0: 0, 1: 0, 2: 0}

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils


def normalize(hand, width, height):
    # تحويل الإحداثيات إلى مقياس موحد بالبكسل
    points = np.array(
        [[p.x * width, p.y * height] for p in hand.landmark],
        dtype=np.float32,
    )

    # جعل الرسغ نقطة الأصل
    points = points - points[0]

    # تقليل تأثير حجم اليد داخل الصورة
    scale = np.max(np.abs(points))
    if scale < 0.000001:
        return None

    return (points / scale).flatten().tolist()


camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    new_file = not data_path.exists() or data_path.stat().st_size == 0

    with data_path.open("a", newline="") as file:
        writer = csv.writer(file)

        if new_file:
            features = [
                f"{axis}{i}"
                for i in range(21)
                for axis in ("x", "y")
            ]
            writer.writerow(["session", "label"] + features)

        with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                sample = None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    sample = normalize(hand, width, height)
                    drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                status = (
                    f"0 Open: {counts[0]} | "
                    f"1 Fist: {counts[1]} | "
                    f"2 Point: {counts[2]}"
                )
                cv2.putText(
                    frame, status, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2,
                )

                cv2.imshow("Collect gestures - Q to quit", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key in (ord("0"), ord("1"), ord("2")):
                    if sample is None:
                        print("No hand detected. Sample not saved.")
                        continue

                    label = key - ord("0")
                    writer.writerow([session_id, label] + sample)
                    file.flush()
                    counts[label] += 1
                    print(f"Saved {labels[label]}: {counts[label]}")

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Data file: {data_path}")