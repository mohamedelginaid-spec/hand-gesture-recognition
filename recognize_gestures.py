from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

# تحميل النموذج الذي دربناه
folder = Path(__file__).resolve().parent
saved = joblib.load(folder / "gesture_model.joblib")
model = saved["model"]
labels = saved["labels"]

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils


def normalize(hand, width, height):
    # نفس تجهيز البيانات المستخدم وقت الجمع
    points = np.array(
        [[p.x * width, p.y * height] for p in hand.landmark],
        dtype=np.float32,
    )
    points = points - points[0]

    scale = np.max(np.abs(points))
    if scale < 0.000001:
        return None

    return (points / scale).flatten().reshape(1, -1)


camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

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

            text = "No hand detected"
            color = (0, 200, 255)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                sample = normalize(hand, width, height)

                if sample is not None:
                    prediction = int(model.predict(sample)[0])
                    text = f"Gesture: {labels[prediction]}"
                    color = (0, 255, 0)

            cv2.putText(
                frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2,
            )

            cv2.imshow("Gesture Recognition - Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()