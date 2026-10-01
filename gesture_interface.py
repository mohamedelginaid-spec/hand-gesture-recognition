import time

import cv2
import mediapipe as mp
import numpy as np

WINDOW = "Touchless Interface"
WIDTH, HEIGHT = 1100, 700

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

state = {
    "enabled": False,
    "page": 0,
    "details": False,
    "message": "Press G to enable gesture control",
}

# حدود الأزرار: يسار، أعلى، يمين، أسفل
buttons = {
    "toggle": (40, 570, 280, 635),
    "previous": (40, 440, 230, 510),
    "next": (250, 440, 440, 510),
    "details": (460, 440, 700, 510),
}

pages = [
    ("Overview", [
        "DEMO-001 | Fictional record",
        "Visit: Demonstration session",
        "Use the buttons to explore this interface.",
    ]),
    ("Reports", [
        "Example report A",
        "Example report B",
        "No real patient information is displayed.",
    ]),
    ("Activity", [
        "Demo record created",
        "Example report added",
        "Interface ready for review",
    ]),
]


def activate(action):
    if action == "toggle":
        state["enabled"] = not state["enabled"]
        state["message"] = (
            "Enabled: separate fingers, then pinch to click"
            if state["enabled"]
            else "Gesture control paused"
        )
    elif action == "next":
        state["page"] = (state["page"] + 1) % len(pages)
        state["details"] = False
        state["message"] = "Next page"
    elif action == "previous":
        state["page"] = (state["page"] - 1) % len(pages)
        state["details"] = False
        state["message"] = "Previous page"
    elif action == "details":
        state["details"] = not state["details"]
        state["message"] = "Details toggled"


def button_at(x, y):
    for name, (x1, y1, x2, y2) in buttons.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        action = button_at(x, y)
        if action:
            activate(action)


def text(image, value, x, y, size=0.7, color=(225, 230, 240)):
    cv2.putText(
        image, value, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, size, color, 2,
        cv2.LINE_AA,
    )


camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, mouse_click)

cursor = None
armed = False
pinch_start = None
pinch_target = None
last_click = 0.0
last_time = time.monotonic()

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
                print("Could not read camera frame.")
                break

            now = time.monotonic()
            dt = now - last_time
            last_time = now

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hover = None
            ratio = None
            hand_found = bool(results.multi_hand_landmarks)

            if hand_found:
                hand = results.multi_hand_landmarks[0]
                drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                points = np.array([
                    [p.x * fw, p.y * fh]
                    for p in hand.landmark
                ])

                palm_width = np.linalg.norm(points[5] - points[17])
                if palm_width > 1:
                    ratio = (
                        np.linalg.norm(points[4] - points[8])
                        / palm_width
                    )

                if state["enabled"]:
                    # استخدام المنطقة الوسطى من الكاميرا للوصول
                    # إلى كل أجزاء الواجهة بسهولة
                    tip = hand.landmark[8]
                    target = np.array([
                        np.clip((tip.x - 0.15) / 0.70, 0, 1)
                        * (WIDTH - 1),
                        np.clip((tip.y - 0.15) / 0.70, 0, 1)
                        * (HEIGHT - 1),
                    ])

                    # تنعيم حركة المؤشر
                    alpha = 1 - np.exp(-dt / 0.08)
                    cursor = (
                        target if cursor is None
                        else cursor + alpha * (target - cursor)
                    )

                    hover = button_at(*cursor)

            if not hand_found or not state["enabled"] or ratio is None:
                cursor = None
                armed = False
                pinch_start = None
                pinch_target = None
            else:
                # يجب فتح الإصبعين قبل السماح بضغطة جديدة
                if ratio > 0.45:
                    armed = True
                    pinch_start = None
                    pinch_target = None

                elif ratio < 0.25 and armed:
                    if hover is None:
                        pinch_start = None
                        pinch_target = None

                    elif pinch_target != hover:
                        pinch_target = hover
                        pinch_start = now

                    elif (
                        pinch_start is not None
                        and now - pinch_start >= 0.15
                        and now - last_click >= 0.5
                    ):
                        activate(hover)
                        last_click = now
                        armed = False
                        pinch_start = None
                        pinch_target = None

                else:
                    pinch_start = None
                    pinch_target = None

            canvas = np.full(
                (HEIGHT, WIDTH, 3), (28, 24, 20), dtype=np.uint8
            )

            text(canvas, "TOUCHLESS INFORMATION VIEWER", 40, 55, 0.85)
            text(
                canvas, "Prototype | Fictional data only",
                40, 90, 0.55, (170, 180, 190)
            )

            title, lines = pages[state["page"]]
            text(canvas, title, 40, 160, 1.0)

            for i, line in enumerate(lines):
                text(canvas, line, 40, 215 + i * 40, 0.6)

            if state["details"]:
                text(
                    canvas, "Details panel: gesture interaction demo.",
                    40, 370, 0.6, (100, 220, 220)
                )

            button_labels = {
                "toggle": (
                    "Pause gestures" if state["enabled"]
                    else "Enable gestures"
                ),
                "previous": "Previous",
                "next": "Next",
                "details": "Show / hide details",
            }

            for name, (x1, y1, x2, y2) in buttons.items():
                color = (
                    (120, 100, 40) if hover == name
                    else (65, 60, 50)
                )
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                text(
                    canvas, button_labels[name],
                    x1 + 12, y1 + 40, 0.55
                )

            preview = cv2.resize(frame, (320, 240))
            canvas[130:370, 750:1070] = preview

            text(canvas, "Camera preview", 750, 115, 0.55)

            if ratio is not None:
                text(
                    canvas, f"Pinch ratio: {ratio:.2f}",
                    750, 405, 0.55
                )

            text(canvas, "G: enable / pause", 750, 460, 0.55)
            text(canvas, "Q: quit", 750, 495, 0.55)
            text(canvas, "Mouse buttons also work", 750, 530, 0.5)

            if cursor is not None and state["enabled"]:
                position = tuple(cursor.astype(int))
                color = (
                    (80, 220, 100) if ratio < 0.25
                    else (0, 220, 255)
                )
                cv2.circle(canvas, position, 12, color, 2)
                cv2.circle(canvas, position, 3, color, -1)

            text(canvas, state["message"], 40, 675, 0.52)

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("g"):
                activate("toggle")
                armed = False
                pinch_start = None
                pinch_target = None

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

finally:
    camera.release()
    cv2.destroyAllWindows()