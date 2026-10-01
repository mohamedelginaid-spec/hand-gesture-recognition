import cv2
import mediapipe as mp

# أدوات اكتشاف اليد ورسم النقاط
mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

try:
    if not camera.isOpened():
        print("Could not open camera.")
    else:
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

                # تحويل ترتيب الألوان إلى الصيغة المطلوبة
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # اكتشاف نقاط اليد
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    index_tip = hand.landmark[8]

                    text = f"Index tip: x={index_tip.x:.2f}, y={index_tip.y:.2f}"

                    cv2.putText(
                        frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )

                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        drawing.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                cv2.imshow("Hand Tracking - Press Q to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
finally:
    camera.release()
    cv2.destroyAllWindows()