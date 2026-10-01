import cv2

# فتح الكاميرا الافتراضية على macOS
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

try:
    if not camera.isOpened():
        print("Could not open camera.")
    else:
        print("Press Q inside the video window to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame.")
                break

            # عكس الصورة لتظهر مثل المرآة
            frame = cv2.flip(frame, 1)

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()