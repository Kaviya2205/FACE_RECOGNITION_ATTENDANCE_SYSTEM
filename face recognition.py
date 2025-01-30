import cv2
import os

dataset_path = "dataset/"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_face(name):
    cap = cv2.VideoCapture(0)
    count = 0
    user_path = os.path.join(dataset_path, name)
    os.makedirs(user_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(f"{user_path}/{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        if count >= 30 or cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

name = input("Enter Name: ")
capture_face(name)
