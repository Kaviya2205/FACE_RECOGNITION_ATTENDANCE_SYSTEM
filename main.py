import cv2
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from db import mark_attendance

model = tf.keras.models.load_model("models/face_model.h5")
labels_dict = np.load("models/labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)).reshape(1, 100, 100, 1) / 255.0
        prediction = model.predict(face)
        label = np.argmax(prediction)
        confidence = prediction[0][label]

        if confidence > 0.7:
            name = list(labels_dict.keys())[label]
            mark_attendance(name)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
