import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

dataset_path = "dataset/"
categories = os.listdir(dataset_path)
labels_dict = {name: i for i, name in enumerate(categories)}

X, y = [], []

for person in categories:
    path = os.path.join(dataset_path, person)
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))
        X.append(img)
        y.append(labels_dict[person])

X = np.array(X).reshape(-1, 100, 100, 1) / 255.0
y = to_categorical(y, len(categories))

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(100, 100, 1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(categories), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=8)
model.save("models/face_model.h5")
np.save("models/labels.npy", labels_dict)

print("Model trained and saved!")
