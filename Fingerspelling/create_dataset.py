import os
import pickle

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

data = []
labels = []

total_images = 0
detected_images = 0

for label in os.listdir(DATA_DIR):
    print(f"Processing class {label}...")
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        total_images += 1

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )
        if coords.shape != (21, 3):
            continue

        if handedness == "Left":
            coords[:, 0] = 1.0 - coords[:, 0]

        wrist = coords[0].copy()
        coords = coords - wrist

        scale = np.max(np.abs(coords))
        if scale < 1e-6:
            continue

        coords = coords / scale
        data.append(coords.flatten().tolist())
        labels.append(label)

        detected_images += 1

print("Finished processing")
print("Number of samples:", len(data))
print("Total images processed:", total_images) # 26000 of course
print("Images with detected hands:", detected_images) # Result was 25858.

pickle_path = os.path.join(BASE_DIR, "data.pickle")

with open(pickle_path, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Saved to:", pickle_path)

hands.close()