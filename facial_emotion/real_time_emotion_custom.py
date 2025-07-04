import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion labels based on training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Setup CSV logging
log_dir = "logs"
csv_path = os.path.join(log_dir, "emotion_log.csv")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns=["Timestamp", "Dominant Emotion"] + emotion_labels)
    df.to_csv(csv_path, index=False)

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            face = cv2.resize(roi_gray, (48, 48))
        except:
            continue

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict
        prediction = model.predict(face)[0]
        dominant_index = np.argmax(prediction)
        dominant_emotion = emotion_labels[dominant_index]

        # Log to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, dominant_emotion] + prediction.tolist()
        df = pd.DataFrame([row], columns=["Timestamp", "Dominant Emotion"] + emotion_labels)
        df.to_csv(csv_path, mode='a', header=False, index=False)

        # Display result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection complete. Log saved in:", csv_path)
