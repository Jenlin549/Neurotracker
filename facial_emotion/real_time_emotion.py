import cv2
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import os
import time

# Create or open a CSV to log data
csv_file = "emotion_log.csv"

# Define emotion columns
columns = ["Timestamp", "Dominant Emotion", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Create file if not exists
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file, index=False)

# Start webcam
cap = cv2.VideoCapture(0)

# Loop for real-time detection
try:
    print("Starting webcam. Press Ctrl+C to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Use DeepFace to analyze emotions
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'  # more stable
            )
            emotions = result[0]['emotion']
            dominant = result[0]['dominant_emotion']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create row with all values
            row = {
                "Timestamp": timestamp,
                "Dominant Emotion": dominant,
                "Angry": emotions.get('angry', 0),
                "Disgust": emotions.get('disgust', 0),
                "Fear": emotions.get('fear', 0),
                "Happy": emotions.get('happy', 0),
                "Sad": emotions.get('sad', 0),
                "Surprise": emotions.get('surprise', 0),
                "Neutral": emotions.get('neutral', 0),
            }

            # Append to CSV
            df = pd.DataFrame([row])
            df.to_csv(csv_file, mode='a', header=False, index=False)

            print(f"[{timestamp}] Emotion: {dominant} | Probabilities: {emotions}")

        except Exception as e:
            print("Detection error:", e)

        # Wait 10 seconds before next detection
        time.sleep(10)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
