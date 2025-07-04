from deepface import DeepFace
import cv2

# Load the image
img = cv2.imread("face.jpg")

# Analyze the image for emotion
result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

# Print the dominant emotion
print("Detected Emotion:", result[0]['dominant_emotion'])

# Print all emotion probabilities
print("Emotion Probabilities:", result[0]['emotion'])
