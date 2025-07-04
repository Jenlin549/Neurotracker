import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path

# Set project root directory
ROOT = Path(__file__).resolve().parents[1]

# Load & preprocess audio file
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # ✅ Extract 40 MFCCs
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Run inference using TFLite model
def run_inference(mfcc_features):
    model_path = ROOT / "model" / "emotion_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(mfcc_features, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_label_index = np.argmax(output_data)
    return predicted_label_index, output_data[0]

# Label map (same order as model training)
LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# ✅ Path to your test .wav file
test_file = (
    ROOT / "datasets" / "CREMA-D" / "AudioWAV" / "1001_IOM_HAP_XX.wav"
)

# Run
mfcc = extract_mfcc(test_file)
predicted_index, prediction_vector = run_inference(mfcc)

print(f"\n✅ Predicted Emotion: {LABELS[predicted_index]}")
print(f"📊 Full Output: {prediction_vector}")
