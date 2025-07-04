# scripts/convert_to_tflite.py
from pathlib import Path
import tensorflow as tf

ROOT        = Path(__file__).resolve().parents[1]          # project root
SAVEDMODEL  = ROOT / "model" / "emotion_model"             # SavedModel dir
TFLITE_OUT  = ROOT / "model" / "emotion_model.tflite"      # output path

if not (SAVEDMODEL / "saved_model.pb").exists():
    raise FileNotFoundError(f"❌ SavedModel not found at {SAVEDMODEL}")

converter   = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL))
tflite_model = converter.convert()

with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

print(f"✅  TFLite model written to  {TFLITE_OUT}")
