# scripts/train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1) ---------- LOAD FEATURES ----------
X = np.load("../features/X_combined.npy")
y = np.load("../features/y_combined.npy")

# If you used 8 emotions (0-7) map them to one-hot vectors
y_cat = to_categorical(y, num_classes=8)

# 2) ---------- TRAIN / VAL SPLIT ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# 3) ---------- BUILD SIMPLE DENSE MODEL ----------
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(8, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 4) ---------- TRAIN ----------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# 5) ---------- SAVE ----------
model.export("../model/emotion_model")
print("✅ Model saved in ../model/emotion_model")
