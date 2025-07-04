from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


# 📁 Set folder paths
train_dir = "archive/train"
test_dir = "archive/test"

# 📐 Image size & batch size
img_size = 48
batch_size = 64

# 🧪 Prepare image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 🧠 Build CNN model
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))  # 7 classes

# ⚙️ Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 💾 Save the best model
checkpoint = ModelCheckpoint("emotion_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# ▶️ Train the model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=25,
    callbacks=[checkpoint]
)

print("✅ Training complete. Model saved as emotion_model.h5")
