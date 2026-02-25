import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json
import os

# ============================================
# SETTINGS - உங்க dataset path இங்க போடு
# ============================================
DATASET_PATH = "PlantVillage"  # உங்க dataset folder name
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # 10 rounds train பண்ணும்

print("=" * 50)
print("🌿 PlantCure AI - Model Training Started!")
print("=" * 50)

# ============================================
# STEP 1: Dataset Load பண்ணு
# ============================================
print("\n📂 Loading Dataset...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Class names save பண்ணு
class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)

print(f"✅ Total Classes Found: {num_classes}")
print(f"✅ Training Images: {train_data.samples}")
print(f"✅ Validation Images: {val_data.samples}")

# Save class names
with open('model/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✅ Class names saved!")

# ============================================
# STEP 2: AI Model Create பண்ணு
# ============================================
print("\n🧠 Building AI Model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ AI Model Ready!")
print(f"✅ Total Parameters: {model.count_params():,}")

# ============================================
# STEP 3: Training Start!
# ============================================
print("\n🚀 Training Started! Please wait...")
print(f"Total Epochs: {EPOCHS}")
print("-" * 50)

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    verbose=1
)

# ============================================
# STEP 4: Model Save பண்ணு
# ============================================
print("\n💾 Saving Model...")
model.save('model/plantcure_model.h5')
print("✅ Model saved to model/plantcure_model.h5")

# Final accuracy print பண்ணு
final_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print(f"✅ Training Accuracy: {final_acc:.2f}%")
print(f"✅ Validation Accuracy: {final_val_acc:.2f}%")
print("=" * 50)
print("\n✅ Now run: python app.py")
