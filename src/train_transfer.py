import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os

TRAIN_PATH = "../data/train"
VALID_PATH = "../data/valid"

IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 3

# ======================
# Data Generators
# ======================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = valid_datagen.flow_from_directory(
    VALID_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)

# ======================
# Load Pretrained Model
# ======================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze base

# ======================
# Custom Classification Head
# ======================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# ======================
# Phase 1: Train Top Layers
# ======================

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting Transfer Learning Training...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS
)

# ======================
# Phase 2: Fine-Tuning
# ======================

print("Starting Fine-Tuning...")

base_model.trainable = True

# Freeze first 100 layers (keep low-level features stable)
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Very low LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS
)

# ======================
# Save Model
# ======================

model.save("../models/plant_model_transfer.keras")

print("Transfer Learning + Fine-Tuning Model Saved Successfully!")