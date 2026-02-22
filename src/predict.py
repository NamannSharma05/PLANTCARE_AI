import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

MODEL_PATH = "../models/plant_model.h5"
IMG_SIZE = 128

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Get class labels from training folder
import os
class_names = sorted([
    folder for folder in os.listdir("../data/train")
    if os.path.isdir(os.path.join("../data/train", folder))
])

def predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print("Predicted Disease:", predicted_class)
    print("Confidence:", round(float(confidence) * 100, 2), "%")
    print("prediction raw output:", predictions)

if __name__ == "__main__":
    print("Script started")
    print("image path:", sys.argv[1])
    img_path = sys.argv[1]
    predict(img_path)