import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)
MODEL_PATH = "simple_cnn_model.keras"
DATASET_PATH = "dataset"
TEST_IMAGE = "E1-612_Model_Front__.jpg"

class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
print("Sınıflar:", class_names)

model = load_model(MODEL_PATH)

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f" Hata: {img_path} dosyası bulunamadı.")
        return
    
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]
        confidence = predictions[0][predicted_index]

        print(f"\n Tahmin: {predicted_label} ({confidence*100:.2f}%)")

    except Exception as e:
        print(f"Tahmin sırasında hata oluştu: {e}")

predict_image(TEST_IMAGE)
