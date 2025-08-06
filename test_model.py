import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
MODEL_PATH = "clothes_model_mobilenet.keras"
CLASS_NAMES_FILE = "class_names.npy"
TEST_DIR = "dataset_split/test"

model = load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_FILE)

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"\n Toplam test doğruluğu: {accuracy * 100:.2f}%")

labels = list(test_gen.class_indices.keys())
class_correct = {label: 0 for label in labels}
class_total = {label: 0 for label in labels}

for pred, true in zip(predicted_classes, true_classes):
    true_label = labels[true]
    if pred == true:
        class_correct[true_label] += 1
    class_total[true_label] += 1

print("\n Sınıf bazlı başarı oranları:")
for label in labels:
    total = class_total[label]
    correct = class_correct[label]
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"{label:10}: {correct}/{total} doğru → %{acc:.2f}")
