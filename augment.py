import os
from PIL import Image, ImageEnhance, ImageOps
import random

input_dir = "dataset_split/train"
output_dir = "augmented_dataset"
os.makedirs(output_dir, exist_ok=True)

def augment_image(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    
    if random.random() < 0.5:
        angle = random.uniform(-7, 7)  # daha az bozulma
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))
    
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    return img

AUG_PER_IMAGE = 3

for category in os.listdir(input_dir):
    input_path = os.path.join(input_dir, category)
    output_path = os.path.join(output_dir, category)
    os.makedirs(output_path, exist_ok=True)

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Hata ({img_name}):", e)
            continue

        img.save(os.path.join(output_path, f"real_{img_name}"))

        for i in range(AUG_PER_IMAGE):
            aug_img = augment_image(img)
            aug_img.save(os.path.join(output_path, f"aug_{img_name[:-4]}_{i}.jpg"))
