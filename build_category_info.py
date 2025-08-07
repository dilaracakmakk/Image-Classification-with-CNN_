import os
import json 
split_dir ="augmented_dataset"
output_path="category_info.json"

category_info={}

for category_name in os.listdir(split_dir):
    category_path = os.path.join(split_dir, category_name)
    if not os.path.isdir(category_path):
        continue

    for img_file in os.listdir(category_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(category_path, img_file)
            category_info[img_file] = [img_file.split('.')[0], category_name]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(category_info, f, ensure_ascii=False, indent=4)

print(f"{len(category_info)} kayıtlı görsel bulundu ve kaydedildi")