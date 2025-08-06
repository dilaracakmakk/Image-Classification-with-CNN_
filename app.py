import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import pickle
import json
import os
import re
from functools import partial
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Model
from prediction_log import init_db, save_prediction

IMG_SIZE = (224, 224)
MODEL_PATH = "simple_cnn_model.keras"
DATASET_PATH = "augmented_dataset"
CLASS_NAMES_FILE = "class_names.npy"

df = pd.read_csv("model_kodlari.csv", delimiter=";", on_bad_lines="skip")
df.columns = df.columns.str.strip()
df["Model No"] = df["Model No"].astype(str).str.strip().str.lower()
df["Firma Model No"] = df["Firma Model No"].astype(str).str.strip().str.lower()



def load_class_indices(path="class_indices.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return [raw[str(i)] for i in range(len(raw))]
    else:
        return ["dress", "pants", "tshirt"]

CATEGORIES = load_class_indices() 
model = load_model(MODEL_PATH)

def load_category_json(path="category_info.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_category_json(data, path="category_info.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_folder_code(file_path):
    match = re.search(r"E\d{2}-S\d{2}", file_path)
    return match.group(0) if match else None

def get_real_class_from_path(file_path):
    abs_path = os.path.abspath(file_path)
    parts = abs_path.split(os.path.sep)
    try:
        idx = parts.index("augmented_dataset")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "-"




def get_matching_row(predicted_label):
    predicted_label = str(predicted_label).strip().lower()
    for _, row in df.iterrows():
        model_no = str(row["Model No"]).strip().lower()
        firma_model_no = str(row["Firma Model No"]).strip().lower()
        if predicted_label == model_no or predicted_label == firma_model_no:
            return row
    return None






def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = CATEGORIES[predicted_index]

    for idx, score in enumerate(prediction[0]):
        print(f"{CATEGORIES[idx]}: %{score * 100:.2f}")

    return predicted_class




def cosine_similarity_np(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_feature_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base.input, outputs=base.output)

feature_model = load_feature_model()

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img = preprocess_input(np.expand_dims(np.array(img), axis=0))
    return feature_model.predict(img).flatten()

def find_similar_images(image_path, top_k=3):
    with open("features.pkl", "rb") as f:
        features_list, image_paths = pickle.load(f)
    new_feat = extract_features(image_path)
    sims = [cosine_similarity_np(new_feat, feat) for feat in features_list]
    indices = np.argsort(sims)[::-1][:top_k]
    return [(image_paths[i], sims[i]) for i in indices]

def show_info_popup(image_path):
    popup = tk.Toplevel()
    popup.title("Ürün Bilgisi")
    popup.geometry("320x180")
    popup.configure(bg="#ffffff")

    tk.Label(popup, text=f"Dosya: {os.path.basename(image_path)}", bg="white").pack(pady=5)
    tk.Label(popup, text=f"(Detaylı bilgi görsel seçildikten sonra sağ tarafta gösterilir)", bg="white").pack()

def show_similar_images(result_list):
    sim_window = tk.Toplevel(root)
    sim_window.title("Benzer Fotoğraflar")
    sim_window.configure(bg="white")

    for idx, (path, score) in enumerate(result_list):
        frame = tk.Frame(sim_window, bg="#ffffff", bd=1, relief="solid")
        frame.grid(row=0, column=idx, padx=15, pady=10)

        img = Image.open(path).resize((160, 160))
        img = ImageTk.PhotoImage(img)

        lbl = tk.Label(frame, image=img, bg="#ffffff", cursor="hand2")
        lbl.image = img
        lbl.pack()
        lbl.bind("<Button-1>", partial(on_similar_click, image_path=path))

        tk.Label(frame, text=f"Benzerlik: {score:.2f}", bg="#ffffff").pack()

def on_similar_click(event, image_path):
    show_info_popup(image_path)

def extract_model_no_from_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    name = re.sub(r"\(.*?\)", "", name)
    name = name.split("_")[0]
    return name.strip()

def process_file_for_prediction(file_path):
    print("Tahmin ediliyor...")
    predicted_label = predict_image(file_path)

    tahmin_map = {
        "dress": "Elbise",
        "pants": "Pantolon",
        "tshirt": "Tişört"
    }

    try:
        df["Firma Model No"] = df["Firma Model No"].astype(str).str.strip().str.lower()
        df["Giysi Cinsi"] = df["Giysi Cinsi"].astype(str).str.strip().str.lower()
        df["Model No"] = df["Model No"].astype(str).str.strip().str.lower()

        model_no_clean = extract_model_no_from_filename(file_path).strip().lower()
        tahmin_turkce = tahmin_map.get(predicted_label.lower(), predicted_label).strip().lower()

        df.dropna(subset=["Firma Model No", "Giysi Cinsi"], inplace=True)

        matched = df[
            (df["Firma Model No"] == model_no_clean) |
            (df["Model No"] == model_no_clean)
        ]

    except Exception as e:
        print("Eşleşme sırasında hata oluştu:", e)
        matched = pd.DataFrame()

    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    if not matched.empty:
        row = matched.iloc[0]
        model_no = row["Model No"]
        firma_model_no = row["Firma Model No"]
        giysi_grubu = row["Giysi Grubu"]
        giysi_cinsi = row["Giysi Cinsi"]
    else:
        model_no = extract_model_no_from_filename(file_path)
        firma_model_no = "-"
        giysi_grubu = "-"
        giysi_cinsi = tahmin_map.get(predicted_label.lower(), predicted_label)

    text = (
        f"Model No: {model_no}\n"
        f"Firma Model No: {firma_model_no}\n"
        f"Giysi Grubu: {giysi_grubu}\n"
        f"Giysi Cinsi: {giysi_cinsi}"
    )
    result_label.config(text=text)

    try:
        save_prediction(
            folder_name=os.path.basename(file_path),
            real_class=get_real_class_from_path(file_path) or "-",
            model_no=model_no,
            clothes_type=giysi_cinsi
        )
    except Exception as e:
        print(f"Veritabanı kaydı başarısız: {e}")

    try:
        similar = find_similar_images(file_path)
        show_similar_images(similar)
    except Exception as e:
        print("Benzerlik hatası:", e)

def upload_and_predict():
    path = filedialog.askopenfilename(filetypes=[("Görsel", "*.jpg *.jpeg *.png")])
    if path:
        process_file_for_prediction(path)

def apply_filters():
    selected_cat = category_combo.get().lower()
    model_kw = model_var.get().lower()
    clothes_kw = clothes_var.get().lower()

    data = load_category_json()
    matches = []

    for filename, values in data.items():
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            continue
        model_code, category = values
        if (selected_cat in category.lower() or not selected_cat) and \
           (model_kw in model_code.lower()) and \
           (clothes_kw in category.lower() or not clothes_kw):
            matches.append(f"{filename} → {model_code} | {category}")

    filter_result.delete("1.0", tk.END)
    if matches:
        filter_result.insert(tk.END, "\n".join(matches))
    else:
        filter_result.insert(tk.END, "Eşleşme bulunamadı.")

root = tk.Tk()
info_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
info_label.pack(pady=10)

root.title("Kıyafet Tanımlama Sistemi")
root.geometry("950x820")
root.configure(bg="#f6f7fb")

header = tk.Label(root, text="Kıyafet Tanımlama Paneli", font=("Segoe UI", 22, "bold"), bg="#f6f7fb")
header.pack(pady=20)

content_frame = tk.Frame(root, bg="white", bd=1, relief="solid")
content_frame.pack(padx=20, pady=10)

img_label = tk.Label(content_frame, bg="#ffffff")
img_label.grid(row=0, column=0, padx=20, pady=20)

result_label = tk.Label(content_frame, text="", font=("Segoe UI", 11), bg="#ffffff", justify="left")
result_label.grid(row=0, column=1, padx=20, pady=20, sticky="n")

filter_frame = tk.Frame(content_frame, bg="#ffffff")
filter_frame.grid(row=0, column=2, padx=20, pady=20, sticky="n")

ttk.Label(filter_frame, text=" Gelişmiş Filtreleme", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=10)

category_var = tk.StringVar()
ttk.Label(filter_frame, text="Kategori").pack(anchor="w")
category_combo = ttk.Combobox(filter_frame, textvariable=category_var, values=CATEGORIES, state="readonly")
category_combo.pack(pady=5)

model_var = tk.StringVar()
ttk.Label(filter_frame, text="Model Kodu").pack(anchor="w")
model_entry = ttk.Entry(filter_frame, textvariable=model_var)
model_entry.pack(pady=5)

clothes_var = tk.StringVar()
ttk.Label(filter_frame, text="Giysi Türü").pack(anchor="w")
clothes_entry = ttk.Entry(filter_frame, textvariable=clothes_var)
clothes_entry.pack(pady=5)

filter_result = tk.Text(filter_frame, height=12, width=40)
filter_result.pack(pady=5)

ttk.Button(filter_frame, text=" Ara", command=apply_filters).pack(pady=10)

select_btn = tk.Button(
    root, text="Görsel Seç", command=upload_and_predict,
    bg="#1abc9c", fg="white", font=("Segoe UI", 12, "bold"),
    padx=20, pady=10, relief="flat", cursor="hand2"
)
select_btn.pack(pady=20)

progress_label = tk.Label(root, text="", fg="green", bg="#f6f7fb", font=("Segoe UI", 10))
progress_label.pack()

footer = tk.Label(root, text="@Emirali Akıllı Görsel Tanıma", bg="#f6f7fb", fg="#95a5a6", font=("Segoe UI", 9))
footer.pack(pady=10)

DEFAULT_IMAGE_PATH = "emirali.jpg"
if os.path.exists(DEFAULT_IMAGE_PATH):
    img = Image.open(DEFAULT_IMAGE_PATH).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    result_label.config(text="")

init_db()
root.mainloop()
