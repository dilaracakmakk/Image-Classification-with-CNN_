# Clothing Classification with Tkinter and CNN

A textile-focused image classification desktop app built with Python, CNN models, and Tkinter GUI.

## 📌 Project Overview

This project aims to classify clothing images into categories such as `dress`, `pants`, and `tshirt` using a trained Convolutional Neural Network (CNN) model. The application also retrieves related model information (Model No, Company Model No, Group, Type) from a CSV file and displays them in the GUI.

### 💡 Key Features

- GUI interface built with **Tkinter**
- Image classification using **custom CNN or transfer learning models (e.g., MobileNetV2, EfficientNet)**
- CSV-based metadata lookup for matched predictions
- Category-based **filtering system** using dropdown or hamburger menu
- Visual similarity search (optional)
- Logging predictions to file or SQLite (optional)

## 🧠 Tech Stack

- Python 3
- TensorFlow / Keras
- Tkinter
- Pandas / NumPy
- PIL (Pillow)
- CSV / JSON handling

## 🚀 How to Run

```bash
## TR
## YÜKLEME 
1) Python 3.10+ install and virtual environment kurun ve sanal ortam:
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt

2) (Eğitim) Verinizi `dataset/` altına koyun. `augment.py` ile artırın, `train_model.py` ile eğitin.
   Üretilen modeli köke `simple_cnn_model.keras` adıyla yerleştirin (Git'e dahil edilmez).

3) (Uygulama) Klasöre `model_kodlari.csv` ekleyin (paylaşılmaz).
   Format için `model_kodlari.sample.csv` dosyasına bakıp kendi csv dosyanızı oluşturun.

## EN
## INSTALL
1) Install Python 3.10+ and create/activate a virtual environment:
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

2) Training: Put your data under dataset/. Use augment.py to augment it, then train_model.py to train.
Place the resulting model in the project root as simple_cnn_model.keras (not committed to Git).

3) Application: Add model_kodlari.csv to the project folder (do not share/commit).
Use model_kodlari.sample.csv as a reference to create your own CSV in the same format.
