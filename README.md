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
pip install -r requirements.txt
python app.py
pip install -r requirements.txt
python app.py
