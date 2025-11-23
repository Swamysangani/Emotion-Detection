# 🎭 Emotion Detection Using LSTM (NLP + Streamlit + Tkinter)

A machine learning project that detects human **emotions** from text using an **LSTM neural network**.  
Built with **TensorFlow**, **Streamlit**, **Tkinter**, and the **ISEAR Emotion Dataset**.

This project provides **two ways to use the model**:

1. 🌐 **Streamlit Web App** (recommended)  
2. 🖥️ **Tkinter GUI App** (offline fallback)  

---

## 🚀 Features

- Detects emotions from text such as:
  - 😊 Joy  
  - 😢 Sadness  
  - 😡 Anger  
  - 😨 Fear  
  - 😲 Surprise  
  - ❤️ Love  
- Clean LSTM neural network for text classification  
- Streamlit web interface  
- Tkinter GUI (offline fallback)  
- Modular project structure  
- Fast + accurate  

---

## 📂 Project Structure

```
Emotion_detection/
│
├── app/
│   ├── emotion_app.py        # Streamlit web app
│   ├── gui.py                # Tkinter GUI fallback
│   └── test_model.py         # Quick testing script
│
├── dataset/
│   ├── train.txt
│   ├── test.txt
│   ├── val.txt
│   └── emotion.csv           # Generated dataset
│
├── training/
│   └── train_model.py        # LSTM model training pipeline
│
├── model/
│   ├── emotion_lstm.h5       # Final model
│   ├── tokenizer.pkl         # Tokenizer
│   └── class_names.txt       # Emotion labels
│
├── venv/                     # Virtual environment (ignored)
├── .gitignore
└── README.md
```

---

## 📦 Requirements

Install required packages:

```bash
pip install numpy pandas tensorflow scikit-learn nltk matplotlib streamlit
```

---

## 🧠 Training the Model

1. Place `train.txt`, `test.txt`, and `val.txt` into `/dataset`.
2. Convert them into a CSV:

```bash
cd dataset
python convert_to_csv.py
```

3. Train the model:

```bash
cd training
python train_model.py
```

This generates:

- `emotion_lstm.h5`
- `tokenizer.pkl`
- `class_names.txt`

inside the `model/` folder.

---

## 🌐 Running the Streamlit Web App

```bash
cd app
streamlit run emotion_app.py
```

Access the app at:

```
http://localhost:8501/
```

---

## 🖥️ Running the Tkinter GUI App

```bash
cd app
python gui.py
```

Works completely offline and uses the same trained model.

---

## 🧪 Quick Model Test

Run quick test:

```bash
cd app
python test_model.py
```

---

## 🧬 Model Architecture

- **Embedding Layer (10k vocab)**
- **LSTM Layer (128 units)**
- **Dense Layer (64 units, ReLU)**
- **Output Layer (Softmax with 6 classes)**

Trained on merged ISEAR dataset.

---

## 🗂 Dataset

This project uses the **ISEAR Emotion Dataset**, provided in text files:

- `train.txt`
- `test.txt`
- `val.txt`

Each line:

```
text ; emotion
```

Converted into a single CSV for training.

---

## 📌 Notes

- `.h5` and `.pkl` files are NOT uploaded to GitHub due to file size limits.  
- You can upload them to Google Drive and add a link.  
- Streamlit version → Best UI  
- Tkinter version → Offline fallback  

---

## ⭐ Future Enhancements

- Deploy to Streamlit Cloud  
- Add emotion emojis in UI  
- Confidence bar chart  
- Convert speech → text → emotion  
- BERT-based model  

---

## ❤️ Credits

Developed by **Shruthi Ramesh (Sam)** ❤️  
Emotion Classification using NLP + LSTM  
