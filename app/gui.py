import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = "../model/emotion_lstm.h5"
TOKENIZER_PATH = "../model/tokenizer.pkl"
CLASS_FILE = "../model/class_names.txt"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load class names
with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)
    idx = np.argmax(pred)
    confidence = pred[0][idx] * 100

    return CLASS_NAMES[idx], confidence

def on_click():
    text = entry.get("1.0", "end-1c")
    if text.strip() == "":
        messagebox.showwarning("Error", "Please enter some text!")
        return

    emotion, conf = predict_emotion(text)
    result_label.config(text=f"Emotion: {emotion}  ({conf:.2f}%)")

# GUI Setup
root = tk.Tk()
root.title("Emotion Detection App")
root.geometry("400x300")

label = tk.Label(root, text="Enter your text:", font=("Arial", 14))
label.pack(pady=10)

entry = tk.Text(root, height=5, width=40)
entry.pack()

button = tk.Button(root, text="Predict Emotion", command=on_click)
button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
