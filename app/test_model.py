import numpy as np
import tensorflow as tf
import pickle

MODEL_PATH = "../model/emotion_lstm.h5"
TOKENIZER_PATH = "../model/tokenizer.pkl"
CLASS_FILE = "../model/class_names.txt"

# Load class names
with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)
    idx = np.argmax(pred)
    confidence = pred[0][idx] * 100

    print("\nEmotion:", CLASS_NAMES[idx])
    print("Confidence:", round(confidence, 2), "%")

# Test here
predict_emotion("I am extremely sad today!")
