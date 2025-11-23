import streamlit as st
import tensorflow as tf
import numpy as np
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

# Streamlit UI
st.title("🎭 Emotion Detection App")

st.write("Type any sentence and the model will predict your emotion.")

user_input = st.text_area("Enter your text here:", height=150)

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        emotion, conf = predict_emotion(user_input)
        st.success(f"**Emotion:** {emotion}")
        st.info(f"**Confidence:** {conf:.2f}%")
