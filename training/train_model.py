import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

DATA_PATH = "../dataset/emotion.csv"
MODEL_PATH = "../model/emotion_lstm.h5"
CLASS_FILE = "../model/class_names.txt"

# Load dataset
data = pd.read_csv(DATA_PATH)
texts = data["text"].astype(str)
labels = data["emotion"]

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Save class names
os.makedirs("../model", exist_ok=True)
with open(CLASS_FILE, "w") as f:
    for cls in encoder.classes_:
        f.write(cls + "\n")

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=100)

# Save tokenizer
import pickle
with open("../model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, encoded_labels, test_size=0.2, random_state=42
)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.summary()

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=5,
                    batch_size=32)

# Save model
model.save(MODEL_PATH)
print("MODEL SAVED:", MODEL_PATH)
