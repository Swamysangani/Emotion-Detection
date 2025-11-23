import pandas as pd

def load_file(path):
    texts = []
    emotions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split only ONCE at the last semicolon
            if ";" in line:
                parts = line.rsplit(";", 1)
                text = parts[0].strip()
                emotion = parts[1].strip()

                texts.append(text)
                emotions.append(emotion)

    return texts, emotions


# Load each file
train_texts, train_emotions = load_file("train.txt")
val_texts, val_emotions = load_file("val.txt")
test_texts, test_emotions = load_file("test.txt")

# Combine everything
all_texts = train_texts + val_texts + test_texts
all_emotions = train_emotions + val_emotions + test_emotions

# Create dataframe
df = pd.DataFrame({
    "text": all_texts,
    "emotion": all_emotions
})

# Save CSV
df.to_csv("emotion.csv", index=False)

print("CSV created successfully!")
print(df.head())
print("Total samples:", len(df))
print("Emotion categories:", df['emotion'].unique())
