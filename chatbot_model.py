# ==================================================
# MENTAL HEALTH SUPPORT CHATBOT - MODEL TRAINING
# ==================================================

import json
import random
import pickle
import numpy as np

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# NLTK setup (RUN ONCE, THEN COMMENT OUT IF NEEDED)
# --------------------------------------------------
# Uncomment ONLY if you get punkt errors
# nltk.download('punkt')
# nltk.download('punkt_tab')

# --------------------------------------------------
# Initialize stemmer
# --------------------------------------------------
stemmer = PorterStemmer()

# --------------------------------------------------
# Load intents JSON
# --------------------------------------------------
with open("student_mental_health.json", "r") as f:
    data = json.load(f)

print("Total intents:", len(data["intents"]))

# --------------------------------------------------
# Prepare data containers
# --------------------------------------------------
words = []
labels = []
documents = []

# --------------------------------------------------
# Tokenization + Stemming
# --------------------------------------------------
for intent in data["intents"]:
    tag = intent["tag"]
    labels.append(tag)

    for pattern in intent["patterns"]:
        tokens = word_tokenize(pattern.lower())
        stemmed_words = [stemmer.stem(word) for word in tokens if word.isalnum()]

        words.extend(stemmed_words)
        documents.append((stemmed_words, tag))

# --------------------------------------------------
# Clean vocabulary & labels
# --------------------------------------------------
words = sorted(set(words))
labels = sorted(set(labels))

print("Total unique words:", len(words))
print("Total labels:", len(labels))

# --------------------------------------------------
# Create Bag of Words
# --------------------------------------------------
X = []
y = []

for doc_words, tag in documents:
    bag = []
    for word in words:
        bag.append(1 if word in doc_words else 0)

    X.append(bag)
    y.append(tag)

X = np.array(X)
y = np.array(y)

# --------------------------------------------------
# Encode labels
# --------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --------------------------------------------------
# Train ML Model (Logistic Regression)
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

print("Model training completed")

# --------------------------------------------------
# Save trained components
# --------------------------------------------------
pickle.dump(model, open("chatbot_model.pkl", "wb"))
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(label_encoder, open("labels.pkl", "wb"))
pickle.dump(data, open("intents.pkl", "wb"))

print("Files saved successfully:")
print("✔ chatbot_model.pkl")
print("✔ words.pkl")
print("✔ labels.pkl")
print("✔ intents.pkl")

# --------------------------------------------------
# Test the model with sample input
# --------------------------------------------------
def clean_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    stemmed = [stemmer.stem(word) for word in tokens if word.isalnum()]
    return stemmed

def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Test prediction
test_sentence = "I feel very stressed and anxious"
bow = bag_of_words(test_sentence, words)
prediction = model.predict([bow])[0]
intent = label_encoder.inverse_transform([prediction])[0]

print("\nTest sentence:", test_sentence)
print("Predicted intent:", intent)
