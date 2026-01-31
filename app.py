from flask import Flask, render_template, request
import pickle
import numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --------------------------------------------------
# Initialize Flask app
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Load trained files
# --------------------------------------------------
model = pickle.load(open("chatbot_model.pkl", "rb"))
words = pickle.load(open("words.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
intents = pickle.load(open("intents.pkl", "rb"))

stemmer = PorterStemmer()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def clean_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    stemmed = [stemmer.stem(word) for word in tokens if word.isalnum()]
    return stemmed

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_intent(sentence):
    bow = bag_of_words(sentence)
    prediction = model.predict([bow])[0]
    tag = labels.inverse_transform([prediction])[0]
    return tag

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm here to listen. Please tell me more."

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]
    intent = predict_intent(user_message)
    response = get_response(intent)
    return response

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
