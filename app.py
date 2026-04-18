import os
import re
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Path
MODEL_PATH = 'models/sentiment_model_lstm.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'  # simpan tokenizer dari training

# Load model & tokenizer di global scope tapi dengan error handling
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    max_len = 100
    model_ready = True
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    model_ready = False

def predict_sentiment(text):
    if not model_ready:
        return "Model belum siap"
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    labels = ['Negatif', 'Netral', 'Positif']
    return labels[np.argmax(pred)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if not model_ready:
        return "Model sedang error, coba lagi nanti.", 500
    result = None
    user_input = ""
    if request.method == 'POST':
        user_input = request.form.get('komentar', '')
        if user_input:
            result = predict_sentiment(user_input)
    return render_template('index.html', result=result, user_input=user_input)

# Optional: health check
@app.route('/health')
def health():
    return "OK" if model_ready else "Model not loaded", 200 if model_ready else 500
