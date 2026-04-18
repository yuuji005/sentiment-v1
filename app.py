from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re

app = Flask(__name__)

# Konfigurasi Path
MODEL_PATH = 'models/sentiment_model_lstm.h5'
DATA_PATH = 'data/komentar_cleaned.csv'
ASSETS_DIR = 'assets'

# Load Model & Tokenizer
model = load_model(MODEL_PATH)
df_train = pd.read_csv(DATA_PATH)
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df_train['cleaned_comment'].astype(str).values)

def predict_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tkn = tokenizer.texts_to_sequences([text])
    tkn = pad_sequences(tkn, maxlen=max_len)
    pred = model.predict(tkn)
    labels = ['Negatif', 'Netral', 'Positif']
    return labels[np.argmax(pred)]

@app.route('/assets/<filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = ""
    if request.method == 'POST':
        user_input = request.form['komentar']
        if user_input:
            result = predict_sentiment(user_input)
    return render_template('index.html', result=result, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
app = app 
