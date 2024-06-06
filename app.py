""" from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model
model = load_model('emotion_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=79)  # Use the same maxlen used during training
    prediction = model.predict(padded_sequences)
    predicted_label = label_encoder.inverse_transform([prediction.argmax(axis=-1)[0]])
    return jsonify({'emotion': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True) """


import numpy as np
import pymysql
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load pre-trained Bi-directional LSTM model
model = load_model('/mnt/data/model.h5')  # Adjust the path to your actual model file
tokenizer = Tokenizer(num_words=5000)  # Adjust based on your tokenizer configuration

# Tokenizer parameters (adjust based on your training configuration)
max_sequence_length = 100  # Adjust according to your training configuration

# Predict emotion function
def predict_emotion(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequences)
    emotion_id = np.argmax(prediction)
    return emotion_id

# Database connection function
def get_movies_by_emotion(emotion_id):
    connection = pymysql.connect(
        host='localhost',
        user='your_username',
        password='your_password',
        database='MovieMatchMaker',
    )

    try:
        with connection.cursor() as cursor:
            sql_query = "SELECT title, reason, description, imdb_rating FROM Movies WHERE emotion_id = %s"
            cursor.execute(sql_query, (emotion_id,))
            result = cursor.fetchone()  # Fetch one movie
            return result
    finally:
        connection.close()

# Flask route to serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle form submission and return movie recommendation
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    emotion_id = predict_emotion(text)
    movie = get_movies_by_emotion(emotion_id)
    if movie:
        response = {
            'title': movie[0],
            'reason': movie[1],
            'description': movie[2],
            'imdb_rating': movie[3]
        }
    else:
        response = {
            'error': 'No movie found for the detected emotion.'
        }
    return jsonify(response)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)

