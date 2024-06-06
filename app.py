import numpy as np
import pymysql
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load pre-trained Bi-directional LSTM model
model = load_model('saved_model/emotion_model2.keras')  # Adjust the path to your actual model file

# Load the tokenizer
with open('saved_model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the label encoder
with open('saved_model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Tokenizer parameters (adjust based on your training configuration)
max_sequence_length = 79  # Use the same maxlen used during training

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
    return render_template('index.html')  # Ensure this matches the filename in the templates directory

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
