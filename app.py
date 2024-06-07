import numpy as np
import psycopg2
import logging
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load pre-trained Bi-directional LSTM model
model = load_model('saved_model/emotion_model2.keras')  # Adjust the path to your actual model directory
logging.info("Model loaded successfully")

# Load the tokenizer
with open('saved_model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
logging.info("Tokenizer loaded successfully")

# Load the label encoder
with open('saved_model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
logging.info("Label encoder loaded successfully")

# Tokenizer parameters (adjust based on your training configuration)
max_sequence_length = 79  # Use the same maxlen used during training

# Predict emotion function
def predict_emotion(sentence):
    logging.debug(f"Predicting emotion for: {sentence}")
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequences)
    emotion_id = np.argmax(prediction)
    logging.debug(f"Predicted emotion ID: {emotion_id}")
    return int(emotion_id) + 1  # Convert to native Python integer and adjust for database ID

# Database connection function
def get_movies_by_emotion(emotion_id):
    logging.debug(f"Fetching movies for emotion ID: {emotion_id}")
    connection = psycopg2.connect(
        host='localhost',
        user='postgres',  # Ensure you have the correct credentials
        password='Dqh75dba',  # Ensure you have the correct credentials
        database='moviematchmaker',
    )

    try:
        with connection.cursor() as cursor:
            sql_query = "SELECT title, reason, description, imdb_rating FROM Movies WHERE emotion_id = %s"
            cursor.execute(sql_query, (emotion_id,))
            result = cursor.fetchall()  # Fetch all movies
            logging.debug(f"Query result: {result}")
            if result:
                return random.choice(result)  # Randomly pick one movie
            else:
                return None
    except Exception as e:
        logging.error(f"Database query failed: {e}")
    finally:
        connection.close()

# Search movies function
def search_movies(query):
    logging.debug(f"Searching movies with query: {query}")
    connection = psycopg2.connect(
        host='localhost',
        user='postgres',
        password='Dqh75dba',
        database='moviematchmaker',
    )

    try:
        with connection.cursor() as cursor:
            sql_query = "SELECT title, reason, description, imdb_rating FROM Movies WHERE title ~* %s OR description ~* %s"
            cursor.execute(sql_query, (query, query))
            results = cursor.fetchall()
            logging.debug(f"Search results: {results}")
            return results
    except Exception as e:
        logging.error(f"Database query failed: {e}")
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
    logging.info(f"Received text: {text}")
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

# Flask route to handle search functionality
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    logging.info(f"Search query: {query}")
    results = search_movies(query)
    return render_template('search_results.html', query=query, results=results)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)

