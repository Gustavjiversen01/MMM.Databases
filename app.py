from flask import Flask, request, jsonify, render_template
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
    app.run(debug=True)
