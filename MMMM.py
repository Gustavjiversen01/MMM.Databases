import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing import sequence
from tensorflow.keras.layers import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix


# Load data
df = pd.read_csv('C:/Users/gusta/OneDrive/Documents/GitHub/MMM.Databases/text.csv')

# Drop extra column
df.drop(columns='Unnamed: 0', inplace=True)

# Remove duplicates
df = df.drop_duplicates()

# Map the data for visualization
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

df['label'] = df['label'].map(emotion_map)

# Encode labels
label_mapping = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
df['label'] = df['label'].map(label_mapping)

# Verify label mapping
print("Label value counts:\n", df['label'].value_counts())
print("Unique label values:", df['label'].unique())

# Splitting the data
X = df['text']
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=60000)
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Perform padding on X_train and X_test sequences
maxlen = 79  # Use the same maxlen for both train and test sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen)

# Embedding Input Size
input_size = np.max(X_train_padded) + 1

# Define the model
model = Sequential()
model.add(Embedding(input_dim=input_size, output_dim=100, input_shape=(maxlen,)))
model.add(Bidirectional(LSTM(128)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train_padded, y_train, epochs=1, batch_size=32, validation_data=(X_test_padded, y_test), callbacks=[EarlyStopping(patience=3)])

# Define save paths
save_dir = 'C:/Users/gusta/OneDrive/Documents/GitHub/MMM.Databases/saved_model'
os.makedirs(save_dir, exist_ok=True)

# Save the model with .keras extension
model_path = os.path.join(save_dir, 'emotion_model2.keras')
if os.path.exists(model_path):
    os.remove(model_path)
model.save(model_path)

# Save the tokenizer
tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Save the label encoder
label_encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
