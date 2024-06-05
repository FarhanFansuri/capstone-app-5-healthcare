import pickle
from tensorflow.keras.models import load_model # type: ignore
import json
import numpy as np
import random
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load tokenizer and classes
with open('tokenizer.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# Load model
model = load_model('chatbot_model.h5')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load JSON file with intents and responses
with open('dataset_chatbot.json', 'r') as file:
    intents = json.load(file)

# Function to get response for predicted tag
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)

def predict_intent(input_data):
    input_data = nltk.word_tokenize(input_data)
    input_data = [lemmatizer.lemmatize(word.lower()) for word in input_data]
    
    bag = [0] * len(words)
    for word in input_data:
        if word in words:
            bag[words.index(word)] = 1
    
    # Predict
    result = model.predict(np.array([bag]))[0]
    threshold = 0.25
    result_index = np.argmax(result)
    if result[result_index] > threshold:
        return classes[result_index]
    else:
        return "tidak_jelas"

# Streamlit app
st.title('🤖 Tanya Cardiovascular!')
st.write("Selamat datang! Ayo tanyakan apa saja tentang kesehatan kardiovaskular.")

# Initialize chat history (similar to the provided code)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and response handling (combining best aspects)
prompt = st.chat_input("Tanyakan sesuatu")

if prompt:
  # Predict intent and get response (using your implementations)
  predicted_tag = predict_intent(prompt)
  response = get_response(predicted_tag)

  # Update chat history
  st.chat_message("user").markdown(prompt)
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
  
# Clear chat history button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()