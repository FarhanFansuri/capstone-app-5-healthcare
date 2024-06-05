import streamlit as st
from streamlit_option_menu import option_menu
import os
import firebase_admin
from firebase_admin import credentials
import streamlit as st
from streamlit_option_menu import option_menu
from login_register import login, signup, initialize_firebase

# ============================== Chatbot ======================================

import pickle
from tensorflow.keras.models import load_model # type: ignore
import json
import numpy as np
import random
import streamlit as st
from tensorflow import keras
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: # Set the NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)
initialize_firebase()


# Load tokenizer and classes
with open('./Chatbot_Cardiovascular/tokenizer.pkl', 'rb') as file:
    words = pickle.load(file)
with open('./Chatbot_Cardiovascular/classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# Load model
model = load_model('./chatbot_model.h5')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load JSON file with intents and responses
with open('./dataset_chatbot.json', 'r') as file:
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

# =============================================================================


# Custom CSS for styling
st.markdown(
    """
    <style>
    .navbar {
        background-color: #f9f9f9;
        padding: 10px;
    }
    .navbar .nav-link {
        font-size: 18px;
        text-align: center;
        margin: 0px;
        padding: 10px;
        color: black;
        background-color: white;
        transition: background-color 0.3s, color 0.3s;
    }
    .navbar .nav-link:hover {
        background-color: green;
        color: white;
    }
    .navbar .nav-link-selected {
        background-color: #02ab21;
        color: #fff;
    }
    .red-box {
        background-color: red;
        padding: 10px;
        color: white;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Navbar options
navbar_options = ["Home"]
if st.session_state.logged_in:
    navbar_options += ["Predict", "Chatbot"]


        
# Sidebar login/logout
if st.session_state.logged_in:
    st.sidebar.title("Keluar")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.success("Successfully logged out")
        st.experimental_rerun()
else:
    login_or_signup = st.sidebar.selectbox(
            "Login/Sign Up",
            ("Login", "Sign Up")
        )

    if login_or_signup == "Login":
        (email_now) = login()
    else:
        signup()
    # st.sidebar.title("Login")
    # username = st.sidebar.text_input("Username")
    # password = st.sidebar.text_input("Password", type="password")

    # if st.sidebar.button("Login"):
    #     if username == "user" and password == "password":  # Replace with your authentication logic
    #         st.session_state.logged_in = True
    #         st.success("Successfully logged in")
    #     else:
    #         st.error("Invalid username or password")

# Navbar navigation
selected = option_menu(
    menu_title=None,  # Title of the menu
    options=navbar_options,  # Options in the menu
    icons=["house", "graph-up", "robot"],  # Icons for the options
    menu_icon="cast",  # The icon for the menu
    # Default selected option
    orientation="horizontal",  # Orientation of the menu
    styles={
        "container": {"padding": "0!important", "background-color": "#f9f9f9"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px", "color": "black", "background-color": "white"},
        "nav-link:hover": {"background-color": "green", "color": "white"},
        "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
    },
)

# Red box above the columns
# st.markdown('<div class="red-box">Fitur Utama</div>', unsafe_allow_html=True)

# Main content based on navbar selection
if selected == "Home":
    st.title("Homepage")
    st.header("Welcome to Website Application👋😁", divider="blue")
    st.title('HartZorg')
    st.write(f'tensorflow: {tf.__version__}')
    st.write(f'streamlit: {st.__version__}')
    "Hartzorg merupakan sebuah aplikasi yang berbasis website yang berguna untuk mencari informasi mengenai Penyakit Cardiovascular atau penyakit yang berhubungan dengan jantung dan pembuluh darah."
    "Tujuan dari aplikasi website ini yaitu, untuk membantu pengguna dalam mendeteksi dan pencegahan terhadap Penyakit Cardiovascular."


    st.subheader(':books: **Fitur Utama**')

    "Dalam aplikasi website ini mempunyai dua fitur utama, yaitu **Predict** dan **Chatbot:**"
    st.markdown(":open_book: **Dalam fitur Predict atau Prediksi:** pengguna dapat memasukkan beberapa faktor dan sistem akan mengeluarkan hasil analisis.")
    st.markdown(":open_book: **Dalam fitur Chatbot:** pengguna dapat bertanya atau mencari informasi tentang tentang Penyakit Cardiovascular.")
elif selected == "Predict":
    st.title("Predict")
    st.subheader("Prediction Page")
    st.write("Here, users can input various factors to get a cardiovascular disease prediction.")

elif selected == "Chatbot":
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

