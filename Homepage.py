import streamlit as st
from streamlit_option_menu import option_menu
import os
import firebase_admin
from firebase_admin import credentials
import streamlit as st
from streamlit_option_menu import option_menu
from login_register import login, signup
import joblib

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
    st.header('Welcome to Website Application', divider='blue')
    st.title('HartZorg')
    "Hartzorg merupakan sebuah aplikasi yang berbasis website yang berguna untuk mencari informasi mengenai Penyakit Cardiovascular atau penyakit yang berhubungan dengan jantung dan pembuluh darah."
    "Tujuan dari aplikasi website ini yaitu, untuk membantu pengguna dalam mendeteksi dan pencegahan terhadap Penyakit Cardiovascular."


    st.subheader(':books: **Fitur Utama**')


    "Dalam aplikasi website ini mempunyai dua fitur utama, yaitu **Predict** dan **Chatbot:**"
    st.markdown(":open_book: **Dalam fitur Predict atau Prediksi:** pengguna dapat memasukkan beberapa faktor dan sistem akan mengeluarkan hasil analisis.")
    st.markdown(":open_book: **Dalam fitur Chatbot:** pengguna dapat bertanya atau mencari informasi tentang tentang Penyakit Cardiovascular.")


    st.subheader(':star: Note')


    st.markdown(':pencil: **Tekanan Darah Sistolik,** merupakan tekanan tertinggi yang dicapai saat otot jantung berkontraksi. **Normalnya** pada orang dewasa adalah 90-120 mmHg')
    st.markdown(':pencil: **Tekanan Darah Diastolik,** merupakan tekanan jantung yang terjadi saat otot jantung beristirahat atau relaksasi. Pada format penulisan, tekanan diastolik ditulis seebagai angka kedua atau pembilang. **Contohnya,** tekanan darah **120/80 mmHg** artiny tekanan darah diastoliknya adalah **80mmHg**. **Normalnya** pada orang dewasa berada dalam kisaran **60-80mmHg**. Jika angkanya berada di antara **80-89 mmHg**, masih termasuk **normal**, tetapi **kurang ideal**.')
    st.markdown(':pencil: **Kadar Kolesterol** yang normal pada orang dewasa **tidak lebih dari 200 mg/dL**')
    st.markdown(':pencil: **Kadar Gula Darah** yang termasuk normal berbeda tergantung pada usia. Berikut adalah rentang kadar gula darah normal berdasarkan usia:')


    st.markdown(":white_circle: **Anak di bawah 6 tahun:**")
    ":heavy_check_mark: Setelah puasa: >80 – 180 mg/dL"
    ":heavy_check_mark: Sebelum makan: 100 – 180 mg/dL"
    ":heavy_check_mark: 1-2 jam setelah makan: 180 mg/dL"
    ":heavy_check_mark: Sebelum tidur: 110 – 200 mg/dL"

    st.markdown(":white_circle: **Anak usia 6-12 tahun:**")
    ":heavy_check_mark: Setelah puasa: >80 – 180 mg/dL"
    ":heavy_check_mark: Sebelum makan: 90 – 180 mg/dL"
    ":heavy_check_mark: 1-2 jam setelah makan: >140 mg/dL"
    ":heavy_check_mark: Sebelum tidur: 100 – 180 mg/dL"

    st.markdown(":white_circle: **Remaja usia 13-19 tahun:**")
    ":heavy_check_mark: Setelah puasa: 70–100 mg/dL"
    ":heavy_check_mark: Sebelum tidur atau 2 jam setelah makan: <140 mg/dL"

    st.markdown(':white_circle: **Untuk orang dewasa:**')
    ":heavy_check_mark: < 8 jam sebelum makan atau saat sedang berpuasa: <100 mg/dL"
    ":heavy_check_mark: Setelah makan: 90-100 mg/dL"
elif selected == "Predict":
    # Fungsi untuk melakukan prediksi
    def predict_cardiovascular(model, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years):
        # Membuat prediksi menggunakan model
        prediction = model.predict([[gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years]])
        if prediction[0] == 0:
            return "Tidak memiliki risiko kardiovaskular"
        else:
            return "Memiliki risiko kardiovaskular"

    # Muat model dari file
    model = joblib.load('./Deteksi_Cardiovascular/Deteksi_Cardiovascular/model_prediksi_cardio2.pkl')

    # Title of the web app
    st.title('Prediksi Penyakit Cardiovascular')

    # Input columns
    col1, col2, col3 = st.columns(3)

    # Gender input (1 for Female, 2 for Male)
    with col1:
        gender = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Laki-laki' if x == 1 else 'Perempuan')

    # Age input in years
    with col2:
        age_years = st.number_input('Umur (tahun)', min_value=0, max_value=100)

    # Height input in cm
    with col3:
        height = st.number_input('Tinggi (cm)', min_value=0, max_value=300)

    # Weight input in kg
    with col1:
        weight = st.number_input('Berat Badan (kg)', min_value=0.0, max_value=300.0, format="%.2f")

    # Systolic blood pressure input
    with col2:
        ap_hi = st.number_input('Tekanan Darah Sistolik', min_value=0, max_value=300)

    # Diastolic blood pressure input
    with col3:
        ap_lo = st.number_input('Tekanan Darah Diastolik', min_value=0, max_value=200)

    # Cholesterol input (1: normal, 2: above normal, 3: well above normal)
    with col1:
        cholesterol = st.selectbox('Nilai Kolesterol', [1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else 'Above Normal' if x == 2 else 'Well Above Normal')

    # Glucose input (1: normal, 2: above normal, 3: well above normal)
    with col2:
        gluc = st.selectbox('Gula Darah', [1, 2, 3], format_func=lambda x: 'Normal' if x == 1 else 'Above Normal' if x == 2 else 'Well Above Normal')

    # Smoking input (0: No, 1: Yes)
    with col3:
        smoke = st.radio('Perokok atau tidak', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

    # Alcohol intake input (0: No, 1: Yes)
    with col1:
        alco = st.radio('Minum alkohol atau tidak', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

    # Physical activity input (0: No, 1: Yes)
    with col2:
        active = st.radio('Berohlaraga atau tidak', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

    # Prediksi ketika tombol ditekan
    if st.button('Prediksi Penyakit Cardiovascular'):
        prediction = predict_cardiovascular(model, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years)
        
        st.success(prediction)

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

