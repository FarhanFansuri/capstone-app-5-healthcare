import pickle
import numpy as np
import streamlit as st

# Load the saved model and scaler
with open('model_prediksi_cardio.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def make_predictions(data):
    try:
        # Scale the input data
        scaled_data = scaler.transform(data)
        predictions = model.predict(scaled_data)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Title of the web app
st.title('Prediksi Penyakit Cardiovascular')

# Input columns
col1, col2, col3 = st.columns(3)

# Gender input (1 for Female, 2 for Male)
with col1:
    gender = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Laki-laki' if x == 1 else 'Perempuan')

# Age input in years
with col2:
    age = st.number_input('Umur (tahun)', min_value=0, max_value=100)

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

# Prediction result placeholder
prediksi_cardiovascular = ''

# Prediction button
if st.button('Prediksi Penyakit Cardiovascular'):
    try:
        input_data = np.array([[gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age]])
        st.write(f"Input Data: {input_data}")

        # Make predictions
        prediction = make_predictions(input_data)
        st.write(f"Raw Prediction: {prediction}")

        if prediction is not None:
            if prediction[0] == 0:
                prediksi_cardiovascular = 'Pasien terkena penyakit cardiovascular'
            else:
                prediksi_cardiovascular = 'Pasien tidak terkena penyakit cardiovascular'
            
            st.success(prediksi_cardiovascular)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
