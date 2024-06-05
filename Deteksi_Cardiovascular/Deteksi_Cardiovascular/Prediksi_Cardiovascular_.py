import streamlit as st
import joblib

# Fungsi untuk melakukan prediksi
def predict_cardiovascular(model, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years):
    # Membuat prediksi menggunakan model
    prediction = model.predict([[gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years]])
    if prediction[0] == 0:
        return "Tidak memiliki risiko kardiovaskular"
    else:
        return "Memiliki risiko kardiovaskular"

# Muat model dari file
model = joblib.load('model_prediksi_cardio.pkl')

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