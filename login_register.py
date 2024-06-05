import firebase_admin
from firebase_admin import credentials, auth
import streamlit as st
import requests

# Fungsi untuk menginisialisasi Firebase
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('healthcare-team5-d7a0f21f4317.json')
        firebase_admin.initialize_app(cred)

# Fungsi login
def login():
    st.sidebar.title("Login")

    # Kolom masukan untuk email dan kata sandi
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if not email or not password:
            st.error("Harap isi semua kolom")
        else:
            try:
                # URL REST API Otentikasi Firebase
                url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
                payload = {
                    'email': email,
                    'password': password,
                    'returnSecureToken': True
                }
                # Ganti 'YOUR_FIREBASE_API_KEY' dengan kunci API proyek Firebase Anda
                params = {
                    'key': 'AIzaSyAFyDQTml6uHD2LKvWwJBZNd-JVuY1OMnQ'
                }
                response = requests.post(url, params=params, json=payload)
                response_data = response.json()

                if 'error' in response_data:
                    error_message = response_data['error']['message']
                    if error_message == 'EMAIL_NOT_FOUND':
                        st.error("Email tidak ditemukan. Silakan daftar jika belum memiliki akun.")
                    elif error_message == 'INVALID_PASSWORD':
                        st.error("Password salah. Silakan coba lagi.")
                    else:
                        st.error(f"Autentikasi gagal: {error_message}")
                else:
                    st.success("Berhasil masuk!")
                    st.session_state['logged_in'] = True
                    st.experimental_rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Terjadi kesalahan jaringan: {e}")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    return (email)

# Fungsi pendaftaran
def signup():
    st.title("Daftar")

    # Kolom masukan untuk email dan kata sandi
    email = st.text_input("Email")
    password = st.text_input("Kata Sandi", type="password")

    # Tombol Daftar
    if st.button("Daftar"):
        if not email or not password:
            st.error("Harap isi semua kolom")
        else:
            try:
                # Membuat pengguna dengan email dan kata sandi menggunakan SDK Admin Firebase
                user = auth.create_user(email=email, password=password)
                st.success("Pengguna berhasil dibuat!")
                
                # Otomatis masuk pengguna setelah mendaftar
                st.session_state['page'] = login
                st.experimental_rerun()
            except auth.EmailAlreadyExistsError:
                st.error("Email sudah digunakan. Gunakan email lain.")
            except Exception as e:
                st.error(f"Error saat membuat pengguna: {e}")