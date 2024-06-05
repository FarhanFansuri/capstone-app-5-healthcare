import firebase_admin
from firebase_admin import credentials, auth
import streamlit as st
import requests

cred_data = {
  "type": "service_account",
  "project_id": "healthcare-team5",
  "private_key_id": "d7a0f21f4317bdf59fec86b7e731e955dee6131e",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9iKUb3SNZ6fov\nIs7/UDSpw/hg00mQ+v95ky+s+Q6WNnoMnjrR1DvsKhgu3kRPKYJsZLMHJvzsMlF3\nGLK1hOezv/gLfj2o/oI8PzAWdrBrTcvQ+g36I/UOFmwgyqkbzJG00z5QOouZ+XIe\nbar25UpHJYlqsNm+SwwRD3/Bh8hTpuhmxxt9IvE/tWNGK+Qo7wce4u1dg0I+EQDC\n25ICqNJijgq47sd+ElFfTuvlfvBIFHsGbiMeioSQGLqFS+IejPJQoYBzRjC9O3g3\n0l1CxgzZQsVdEDcMphl9Wt4vQFVpTTLMY/iXyjHB8kc6THgOB292QS8kaspk59td\n+ZKO41s3AgMBAAECggEAHs0mm/UfqhySqndbMXW84Tv2UPH1WijYPM81J8Sf4LyA\nsC1EvNXjWRlsLa20ZzMFPY1N5OMqpk31HEg4vrQXHYicjG2bIqwdySGxtGzcdcvC\nTDde7yBJ0lzFfLzCp8Bii4Lq9QnWpxK5P4WuC4quvAo3H04XTMwQbXNZ3T8uSJ2m\nxKGBe1Y1O2nB6f0GkpBbLET44idhTR8D2E4NNtqYUSsJaT1e2EmUFVZvoka159sd\n8SrUf8CTkZePBja9vnPGgAILdBI3WdxsVgD+2yLb+YkfhRPO2SzJ3O1bNQllokns\nhIV72eu9fdBcbjwM2bO3b5xQINl+7XS2tXZGYoaRCQKBgQD3F2KBBtYG1JR8pTwF\ntw4fQlnzd76tKe2UHxuV1s2/Bdiqkcp1lju1gNH7+U3LgXOBHP0oOO6Oz/WWzixN\nE/hVL+dupRoVyss3OlHgm9LeXFFyWiuAGCvuNIHlKev47sIHLr962thGdEHp2BzT\nY0UITEKbOIM2BkQ7qCOaFlbojwKBgQDEXgM3kWizPWxqXc4ACJnGtlNfCjBTo7NL\nlPU/0vk/z6qOqzQJqQuEIphEbfr8NntXC9S3GaVyVt3dHSCqM2TXr4/vQyElglmb\nWE22nOKwn7Juffz1Cp+d7uLM547TZPtH64VCRjPs4GYZLYjmhVIvgwgK3KN0I1Ns\nH7d4150m2QKBgAD3Fmtqm2traSkTfCiJjoSQvck1roj+oO7zMChlrmgQncgMyb2r\n2I/c8PULBELHPqxCcLq3fjcQmOT96S5j5ZRvm3fTLUbog+KzV56I/UnhEnBf6cuW\nTJWhnuBGSaWvSRK+HAFe2Xls+tTvAy3QNXTS12/DyL9tRtb00S7I544BAoGAdmLT\nf7HbtIwoi6wIIP8Obr3HX7oNzq8Xzk582UDBYlY51o+i7NwMhFozALZppZrXG3CX\nitgcZeW3FrKT2ejoOByxIyeOmUWP7lb7a0gEZ0WwVmsAkV8ZQicedzh3ZD0yPXbh\nvUYD0iCrjSxlp8zi8qAFvj6tELync71faKxdI2kCgYEAxCwklk/zlypR167f2Ovx\n02zyzl91uQqxO9ZboubDmFJ0MIvEPw2FUyj8R/caK7wOtLy4V3C/gFRbmKRTrP7t\nfBT/g3xl8keKELVy+Ib1pYYGRRfIpUL+48ulpHnXQGqik9mJKxnXjRhp+eU7vYoY\nR0yoSSSBWZ0PU5yCEFuu3sk=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-bq69y@healthcare-team5.iam.gserviceaccount.com",
  "client_id": "111278925958481905704",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-bq69y%40healthcare-team5.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# Fungsi untuk menginisialisasi Firebase
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_data)
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
    st.sidebar.title("Daftar")
    initialize_firebase()
    # Kolom masukan untuk email dan kata sandi
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Kata Sandi", type="password")

    # Tombol Daftar
    if st.sidebar.button("Daftar"):
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
