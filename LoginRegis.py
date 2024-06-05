import firebase_admin
from firebase_admin import credentials
import streamlit as st
from streamlit_option_menu import option_menu
from login_register import login, signup
# import homepage
# import chatbot
# import deteksi_penyakit
cred = {
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
# Function to initialize Firebase
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred)
        firebase_admin.initialize_app(cred)

# Main function
def main():
    initialize_firebase()

    # Check if user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # If not logged in, show login or signup page
    if not st.session_state['logged_in']:
        login_or_signup = st.selectbox(
            "Login/Sign Up",
            ("Login", "Sign Up")
        )

        if login_or_signup == "Login":
            login()
        else:
            signup()
    else:
        # Show content page with sidebar after login
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Home", "Deteksi Penyakit", "Chatbot"],
                icons=["house", "file-medical", "chat"],
                menu_icon="cast",
                default_index=0,
            )

        # Display the selected page
        if selected == "Home":
            # homepage.app()
            st.write("Home")
        elif selected == "Deteksi Penyakit":
            st.write("Deteksi penyakit")
        elif selected == "Chatbot":
            st.write("Chatbot")

if __name__ == "__main__":
    main()