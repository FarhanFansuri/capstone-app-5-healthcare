import firebase_admin
from firebase_admin import credentials
import streamlit as st
from streamlit_option_menu import option_menu
from login_register import login, signup
# import homepage
# import chatbot
# import deteksi_penyakit

# Function to initialize Firebase
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate('healthcare-team5-d7a0f21f4317.json')
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