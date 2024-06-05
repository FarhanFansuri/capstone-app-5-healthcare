import streamlit as st



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