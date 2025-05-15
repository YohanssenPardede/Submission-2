import streamlit as st
import pickle
import numpy as np
import xgboost

# ====================
# Konfigurasi Halaman
# ====================
import streamlit as st

st.set_page_config(
    page_title='Prediksi Risiko Dropout',
    page_icon='🎓',
    layout='centered',
    initial_sidebar_state='expanded'
)

# ====================
# Muat Model
# ====================
@st.cache_resource
def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# ====================
# Tampilan Aplikasi
# ====================

# Judul & Deskripsi
st.title('🎓 Prediksi Risiko Dropout Mahasiswa')
st.markdown(
    """
    Selamat datang di aplikasi prediksi resiko dropout mahasiswa!
    Masukkan data berikut dan klik tombol prediksi untuk melihat
    persentase kemungkinan mahasiswa berhenti studi.
    """
)

# Input Data
st.header('Input Data Mahasiswa')
col1, col2 = st.columns(2)

with col1:
    approved_2nd = st.number_input(
        'Jumlah Mata Kuliah Lulus Semester 2',
        min_value=0, max_value=50, value=5, step=1,
        help='Masukkan berapa banyak mata kuliah yang berhasil lulus pada semester kedua'
    )
    fees_up_to_date = st.selectbox(
        'Pembayaran Biaya Tepat Waktu', [1, 0], index=0,
        format_func=lambda x: 'Ya' if x==1 else 'Tidak',
        help='Pilih "Ya" jika biaya kuliah mahasiswa sudah dibayar tepat waktu'
    )
    approved_1st = st.number_input(
        'Jumlah Mata Kuliah Lulus Semester 1',
        min_value=0, max_value=50, value=5, step=1,
        help='Masukkan berapa banyak mata kuliah yang berhasil lulus pada semester pertama'
    )

with col2:
    debtor = st.selectbox(
        'Ada Tunggakan Biaya?', [1, 0], index=1,
        format_func=lambda x: 'Ya' if x==1 else 'Tidak',
        help='Pilih "Ya" jika mahasiswa masih memiliki tunggakan biaya kuliah'
    )
    enrolled_2nd = st.number_input(
        'Jumlah Mata Kuliah Diambil Semester 2',
        min_value=0, max_value=50, value=6, step=1,
        help='Masukkan jumlah mata kuliah yang diambil pada semester kedua'
    )
    st.write('---')

# Tombol Prediksi
if st.button('🔍 Prediksi Risiko Dropout'):
    # Persiapan Fitur
    features = np.array([
        approved_2nd,
        fees_up_to_date,
        approved_1st,
        debtor,
        enrolled_2nd
    ]).reshape(1, -1)

    # Prediksi Model
    prob = model.predict_proba(features)[0][1]  # probabilitas dropout
    perc = prob * 100

    # Tampilkan Hasil
    st.subheader('🎯 Hasil Prediksi')
    st.metric(label='Risiko Dropout (%)', value=f"{perc:.2f}%")
    st.progress(int(perc))
    
    if perc > 50:
        st.warning('⚠️ Risiko dropout tergolong tinggi. Pertimbangkan tindakan pencegahan segera.')
    else:
        st.success('✅ Risiko dropout tergolong rendah. Pertahankan kinerja baik mahasiswa!')

# Footer
st.write('---')
st.markdown('Powered by Yohanssen Pradana Pardede')
