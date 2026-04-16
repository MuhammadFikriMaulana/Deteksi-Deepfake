import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, Model
import pickle
import tempfile
import math
import os
from moviepy.editor import VideoFileClip 

st.set_page_config(page_title="Deepfake Detector VAE", page_icon="🕵️‍♀️", layout="centered")

latar_belakang = """
<style>
[data-testid="stAppViewContainer"] { background-color: #0E1117; background-image: radial-gradient(circle at 50% 0%, #1f2937 0%, #000000 100%); }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
h1, h2, h3, h4, h5, h6, p, label, span, small, li { color: #FFFFFF !important; }
div[data-testid="stFileUploader"] section { background-color: #1E1E1E !important; border: 2px dashed #3b82f6 !important; border-radius: 10px; }
div[data-testid="stFileUploader"] section * { color: #FFFFFF !important; }
div[data-testid="stUploadedFile"] { background-color: rgba(255,255,255, 0.1) !important; border-radius: 5px; }
div[data-testid="stUploadedFile"] * { color: #FFFFFF !important; }
div[data-testid="stButton"] button { background-color: #3b82f6 !important; border: none !important; border-radius: 8px !important; padding: 10px 24px !important; font-weight: bold !important; }
div[data-testid="stButton"] button * { color: #FFFFFF !important; }
div[data-testid="stButton"] button:hover { background-color: #2563eb !important; border: 1px solid #FFFFFF !important; }
div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
div[data-testid="stMetricLabel"] { color: #CCCCCC !important; }
</style>
"""
st.markdown(latar_belakang, unsafe_allow_html=True)

st.title("🕵️‍♀️ Sistem Deteksi Video Deepfake")
st.write("Aplikasi analisis anomali pergerakan wajah menggunakan Variational Autoencoder (VAE).")
st.markdown("---")

THRESHOLD_OPTIMAL = 0.001116
seq_length = 60 
num_features = 5
input_dim = seq_length * num_features 
latent_dim = 16 

# --- MEMBANGUN ARSITEKTUR & CACHE AI MODEL ---
@st.cache_resource
def load_ai_model():
    tf.keras.backend.clear_session()
    
    # 1. BANGUN ENCODER DAN LANGSUNG MUAT BOBOTNYA
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(128, activation='relu')(inputs)
    h = layers.Dense(64, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    class SamplingLayer(layers.Layer):
        def call(self, inputs_sampling):
            z_mean_val, z_log_var_val = inputs_sampling
            batch = tf.shape(z_mean_val)[0]
            dim = tf.shape(z_mean_val)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean_val + tf.exp(0.5 * z_log_var_val) * epsilon

    z = SamplingLayer()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.load_weights('encoder_bobot.h5') # MEMUAT FILE ENCODER

    # 2. BANGUN DECODER DAN LANGSUNG MUAT BOBOTNYA
    latent_inputs = layers.Input(shape=(latent_dim,))
    h_decoded = layers.Dense(64, activation='relu')(latent_inputs)
    h_decoded = layers.Dense(128, activation='relu')(h_decoded)
    outputs = layers.Dense(input_dim, activation='sigmoid')(h_decoded)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.load_weights('decoder_bobot.h5') # MEMUAT FILE DECODER

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return encoder, decoder, scaler

try:
    # Perhatikan, kita tidak memanggil vae_model lagi
    encoder_model, decoder_model, scaler_model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model. Error: {e}")
    st.stop()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def hitung_jarak(p1, p2, img_w, img_h):
    x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
    x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
    return math.hypot(x1 - x2, y1 - y2)

uploaded_file = st.file_uploader("Pilih video untuk diuji (Format: MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile_asli = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_asli.write(uploaded_file.read())
    nama_file_asli = tfile_asli.name
    tfile_asli.close() 

    nama_file_browser = "video_siap_putar.mp4"
    with st.spinner('Menyiapkan pemutar video untuk web...'):
        try:
            clip = VideoFileClip(nama_file_asli)
            clip.write_videofile(nama_file_browser, codec="libx264", audio=False, logger=None)
            st.video(nama_file_browser)
        except Exception:
            st.warning("Gagal mengonversi video untuk web, tapi AI tetap akan menganalisis datanya.")
    
    if st.button("🔍 Analisis Video Sekarang"):
        with st.spinner('AI sedang memindai pergerakan wajah di setiap frame. Mohon tunggu...'):
            
            cap = cv2.VideoCapture(nama_file_asli)
            data_pose = []
            
            while cap.isOpened():
                success, image = cap.read()
                if not success: break
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        img_h, img_w, _ = image.shape
                        lm = face_landmarks.landmark
                        
                        face_2d, face_3d = [], []
                        for idx in [33, 263, 1, 199, 291, 61]:
                            x, y = int(lm[idx].x * img_w), int(lm[idx].y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm[idx].z])
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)
                        cam_matrix = np.array([[1*img_w, 0, img_h/2], [0, 1*img_w, img_w/2], [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        pitch, yaw, roll = angles[0]*360, angles[1]*360, angles[2]*360

                        ear_kiri = (hitung_jarak(lm[160], lm[144], img_w, img_h) + hitung_jarak(lm[158], lm[153], img_w, img_h)) / (2.0 * hitung_jarak(lm[33], lm[133], img_w, img_h)) if hitung_jarak(lm[33], lm[133], img_w, img_h) != 0 else 0
                        ear_kanan = (hitung_jarak(lm[385], lm[380], img_w, img_h) + hitung_jarak(lm[387], lm[373], img_w, img_h)) / (2.0 * hitung_jarak(lm[362], lm[263], img_w, img_h)) if hitung_jarak(lm[362], lm[263], img_w, img_h) != 0 else 0
                        ear = (ear_kiri + ear_kanan) / 2.0
                        mar = hitung_jarak(lm[13], lm[14], img_w, img_h) / hitung_jarak(lm[78], lm[308], img_w, img_h) if hitung_jarak(lm[78], lm[308], img_w, img_h) != 0 else 0
                        
                        data_pose.append([pitch, yaw, roll, ear, mar])
                        break 
            cap.release()
            
            try:
                os.remove(nama_file_asli) 
            except Exception:
                pass 
            
            if len(data_pose) < seq_length:
                st.warning(f"⚠️ Durasi video terlalu pendek! VAE membutuhkan minimal {seq_length} frame gerakan wajah yang jelas.")
            else:
                data_pose = np.array(data_pose)
                X_test = [data_pose[i : i + seq_length] for i in range(0, len(data_pose) - seq_length + 1, 30)]
                X_test_scaled = scaler_model.transform(np.array(X_test).reshape(-1, 5)).reshape(len(X_test), input_dim)
                
                z_mean_pred, _, _ = encoder_model.predict(X_test_scaled, verbose=0)
                reconstructed = decoder_model.predict(z_mean_pred, verbose=0)
                mse_per_sequence = np.mean(np.square(X_test_scaled - reconstructed), axis=1)
                
                final_score = np.mean(mse_per_sequence)
                
                st.markdown("### 📊 Hasil Analisis")
                col1, col2 = st.columns(2)
                col1.metric("Skor Error (MSE)", f"{final_score:.6f}")
                col2.metric("Batas Toleransi (Threshold)", f"{THRESHOLD_OPTIMAL:.6f}")
                
                if final_score < THRESHOLD_OPTIMAL:
                    st.error("🚨 KEPUTUSAN: VIDEO INI TERDETEKSI SEBAGAI DEEPFAKE / PALSU!")
                    st.info("Alasan AI: Pergerakan wajah (mata & mulut) terlalu kaku atau rekonstruksi geometrinya terlampau mulus.")
                else:
                    st.success("✅ KEPUTUSAN: VIDEO INI TERDETEKSI SEBAGAI MANUSIA ASLI!")
                    st.info("Alasan AI: Terdapat fluktuasi gerakan mikro alami pada wajah.")