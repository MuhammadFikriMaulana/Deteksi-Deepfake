import os
import glob
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import mediapipe as mp
import math
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=== MEMULAI EVALUASI MODEL KESELURUHAN (5 FITUR) ===")

folder_asli = 'uji_asli'
folder_palsu = 'uji_palsu'
# INI ADALAH THRESHOLD EMAS ANDA:
THRESHOLD = 0.001116

if not os.path.exists(folder_asli) or not os.path.exists(folder_palsu):
    print("GAWAT: Folder pengujian tidak ditemukan!")
    exit()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_dim = 60 * 5 # SEKARANG MENGGUNAKAN 5 FITUR
latent_dim = 16

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

latent_inputs = layers.Input(shape=(latent_dim,))
h_decoded = layers.Dense(64, activation='relu')(latent_inputs)
h_decoded = layers.Dense(128, activation='relu')(h_decoded)
outputs = layers.Dense(input_dim, activation='sigmoid')(h_decoded)
decoder = Model(latent_inputs, outputs, name='decoder')

class VAELossLayer(layers.Layer):
    def __init__(self, input_dim_val, **kwargs):
        super().__init__(**kwargs)
        self.input_dim_val = input_dim_val
    def call(self, inputs_loss):
        _, outputs_vae_val, _, _ = inputs_loss
        return outputs_vae_val

z_mean_out, z_log_var_out, z_out = encoder(inputs)
outputs_vae = decoder(z_out)
outputs_with_loss = VAELossLayer(input_dim)([inputs, outputs_vae, z_mean_out, z_log_var_out])

vae = Model(inputs, outputs_with_loss)
vae.load_weights('model_vae_bobot.weights.h5')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def hitung_jarak(p1, p2, img_w, img_h):
    x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
    x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
    return math.hypot(x1 - x2, y1 - y2)

def hitung_skor_video(jalur_video):
    cap = cv2.VideoCapture(jalur_video)
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

                mata_kiri_v1 = hitung_jarak(lm[160], lm[144], img_w, img_h)
                mata_kiri_v2 = hitung_jarak(lm[158], lm[153], img_w, img_h)
                mata_kiri_h = hitung_jarak(lm[33], lm[133], img_w, img_h)
                ear_kiri = (mata_kiri_v1 + mata_kiri_v2) / (2.0 * mata_kiri_h) if mata_kiri_h != 0 else 0
                
                mata_kanan_v1 = hitung_jarak(lm[385], lm[380], img_w, img_h)
                mata_kanan_v2 = hitung_jarak(lm[387], lm[373], img_w, img_h)
                mata_kanan_h = hitung_jarak(lm[362], lm[263], img_w, img_h)
                ear_kanan = (mata_kanan_v1 + mata_kanan_v2) / (2.0 * mata_kanan_h) if mata_kanan_h != 0 else 0
                ear_rata_rata = (ear_kiri + ear_kanan) / 2.0

                bibir_v = hitung_jarak(lm[13], lm[14], img_w, img_h)
                bibir_h = hitung_jarak(lm[78], lm[308], img_w, img_h)
                mar = bibir_v / bibir_h if bibir_h != 0 else 0

                data_pose.append([pitch, yaw, roll, ear_rata_rata, mar])
                break 
    cap.release()
    
    if len(data_pose) < 60: return None
    
    data_pose = np.array(data_pose)
    X_test = [data_pose[i : i + 60] for i in range(0, len(data_pose) - 60 + 1, 30)]
    X_test = np.array(X_test)
    X_test_flat = X_test.reshape(-1, 5)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_final = X_test_scaled.reshape(len(X_test), input_dim)

    z_mean_pred, _, _ = encoder.predict(X_test_final, verbose=0)
    reconstructed = decoder.predict(z_mean_pred, verbose=0)
    mse = np.mean(np.square(X_test_final - reconstructed), axis=1)
    return np.mean(mse)

y_true, y_pred = [], []

print("\nMemproses Video ASLI...")
for vid in glob.glob(os.path.join(folder_asli, '*.mp4')):
    print(f"  -> Evaluasi {os.path.basename(vid)} ... ", end="", flush=True)
    skor = hitung_skor_video(vid)
    if skor is not None:
        y_true.append(0)
        
        # --- LOGIKA CERDAS DETEKSI ANOMALI ---
        if skor < THRESHOLD:
            y_pred.append(1) # Skor rendah (terlalu mulus) = Deepfake
        else:
            y_pred.append(0) # Skor tinggi (getaran alami) = Asli
        # -------------------------------------
        print("Selesai!")

print("\nMemproses Video DEEPFAKE...")
for vid in glob.glob(os.path.join(folder_palsu, '*.mp4')):
    print(f"  -> Evaluasi {os.path.basename(vid)} ... ", end="", flush=True)
    skor = hitung_skor_video(vid)
    if skor is not None:
        y_true.append(1)
        
        # --- LOGIKA CERDAS DETEKSI ANOMALI ---
        if skor < THRESHOLD:
            y_pred.append(1) # Skor rendah (terlalu mulus) = Deepfake
        else:
            y_pred.append(0) # Skor tinggi (getaran alami) = Asli
        # -------------------------------------
        print("Selesai!")

akurasi = accuracy_score(y_true, y_pred)
print("\n" + "="*40)
print(f"AKURASI MODEL (5 FITUR) : {akurasi * 100:.2f}%")
print("="*40)
print("\nLaporan Lengkap:")
print(classification_report(y_true, y_pred, target_names=['Asli', 'Deepfake']))

# MENGGAMBAR CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi Asli', 'Prediksi Deepfake'], 
            yticklabels=['Kenyataan Asli', 'Kenyataan Deepfake'])
plt.title(f'Confusion Matrix (Akurasi: {akurasi*100:.2f}%)')
plt.ylabel('Label Sebenarnya')
plt.xlabel('Label Prediksi AI')
plt.tight_layout()

# TAMBAHKAN BARIS INI TEPAT SEBELUM plt.show():
plt.savefig('grafik_4_confusion_matrix.png', dpi=300)
print("GAMBAR TERSIMPAN: grafik_4_confusion_matrix.png")

plt.show()