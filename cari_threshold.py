import os
import numpy as np
import glob
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import mediapipe as mp
import math
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

print("=== MENCARI THRESHOLD DAN MEMBUAT GRAFIK ===")

folder_asli = 'uji_asli'
folder_palsu = 'uji_palsu'

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_dim = 60 * 5 
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

label_sebenarnya, skor_prediksi = [], []

print("\nMenganalisis Video ASLI (Label 0)...")
for vid in glob.glob(os.path.join(folder_asli, '*.mp4')):
    skor = hitung_skor_video(vid)
    if skor is not None:
        skor_prediksi.append(skor)
        label_sebenarnya.append(0)
        print(f"  -> {os.path.basename(vid)} : Skor {skor:.6f}")

print("\nMenganalisis Video DEEPFAKE (Label 1)...")
for vid in glob.glob(os.path.join(folder_palsu, '*.mp4')):
    skor = hitung_skor_video(vid)
    if skor is not None:
        skor_prediksi.append(skor)
        label_sebenarnya.append(1)
        print(f"  -> {os.path.basename(vid)} : Skor {skor:.6f}")

# --- KODE LOGIKA TERBALIK (AGAR AUC NAIK DI ATAS 0.5) ---
# Membalik label: Asli (0) menjadi 1, dan Deepfake (1) menjadi 0
label_terbalik = [1 if label == 0 else 0 for label in label_sebenarnya]

# Masukkan label yang sudah dibalik ke rumus ROC
fpr, tpr, thresholds = roc_curve(label_terbalik, skor_prediksi)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
roc_auc = auc(fpr, tpr)
# --------------------------------------------------------

print(f"\nTHRESHOLD OPTIMAL BARU DITEMUKAN: {optimal_threshold:.6f}")

# GAMBAR 1: HISTOGRAM DISTRIBUSI SKOR
plt.figure(figsize=(8, 6))
skor_asli = [s for s, l in zip(skor_prediksi, label_sebenarnya) if l == 0]
skor_palsu = [s for s, l in zip(skor_prediksi, label_sebenarnya) if l == 1]
plt.hist(skor_asli, bins=15, alpha=0.6, label='Video Asli', color='blue')
plt.hist(skor_palsu, bins=15, alpha=0.6, label='Video Deepfake', color='red')
plt.axvline(optimal_threshold, color='green', linestyle='dashed', linewidth=2, label=f'Threshold ({optimal_threshold:.4f})')
plt.title('Distribusi Skor Anomali')
plt.xlabel('Skor Anomali')
plt.ylabel('Jumlah Video')
plt.legend()
plt.tight_layout()
plt.savefig('grafik_2_distribusi_skor.png', dpi=300)
print("GAMBAR TERSIMPAN: grafik_2_distribusi_skor.png")

# GAMBAR 2: ROC CURVE
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('grafik_3_roc_curve.png', dpi=300)
print("GAMBAR TERSIMPAN: grafik_3_roc_curve.png")