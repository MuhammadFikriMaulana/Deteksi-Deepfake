import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import matplotlib.pyplot as plt

# KUNCI UTAMA: Bersihkan memori agar nama tidak ganda
tf.keras.backend.clear_session()

print("=== MEMULAI PROGRAM TRAINING VAE PINTAR (5 FITUR) ===")

nama_file_data = 'data_latih_vae.npy'
if not os.path.exists(nama_file_data):
    print(f"GAWAT: File {nama_file_data} tidak ditemukan!")
    exit()

X_train = np.load(nama_file_data)
num_samples, seq_length, num_features = X_train.shape

X_train_flat = X_train.reshape(-1, num_features)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

input_dim = seq_length * num_features 
latent_dim = 16 

X_train_final = X_train_scaled.reshape(num_samples, input_dim)

# --- MEMBERI NAMA (KTP) PERMANEN PADA SETIAP LAYER ---
inputs = layers.Input(shape=(input_dim,), name='input_layer')
h = layers.Dense(128, activation='relu', name='enc_dense_1')(inputs)
h = layers.Dense(64, activation='relu', name='enc_dense_2')(h)
z_mean = layers.Dense(latent_dim, name='z_mean')(h)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)

class SamplingLayer(layers.Layer):
    def call(self, inputs_sampling):
        z_mean_val, z_log_var_val = inputs_sampling
        batch = tf.shape(z_mean_val)[0]
        dim = tf.shape(z_mean_val)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean_val + tf.exp(0.5 * z_log_var_val) * epsilon

z = SamplingLayer(name='sampling_layer')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

latent_inputs = layers.Input(shape=(latent_dim,), name='dec_input_layer')
h_decoded = layers.Dense(64, activation='relu', name='dec_dense_1')(latent_inputs)
h_decoded = layers.Dense(128, activation='relu', name='dec_dense_2')(h_decoded)
outputs = layers.Dense(input_dim, activation='sigmoid', name='dec_output_layer')(h_decoded)
decoder = Model(latent_inputs, outputs, name='decoder')

class VAELossLayer(layers.Layer):
    def __init__(self, input_dim_val, **kwargs):
        super().__init__(**kwargs)
        self.input_dim_val = input_dim_val
    def call(self, inputs_loss):
        original_inputs, outputs_vae_val, z_mean_val, z_log_var_val = inputs_loss
        reconstruction_loss = tf.keras.losses.mse(original_inputs, outputs_vae_val)
        reconstruction_loss *= self.input_dim_val
        kl_loss = 1 + z_log_var_val - tf.square(z_mean_val) - tf.exp(z_log_var_val)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return outputs_vae_val

z_mean_out, z_log_var_out, z_out = encoder(inputs)
outputs_vae = decoder(z_out)
outputs_with_loss = VAELossLayer(input_dim, name='vae_loss_layer')([inputs, outputs_vae, z_mean_out, z_log_var_out])

vae = Model(inputs, outputs_with_loss, name='vae')
vae.compile(optimizer='adam')

# --- MENGEMBALIKAN FITUR PINTAR (EARLY STOPPING) ---
early_stopping = EarlyStopping(
    monitor='val_loss',       # Memantau nilai error pada data validasi
    patience=5,               # Jika dalam 5 epoch berturut-turut error tidak turun, berhenti!
    restore_best_weights=True # OTOMATIS mengambil bobot (weights) di angka paling optimal
)

print("\n=== MULAI MELATIH AI (TRAINING) ===")
# Menambahkan callbacks early_stopping ke dalam proses training
# Epoch saya naikkan jadi 100, karena AI akan otomatis berhenti sendiri saat sudah optimal
history = vae.fit(
    X_train_final, 
    X_train_final, 
    epochs=100, 
    batch_size=8, 
    validation_split=0.1,
    callbacks=[early_stopping] 
)

# MENYIMPAN BOBOT YANG PALING OPTIMAL
vae.save_weights('model_vae_bobot.weights.h5')

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('Kurva Pembelajaran Model VAE (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Nilai Error (Loss)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('grafik_1_loss_curve.png', dpi=300)
print(f"GAMBAR TERSIMPAN: grafik_1_loss_curve.png (Berhenti di Epoch ke-{len(history.history['loss'])})")
print("\n=== TRAINING SELESAI! ===")