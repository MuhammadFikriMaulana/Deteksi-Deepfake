import pandas as pd
import numpy as np
import os
import glob

print("Memulai proses Sliding Window...")

# Pengaturan
folder_input = 'hasil_csv'
SEQ_LENGTH = 60  # Panjang potongan (60 frame = sekitar 2 detik)
STEP_SIZE = 30   # Pergeseran window (overlap 50% untuk memperbanyak data)

daftar_csv = glob.glob(os.path.join(folder_input, '*.csv'))

if len(daftar_csv) == 0:
    print(f"GAWAT: Tidak ada file CSV di folder '{folder_input}'.")
    exit()

all_sequences = []

# Memproses setiap file CSV
for file_csv in daftar_csv:
    # Membaca file CSV
    df = pd.read_csv(file_csv)
    
    # Kita HANYA mengambil kolom angka biomekaniknya (Pitch, Yaw, Roll)
    # Kolom 'Frame' (urutan) dibuang karena tidak dibutuhkan 
    data = df[['Pitch', 'Yaw', 'Roll', 'EAR', 'MAR']].values
    
    # Pemotongan Sliding Window
    for i in range(0, len(data) - SEQ_LENGTH + 1, STEP_SIZE):
        sequence = data[i : i + SEQ_LENGTH]
        all_sequences.append(sequence)

# Mengubah kumpulan list menjadi Numpy Array (Matriks 3D)
X_train = np.array(all_sequences)

print(f"\nSelesai memproses {len(daftar_csv)} file CSV.")
print(f"Total potongan sekuens yang dihasilkan: {X_train.shape[0]} sekuens")
print(f"Bentuk (Shape) Data Latih: {X_train.shape} -> (Jumlah_Sekuens, Panjang_Frame, Fitur_Pose)")

# Menyimpan data menjadi file .npy (Siap masuk ke VAE)
nama_file_output = 'data_latih_vae.npy'
np.save(nama_file_output, X_train)
print(f"\nData berhasil disimpan sebagai '{nama_file_output}'.")