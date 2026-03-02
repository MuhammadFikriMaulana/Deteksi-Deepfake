import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import glob
import math

print("Mempersiapkan Program Ekstraksi Fitur Lengkap (Pose, EAR, MAR)...")

folder_input = 'dataset_video' 
folder_output = 'hasil_csv'    

if not os.path.exists(folder_output):
    os.makedirs(folder_output)

daftar_video = glob.glob(os.path.join(folder_input, '*.mp4'))
if len(daftar_video) == 0:
    print(f"GAWAT: Tidak ada video di '{folder_input}'.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Fungsi menghitung jarak 2 titik
def hitung_jarak(p1, p2, img_w, img_h):
    x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
    x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
    return math.hypot(x1 - x2, y1 - y2)

for jalur_video in daftar_video:
    nama_file_video = os.path.basename(jalur_video)
    jalur_csv = os.path.join(folder_output, nama_file_video.replace('.mp4', '.csv'))
    print(f"Memproses: {nama_file_video} ... ", end="")

    cap = cv2.VideoCapture(jalur_video)
    
    with open(jalur_csv, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        # KITA TAMBAHKAN EAR DAN MAR DI JUDUL KOLOM
        writer.writerow(['Frame', 'Pitch', 'Yaw', 'Roll', 'EAR', 'MAR'])
        frame_count = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            frame_count += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            img_h, img_w, _ = image.shape
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    
                    # 1. HITUNG POSE KEPALA (Pitch, Yaw, Roll)
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

                    # 2. HITUNG EAR (Eye Aspect Ratio - Kedipan Mata)
                    # Mata Kiri: 33, 160, 158, 133, 153, 144
                    mata_kiri_v1 = hitung_jarak(lm[160], lm[144], img_w, img_h)
                    mata_kiri_v2 = hitung_jarak(lm[158], lm[153], img_w, img_h)
                    mata_kiri_h = hitung_jarak(lm[33], lm[133], img_w, img_h)
                    ear_kiri = (mata_kiri_v1 + mata_kiri_v2) / (2.0 * mata_kiri_h) if mata_kiri_h != 0 else 0
                    # Mata Kanan: 362, 385, 387, 263, 373, 380
                    mata_kanan_v1 = hitung_jarak(lm[385], lm[380], img_w, img_h)
                    mata_kanan_v2 = hitung_jarak(lm[387], lm[373], img_w, img_h)
                    mata_kanan_h = hitung_jarak(lm[362], lm[263], img_w, img_h)
                    ear_kanan = (mata_kanan_v1 + mata_kanan_v2) / (2.0 * mata_kanan_h) if mata_kanan_h != 0 else 0
                    ear_rata_rata = (ear_kiri + ear_kanan) / 2.0

                    # 3. HITUNG MAR (Mouth Aspect Ratio - Gerakan Bibir)
                    bibir_v = hitung_jarak(lm[13], lm[14], img_w, img_h) # Bibir atas ke bawah (bagian dalam)
                    bibir_h = hitung_jarak(lm[78], lm[308], img_w, img_h) # Ujung bibir kiri ke kanan
                    mar = bibir_v / bibir_h if bibir_h != 0 else 0

                    # SIMPAN 5 FITUR KE CSV
                    writer.writerow([frame_count, pitch, yaw, roll, ear_rata_rata, mar])
                    break 
    cap.release()
    print(f"Selesai! Terekam {frame_count} frame.")

cv2.destroyAllWindows()
print("\n=== SEMUA VIDEO BERHASIL DIPROSES DENGAN FITUR BARU ===")