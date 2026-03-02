import cv2
import mediapipe as mp
import numpy as np
import csv # Tambahan library bawaan Python untuk membuat file CSV

print("Memulai ekstraksi dan penyimpanan ke CSV...")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Nama video Anda
video_path = 'video_uji.mp4' 
cap = cv2.VideoCapture(video_path)

# PERSIAPAN MEMBUAT FILE CSV
csv_filename = 'data_pergerakan_kepala.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Membuat judul kolom di baris pertama
    writer.writerow(['Frame', 'Pitch', 'Yaw', 'Roll'])

    frame_count = 0 # Penghitung urutan frame

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video selesai diproses.")
            break
        
        frame_count += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 199, 291, 61]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])       
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360

                # MENYIMPAN ANGKA KE DALAM FILE CSV
                writer.writerow([frame_count, pitch, yaw, roll])

                # Menampilkan di layar (opsional, agar Anda tetap bisa melihat prosesnya)
                cv2.putText(image, f'Pitch: {int(pitch)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Yaw: {int(yaw)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Roll: {int(roll)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Ekstraksi Fitur ke CSV', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"SELESAI! Data berhasil disimpan ke dalam file: {csv_filename}")