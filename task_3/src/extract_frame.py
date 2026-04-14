import cv2
import os

VIDEO_PATH = "//data/video/THYZ_2026_Ornek_Veri_1.MP4"
OUTPUT_DIR = "/Users/veranur/Desktop/task_3/data/frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
orig_fps = cap.get(cv2.CAP_PROP_FPS)

print("Original FPS:", orig_fps)

target_fps = 7.5
frame_interval = orig_fps / target_fps

frame_idx = 0
saved_idx = 0
next_capture = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx >= next_capture:
        out_path = os.path.join(OUTPUT_DIR, f"frame_{saved_idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved_idx += 1
        next_capture += frame_interval

    frame_idx += 1

cap.release()
print(f"Kaydedilen frame sayısı: {saved_idx}")