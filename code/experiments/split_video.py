import cv2
import os

# --------------------------
# CONFIGURACIÓN
# --------------------------
VIDEO_INPUT = "data/input/screencap NU onderzoek (2).mp4"         # cambia esto
OUTPUT_DIR = "data/clips"             # subcarpeta de salida
N_CLIPS = 45                          # número de segmentos

# --------------------------
# CREAR CARPETA
# --------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# ABRIR VIDEO
# --------------------------
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise Exception(f"No se pudo abrir el video: {VIDEO_INPUT}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Resolución: {width}x{height}")
print(f"Frames totales: {total_frames}")

# Frames por clip
frames_per_clip = total_frames // N_CLIPS

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# --------------------------
# LOOP PARA CREAR CLIPS
# --------------------------
for i in range(N_CLIPS):
    start_frame = i * frames_per_clip
    end_frame   = start_frame + frames_per_clip

    clip_path = os.path.join(OUTPUT_DIR, f"clip_{i+1:02d}.mp4")
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    print(f"Creando {clip_path} ...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()

cap.release()

print("✔ División completada")
