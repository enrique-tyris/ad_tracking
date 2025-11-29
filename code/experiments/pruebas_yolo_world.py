import cv2
import json
import os
from tqdm import tqdm
from ultralytics import YOLOWorld   # <--- YOLO-World
from PIL import Image
import numpy as np


# -------------------------
# CONFIGURACIÓN DIRECTA
# -------------------------

VIDEO_INPUT = "/home/enrique/Desktop/VARIOS/garzIA/ad_tracking/data/screencap NU onderzoek (2).mp4"
ROI_FILE = "data/roi/roi_data.txt"
OUTPUT_VIDEO = "output/detecciones_yoloworld.mp4"

SHOW = False                  # Mostrar ventana opcional
DOWNSAMPLE = 6                # Procesar 1 de cada N frames

MODEL_ID = "yolov8s-world"    # YOLO-World-S (bueno y rápido)

PROMPT = ["advertisement", "banner", "mobile ad", "display ad"]
CONF_THRESHOLD = 0.30         # Confianza mínima detección


# -------------------------
# CARGAR ROI
# -------------------------

if not os.path.exists(ROI_FILE):
    raise Exception(f"No existe ROI: {ROI_FILE}")

with open(ROI_FILE, "r") as f:
    roi = json.load(f)

(x1, y1), (x2, y2) = roi
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

print(f"ROI cargado: {xmin,ymin,xmax,ymax}")


# -------------------------
# CARGAR MODELO YOLO-WORLD
# -------------------------

print(f"Cargando modelo YOLO-World: {MODEL_ID}")
model = YOLOWorld(MODEL_ID)
model.set_classes(PROMPT)      # textual grounding


# -------------------------
# PREPARAR VIDEO I/O
# -------------------------

cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise Exception("No se pudo abrir el video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"Procesando vídeo: {VIDEO_INPUT}")
print(f"Salida:           {OUTPUT_VIDEO}")
print(f"Downsample:       1 de cada {DOWNSAMPLE} frames")


# -------------------------
# LOOP PRINCIPAL
# -------------------------

for frame_idx in tqdm(range(n_frames), desc="Procesando vídeo"):
    
    ret, frame = cap.read()
    if not ret:
        break

    # Saltar frames si no toca
    if frame_idx % DOWNSAMPLE != 0:
        if SHOW:
            cv2.imshow("YOLO-World Pipeline", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        continue

    # ---- Recorte ROI ----
    roi_frame = frame[ymin:ymax, xmin:xmax]

    # ---- Inferencia YOLO-World ----
    results = model.predict(
        roi_frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    # YOLO-World retorna boxes en results[0].boxes.xyxy
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Dibujar detecciones re-proyectadas al frame original
        for (bx1, by1, bx2, by2) in boxes:
            cv2.rectangle(
                frame,
                (xmin + int(bx1), ymin + int(by1)),
                (xmin + int(bx2), ymin + int(by2)),
                (0, 0, 255),
                2
            )

    # Dibujar ROI en verde
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Guardar frame procesado
    out.write(frame)

    # Mostrar si se pidió
    if SHOW:
        cv2.imshow("YOLO-World Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
out.release()
cv2.destroyAllWindows()

print("✔ Procesamiento completado.")
print(f"✔ Vídeo guardado en: {OUTPUT_VIDEO}")
