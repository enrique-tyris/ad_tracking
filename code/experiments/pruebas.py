import cv2
import torch
import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

from utils_owlv2 import (
    load_owl_model,
    filter_by_score,
    filter_by_area,
    suppress_overlaps,
)


# -------------------------
# CONFIGURACIÓN DIRECTA
# -------------------------

VIDEO_INPUT = "/home/enrique/Desktop/VARIOS/garzIA/ad_tracking/data/screencap NU onderzoek (2).mp4"
ROI_FILE = "data/roi/roi_data.txt"
OUTPUT_VIDEO = "output/detecciones.mp4"

SHOW = False            # ← poner True si quieres ver la ventana
DOWNSAMPLE = 6          # procesar 1 de cada N frames

MODEL_ID = "google/owlv2-base-patch16"

PROMPT = ["detect the advertisment inside the mobile web page"]

SCORE_THRESHOLD = 0.05
TOLERANCE = 0.5
IOU_THRESH = 0.3


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
# CARGAR MODELO OWLv2
# -------------------------
model, processor, device = load_owl_model(MODEL_ID)


# -------------------------
# PREPARAR VIDEO
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

print(f"Procesando: {VIDEO_INPUT}")
print(f"Vídeo salida: {OUTPUT_VIDEO}")
print(f"Downsample: 1 de cada {DOWNSAMPLE} frames")


# -------------------------
# LOOP PRINCIPAL
# -------------------------
for frame_idx in tqdm(range(n_frames), desc="Procesando vídeo"):

    ret, frame = cap.read()
    if not ret:
        break

    # Saltar frames según downsample
    if frame_idx % DOWNSAMPLE != 0:
        out.write(frame)
        if SHOW:
            cv2.imshow("OWLv2 ROI Pipeline", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        continue

    # Extraer ROI
    roi_frame = frame[ymin:ymax, xmin:xmax]
    pil_roi = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))

    # Inference
    with torch.no_grad():
        inputs = processor(text=PROMPT, images=pil_roi, return_tensors="pt").to(device)
        outputs = model(**inputs)

    # Post-procesado
    target_sizes = torch.Tensor([[pil_roi.height, pil_roi.width]]).to(device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=SCORE_THRESHOLD
    )[0]

    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()

    # Filtros
    b, s, l = filter_by_score(boxes, scores, labels, SCORE_THRESHOLD)
    b2, s2, l2 = filter_by_area(b, s, l, tolerance=TOLERANCE)
    b3, s3, l3 = suppress_overlaps(b2, s2, l2, IOU_THRESH)

    # Dibujar detecciones proyectadas al frame original
    for box in b3:
        bx1, by1, bx2, by2 = box
        cv2.rectangle(
            frame,
            (xmin + int(bx1), ymin + int(by1)),
            (xmin + int(bx2), ymin + int(by2)),
            (0, 0, 255),
            2,
        )

    # Dibujar ROI
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Guardar frame procesado
    out.write(frame)

    if SHOW:
        cv2.imshow("OWLv2 ROI Pipeline", frame)
        if cv2.waitKey(1) == ord('q'):
            break


cap.release()
out.release()
cv2.destroyAllWindows()

print("✔ Procesamiento completado.")
print(f"✔ Vídeo guardado en: {OUTPUT_VIDEO}")
