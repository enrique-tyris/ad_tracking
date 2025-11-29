import cv2
import random
import json
import os

# -------------------------
# CONFIG
# -------------------------
video_input = "data/input/screencap NU onderzoek (2).mp4"
save_dir = "data/roi"
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, "roi_data.txt")


# -------------------------
# Cargar frame aleatorio
# -------------------------
cap = cv2.VideoCapture(video_input)
if not cap.isOpened():
    raise Exception("No se pudo abrir el video.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rand_frame_index = random.randint(0, total_frames - 1)

cap.set(cv2.CAP_PROP_POS_FRAMES, rand_frame_index)
ret, frame = cap.read()
if not ret:
    raise Exception("No se pudo leer el frame aleatorio.")
cap.release()

original_frame = frame.copy()

# -------------------------
# Manejo de ROI y clicks
# -------------------------
roi_points = []

# Si existe archivo, cargar ROI
if os.path.exists(save_file):
    with open(save_file, "r") as f:
        roi_points = json.load(f)
    print("ROI cargado desde archivo:", roi_points)

def draw_roi(img, points):
    if len(points) == 2:
        cv2.rectangle(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global roi_points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append([x, y])
        if len(roi_points) > 2:
            roi_points = roi_points[-2:]  # Solo dos puntos

        # Dibujar
        frame = original_frame.copy()
        draw_roi(frame, roi_points)

        # Guardar cuando haya 2 puntos
        if len(roi_points) == 2:
            with open(save_file, "w") as f:
                json.dump(roi_points, f)
            print("ROI guardado:", roi_points)

# -------------------------
# Ventana y loop principal
# -------------------------
cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selector", 1000, 800)
cv2.moveWindow("ROI Selector", 200, 150)
cv2.setMouseCallback("ROI Selector", mouse_callback)

# Si ya hay ROI cargado, dibujarlo
if len(roi_points) == 2:
    draw_roi(frame, roi_points)

print("Instrucciones:")
print("- Click izquierdo: marcar esquinas del ROI (2 puntos)")
print("- Tecla 'r': borrar ROI guardado")
print("- Tecla 'q': salir")

while True:
    cv2.imshow("ROI Selector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r'):
        # Resetear
        roi_points = []
        frame = original_frame.copy()
        if os.path.exists(save_file):
            os.remove(save_file)
        print("ROI eliminado.")
        
cv2.destroyAllWindows()
