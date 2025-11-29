import cv2
import json
import os
from tqdm import tqdm

from ultralytics.models.sam import SAM2VideoPredictor


# =========================
# CONFIG
# =========================

VIDEO_INPUT = "data/clips/clip_23.mp4"
ROI_FILE = "data/roi/roi_data.txt"
SAM_MODEL_PATH = "sam2.1_s.pt"
SAM_PROJECT = "output/experiments"
SAM_RUN_NAME = "sam2_output"
SHOW_INTERACTIVE = True


# =========================
# NUEVO: crear ruta automática para ROI_VIDEO
# =========================
def build_roi_video_path(video_input):
    """
    data/clips/clip_23.mp4 → output/experiments/cropped_clips/clip_23_roi.mp4
    """
    base = os.path.basename(video_input)       # clip_23.mp4
    name, ext = os.path.splitext(base)         # clip_23, .mp4
    out_dir = "output/experiments/cropped_clips"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{name}_roi{ext}")


ROI_VIDEO = build_roi_video_path(VIDEO_INPUT)
print("ROI_VIDEO generado automáticamente:", ROI_VIDEO)


# =========================
# UTIL: CARGAR ROI
# =========================

def load_roi(roi_path):
    if not os.path.exists(roi_path):
        raise Exception(f"No existe archivo de ROI: {roi_path}")
    with open(roi_path, "r") as f:
        pts = json.load(f)
    (x1, y1), (x2, y2) = pts
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])
    return int(xmin), int(ymin), int(xmax), int(ymax)


# =========================
# FASE 1: NAVEGAR + BBOX DENTRO DE ROI
# =========================

def interactive_select_bbox(video_path, roi_box):
    xmin, ymin, xmax, ymax = roi_box

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("No se pudo leer el primer frame.")

    state = {
        "base_frame": frame,
        "p1": None,
        "p2": None,
        "roi_confirmed": False,
        "roi_box": roi_box,
    }

    win_name = "Selector de frame + ROI"
    cv2.namedWindow(win_name)

    def mouse_cb(event, x, y, flags, param):
        s = param
        xmin_, ymin_, xmax_, ymax_ = s["roi_box"]

        if event == cv2.EVENT_LBUTTONDOWN:
            if not (xmin_ <= x <= xmax_ and ymin_ <= y <= ymax_):
                print("Click fuera de la ROI, ignorado.")
                return

            if s["p1"] is None:
                s["p1"] = (x, y)
                s["p2"] = None
                s["roi_confirmed"] = False
            elif s["p2"] is None:
                s["p2"] = (x, y)
                s["roi_confirmed"] = True

    cv2.setMouseCallback(win_name, mouse_cb, state)

    print("Controles:")
    print("  d: siguiente frame")
    print("  a: frame anterior")
    print("  click izq (2 veces dentro de la ROI): definir rectángulo")
    print("  r: reset rectángulo")
    print("  o: OK para usar rectángulo como prompt")
    print("  q: salir sin seleccionar nada")

    selected_bbox = None
    selected_frame_idx = None

    while True:
        display = state["base_frame"].copy()

        cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        if state["p1"] is not None:
            cv2.circle(display, state["p1"], 3, (0, 255, 255), -1)
        if state["p1"] is not None and state["p2"] is not None:
            cv2.rectangle(display, state["p1"], state["p2"], (0, 255, 255), 2)

        cv2.putText(display, f"Frame {current_idx+1}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('d'):
            current_idx = min(total_frames - 1, current_idx + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            if not ret: break
            state["base_frame"] = frame
            state["p1"] = state["p2"] = None

        elif key == ord('a'):
            current_idx = max(0, current_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
            ret, frame = cap.read()
            if not ret: break
            state["base_frame"] = frame
            state["p1"] = state["p2"] = None

        elif key == ord('r'):
            state["p1"] = state["p2"] = None

        elif key == ord('o'):
            if state["roi_confirmed"]:
                x1, y1 = state["p1"]
                x2, y2 = state["p2"]
                x1 = max(xmin, min(x1, xmax))
                x2 = max(xmin, min(x2, xmax))
                y1 = max(ymin, min(y1, ymax))
                y2 = max(ymin, min(y2, ymax))
                xmin_box, xmax_box = sorted([x1, x2])
                ymin_box, ymax_box = sorted([y1, y2])

                selected_bbox = [xmin_box, ymin_box, xmax_box, ymax_box]
                selected_frame_idx = current_idx
                print(f"Bounding box seleccionada:", selected_bbox)
                break
            else:
                print("Aún no hay rectángulo definido.")

    cap.release()
    cv2.destroyWindow(win_name)
    return selected_bbox, selected_frame_idx


# =========================
# FASE 2: CREAR VIDEO-ROI
# =========================

def create_roi_video_from_frame(video_path, roi_box, start_frame, out_path):
    xmin, ymin, xmax, ymax = roi_box

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(xmax - xmin)
    h = int(ymax - ymin)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for idx in tqdm(range(start_frame, total_frames), desc="Creando vídeo ROI"):
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[ymin:ymax, xmin:xmax]
        out.write(roi_frame)

    cap.release()
    out.release()


# =========================
# FASE 3: SAM2 VIDEO PREDICTOR
# =========================

def run_sam2video_on_roi_video(roi_video_path, bbox_full, roi_box, sam_model_path, project, name):
    xmin_roi, ymin_roi, xmax_roi, ymax_roi = roi_box
    x1_full, y1_full, x2_full, y2_full = bbox_full

    bbox_prompt = [[
        int(x1_full - xmin_roi),
        int(y1_full - ymin_roi),
        int(x2_full - xmin_roi),
        int(y2_full - ymin_roi)
    ]]

    overrides = dict(
        task="segment",
        mode="predict",
        imgsz=1024,
        model=sam_model_path,
        save=True,
        project=project,
        name=name,
        exist_ok=True,
    )

    predictor = SAM2VideoPredictor(overrides=overrides)

    return predictor(source=roi_video_path, bboxes=bbox_prompt)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    roi = load_roi(ROI_FILE)
    print("ROI cargada:", roi)

    bbox_full, frame_idx = interactive_select_bbox(VIDEO_INPUT, roi)
    if bbox_full is None:
        print("No se seleccionó bbox. Saliendo.")
        raise SystemExit

    create_roi_video_from_frame(VIDEO_INPUT, roi, frame_idx, ROI_VIDEO)
    print("Vídeo ROI creado en:", ROI_VIDEO)

    run_sam2video_on_roi_video(
        roi_video_path=ROI_VIDEO,
        bbox_full=bbox_full,
        roi_box=roi,
        sam_model_path=SAM_MODEL_PATH,
        project=SAM_PROJECT,
        name=SAM_RUN_NAME,
    )

    print("Proceso completo ✅")
