import torch
from PIL import Image
import cv2
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os
import glob
import math
from PIL import Image, ImageDraw
from typing import List, Tuple

def load_owl_model(model_id, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model = model.to(device)
    return model, processor, device


def inference(img_path, model, processor, device, prompt, score_threshold=0.1):
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    with torch.no_grad():
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
    target_sizes = torch.Tensor([[height, width]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_threshold)[0]
    return results, image


def visualize_boxes(image, boxes, scores, labels, window_name="Detecciones"):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(coord) for coord in box]
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    img_cv = resize_for_display(img_cv)
    cv2.imshow(window_name, img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_for_display(img_cv, max_width=900, max_height=900):
    h, w = img_cv.shape[:2]
    scale = 1.0
    if w > max_width or h > max_height:
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
    if scale >= 1.0:
        return img_cv
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

def compute_area(box: List[float]) -> float:
    xmin, ymin, xmax, ymax = box
    return max(0, (xmax - xmin)) * max(0, (ymax - ymin))

def filter_by_area(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    tolerance: float = 0.25
):
    if not boxes:
        return [], [], []

    areas = [compute_area(b) for b in boxes]
    median_area = np.median(areas)
    lower = median_area * (1 - tolerance)
    upper = median_area * (1 + tolerance)
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for b, s, l, a in zip(boxes, scores, labels, areas):
        if lower <= a <= upper:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            filtered_labels.append(l)
    return filtered_boxes, filtered_scores, filtered_labels

def filter_by_score(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    score_threshold: float = 0.1
):
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for b, s, l in zip(boxes, scores, labels):
        if s >= score_threshold:
            filtered_boxes.append(b)
            filtered_scores.append(s)
            filtered_labels.append(l)
    return filtered_boxes, filtered_scores, filtered_labels

def iou(boxA, boxB):
    """
    Calcula el IOU (Intersection over Union) de dos cajas.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = compute_area(boxA)
    areaB = compute_area(boxB)
    unionArea = float(areaA + areaB - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def suppress_overlaps(
    boxes,
    scores,
    labels,
    iou_thresh=0.5
):
    """
    Elimina cajas solapadas (IOU > iou_thresh), conservando la más cercana al área mediana.
    """
    if not boxes:
        return [], [], []
    areas = [compute_area(b) for b in boxes]
    median_area = np.median(areas)
    all_data = list(zip(boxes, scores, labels, areas))
    kept = []
    while all_data:
        current = all_data.pop(0)
        cb, cs, cl, ca = current
        to_remove = []
        for i, other in enumerate(all_data):
            ob, os, ol, oa = other
            if iou(cb, ob) > iou_thresh:
                dist_current = abs(ca - median_area)
                dist_other = abs(oa - median_area)
                if dist_current > dist_other:
                    current = None
                    break
                else:
                    to_remove.append(i)
        if current:
            kept.append(current)
        for i in sorted(to_remove, reverse=True):
            all_data.pop(i)
    if kept:
        filtered_boxes, filtered_scores, filtered_labels, _ = zip(*kept)
        return list(filtered_boxes), list(filtered_scores), list(filtered_labels)
    return [], [], []
