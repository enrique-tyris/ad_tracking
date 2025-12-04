"""
detect_one_ad.py

Detect and segment a single ad using SAM2 video segmentation.
- Takes one annotated ad (bbox + frame at first detection)
- Creates time window (tracks forward 17s from annotation)
- Runs SAM2 video segmentation

Output: JSON file with per-frame detections for this ad
{
    "ad_id": "ad_1",
    "start_frame": 100,
    "end_frame": 610,
    "detections": [
        {
            "frame_idx": 100,
            "bbox": [x1, y1, x2, y2],
            "mask_area": 12500,
        },
        ...
    ]
}
"""

import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from ultralytics.models.sam import SAM2VideoPredictor


# =========================
# CONFIG
# =========================

SAM_MODEL_PATH = "sam2.1_s.pt"
TIME_WINDOW_AFTER = 15.0   # seconds - track forward from annotated frame

# =========================
# OUTPUT DIRECTORY HELPER
# =========================

def get_output_dir(video_path):
    """Generate output directory based on video filename"""
    from pathlib import Path
    video_name = Path(video_path).stem
    output_dir = f"data/output/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =========================
# TIME WINDOW CALCULATION
# =========================

def calculate_time_window(frame_idx, fps, total_frames, after_sec=17.0):
    """
    Calculate start and end frames for tracking.
    Starts from the annotated frame (first detection) and tracks forward.
    
    Args:
        frame_idx: Frame where ad was annotated (first detection)
        fps: Video FPS
        total_frames: Total frames in video
        after_sec: Seconds to track forward
    
    Returns:
        (start_frame, end_frame)
    """
    start_frame = frame_idx
    frames_after = int(after_sec * fps)
    end_frame = min(total_frames - 1, frame_idx + frames_after)
    
    return start_frame, end_frame


# =========================
# VIDEO EXTRACTION
# =========================

def extract_video_segment(video_path, start_frame, end_frame, output_path):
    """
    Extract a segment of video to a temporary file.
    
    Args:
        video_path: Input video path
        start_frame: Start frame index
        end_frame: End frame index
        output_path: Output video path
    
    Returns:
        output_path
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for idx in tqdm(range(start_frame, end_frame + 1), desc="Extracting segment"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path


# =========================
# SAM2 SEGMENTATION
# =========================

def run_sam2_segmentation(video_segment_path, bbox_prompt, sam_model_path, output_dir):
    """
    Run SAM2 video segmentation on the video segment.
    Bbox prompt is applied at frame 0 of the segment.
    
    Args:
        video_segment_path: Path to video segment
        bbox_prompt: Bounding box [x1, y1, x2, y2] at frame 0
        sam_model_path: Path to SAM2 model
        output_dir: Output directory for SAM2 results
    
    Returns:
        SAM2 prediction results
    """
    bbox_list = [[bbox_prompt]]  # SAM2 expects list of lists
    
    overrides = dict(
        task="segment",
        mode="predict",
        imgsz=1024,
        model=sam_model_path,
        save=True,
        project=output_dir,
        name="sam2_temp",
        exist_ok=True,
    )
    
    predictor = SAM2VideoPredictor(overrides=overrides)
    results = predictor(source=video_segment_path, bboxes=bbox_list)
    
    return results


# =========================
# PARSE SAM2 RESULTS
# =========================

def parse_sam2_results(results, start_frame):
    """
    Parse SAM2 results into structured detections.
    
    Args:
        results: SAM2 prediction results
        start_frame: Global start frame index
    
    Returns:
        List of detection dicts
    """
    detections = []
    
    # TODO: Parse SAM2 output format
    # Extract masks, convert to bboxes, calculate confidence
    
    for idx, result in enumerate(results):
        # Placeholder - need to understand SAM2 output format
        if hasattr(result, 'masks') and result.masks is not None:
            mask = result.masks.data[0].cpu().numpy()
            
            # Convert mask to bbox
            y_indices, x_indices = np.where(mask > 0.5)
            if len(x_indices) > 0:
                x1, y1 = int(x_indices.min()), int(y_indices.min())
                x2, y2 = int(x_indices.max()), int(y_indices.max())
                
                detections.append({
                    "frame_idx": start_frame + idx,
                    "bbox": [x1, y1, x2, y2],
                    "mask_area": int(mask.sum()),
                })
    
    return detections


# =========================
# MAIN PROCESSING
# =========================

def detect_one_ad(video_path, ad_annotation, output_dir=None, sam_model_path=SAM_MODEL_PATH):
    """
    Process one ad annotation and generate detections.
    
    Args:
        video_path: Path to full video
        ad_annotation: Dict with 'frame_idx' and 'bbox'
        output_dir: Output directory for results (auto-generated if None)
        sam_model_path: Path to SAM2 model
    
    Returns:
        Dict with detection results
    """
    # Auto-generate output_dir if not provided
    if output_dir is None:
        output_dir = os.path.join(get_output_dir(video_path), "detections")
    
    ad_id = ad_annotation.get('ad_id', 'unknown')
    frame_idx = ad_annotation['frame_idx']
    bbox = ad_annotation['bbox']
    
    print(f"\n{'='*60}")
    print(f"Processing {ad_id} at frame {frame_idx}")
    print(f"{'='*60}")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate time window (starts from annotated frame, tracks forward)
    start_frame, end_frame = calculate_time_window(
        frame_idx, fps, total_frames, TIME_WINDOW_AFTER
    )
    
    print(f"Time window: frames {start_frame} to {end_frame} ({(end_frame-start_frame)/fps:.1f}s)")
    print(f"Bbox prompt will be applied at frame 0 of segment (global frame {frame_idx})")
    
    # Extract video segment starting from annotated frame
    segment_path = os.path.join(output_dir, "temp", f"{ad_id}_segment.mp4")
    extract_video_segment(video_path, start_frame, end_frame, segment_path)
    
    # Run SAM2 segmentation (bbox is at frame 0 of segment)
    print("Running SAM2 segmentation...")
    sam_output_dir = os.path.join(output_dir, "sam2_temp")
    results = run_sam2_segmentation(
        segment_path, 
        bbox,
        sam_model_path, 
        sam_output_dir
    )
    
    # Parse results
    print("Parsing SAM2 results...")
    detections = parse_sam2_results(results, start_frame)
    
    # Create output
    output = {
        "ad_id": ad_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "annotation_frame": frame_idx,
        "annotation_bbox": bbox,
        "total_detections": len(detections),
        "detections": detections
    }
    
    # Save results
    output_path = os.path.join(output_dir, f"{ad_id}_detections.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Results saved to {output_path}")
    
    # Clean up temporary files
    print("🧹 Cleaning up temporary files...")
    import shutil
    temp_dir = os.path.join(output_dir, "temp")
    sam2_temp_dir = os.path.join(output_dir, "sam2_temp")
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"  Deleted: {temp_dir}")
    
    if os.path.exists(sam2_temp_dir):
        shutil.rmtree(sam2_temp_dir)
        print(f"  Deleted: {sam2_temp_dir}")
    
    return output


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment one ad using SAM2")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--ad_id", required=True, help="Ad ID to process")
    
    args = parser.parse_args()
    
    # Auto-detect annotations file
    video_output_dir = get_output_dir(args.video)
    annotations_path = os.path.join(video_output_dir, "ads_annotations.txt")
    print(f"Using annotations: {annotations_path}")
    
    # Auto-generate output_dir
    output_dir = os.path.join(video_output_dir, "detections")
    print(f"Using output directory: {output_dir}")
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    if args.ad_id not in annotations:
        print(f"❌ Ad ID '{args.ad_id}' not found in annotations")
        exit(1)
    
    ad_annotation = annotations[args.ad_id].copy()
    ad_annotation['ad_id'] = args.ad_id
    
    # Process
    detect_one_ad(args.video, ad_annotation, output_dir, SAM_MODEL_PATH)

