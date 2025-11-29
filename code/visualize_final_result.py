"""
visualize_final_result.py

Visualize all detections on video and save annotated video.
- Loads master file (all_detections.txt)
- Loads individual detection JSONs
- Draws all bboxes on each frame
- Saves annotated video

Usage:
    python code/visualize_final_result.py --video "data/input/video.mp4"
"""

import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# =========================
# CONFIG
# =========================

VIDEO_INPUT = "data/input/screencap NU onderzoek (2).mp4"

# Colors for different ads (cycling)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (255, 255, 255), # White
]


# =========================
# OUTPUT DIRECTORY HELPER
# =========================

def get_output_dir(video_path):
    """Generate output directory based on video filename"""
    video_name = Path(video_path).stem
    clean_name = video_name.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = f"data/output/{clean_name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_color_for_ad(ad_id):
    """Get consistent color for an ad_id"""
    # Extract number from ad_id
    try:
        num = int(ad_id.split('_')[1])
        return COLORS[num % len(COLORS)]
    except:
        return (255, 255, 255)


# =========================
# LOAD DETECTIONS
# =========================

def load_all_detections(output_dir):
    """
    Load master file and all individual detection JSONs.
    
    Returns:
        Dict: {frame_idx: [(ad_id, bbox), ...]}
    """
    master_file = os.path.join(output_dir, "all_detections.txt")
    
    if not os.path.exists(master_file):
        raise Exception(f"Master file not found: {master_file}")
    
    # Load master
    print(f"Loading master file: {master_file}")
    with open(master_file, 'r') as f:
        master = json.load(f)
    
    print(f"Total ads in registry: {len(master['ads_registry'])}")
    print(f"Processed: {master['processed_ads']}, Failed: {master['failed_ads']}")
    
    # Load all individual detection files
    frame_detections = defaultdict(list)
    
    for ad_id, ad_info in tqdm(master['ads_registry'].items(), desc="Loading detections"):
        if ad_info['status'] != 'completed':
            continue
        
        detection_file = ad_info['output_file']
        
        if not os.path.exists(detection_file):
            print(f"⚠️  Detection file not found: {detection_file}")
            continue
        
        # Load detection JSON
        with open(detection_file, 'r') as f:
            detections = json.load(f)
        
        # Organize by frame
        for detection in detections['detections']:
            frame_idx = detection['frame_idx']
            bbox = detection['bbox']
            frame_detections[frame_idx].append((ad_id, bbox))
    
    print(f"Total frames with detections: {len(frame_detections)}")
    
    return frame_detections


# =========================
# VISUALIZE
# =========================

def visualize_and_save(video_path, frame_detections, output_path):
    """
    Draw all detections on video and save.
    
    Args:
        video_path: Path to input video
        frame_detections: Dict {frame_idx: [(ad_id, bbox), ...]}
        output_path: Path to output video
    """
    print(f"\n{'='*60}")
    print("VISUALIZING DETECTIONS")
    print(f"{'='*60}")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections for this frame
        if frame_idx in frame_detections:
            for ad_id, bbox in frame_detections[frame_idx]:
                x1, y1, x2, y2 = bbox
                color = get_color_for_ad(ad_id)
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ad_id label
                label = ad_id
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"✅ Annotated video saved to {output_path}")
    print(f"{'='*60}\n")


# =========================
# MAIN
# =========================

def main(video_path):
    """
    Main visualization function.
    
    Args:
        video_path: Path to input video
    """
    # Get output directory
    output_dir = get_output_dir(video_path)
    
    # Load all detections
    frame_detections = load_all_detections(output_dir)
    
    # Generate output video path
    video_name = Path(video_path).stem
    output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    
    # Visualize and save
    visualize_and_save(video_path, frame_detections, output_video)
    
    return output_video


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize detections on video")
    parser.add_argument("--video", default=VIDEO_INPUT, help="Path to video file")
    
    args = parser.parse_args()
    
    output_video = main(args.video)
    print(f"\n🎬 Done! Annotated video: {output_video}")
