"""
inference_on_queue.py

Batch processing pipeline for all annotated ads.
- Reads ads_annotations.txt
- Processes each ad with segment_one_ad.py
- Saves individual detection files
- Creates master detections file combining all results

Output: all_detections.txt
{
    "video_path": "...",
    "total_ads": 5,
    "processed_ads": 5,
    "failed_ads": [],
    "detections": {
        "ad_1": { ... },
        "ad_2": { ... },
        ...
    }
}
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import time

# Import our segment_one_ad module
from segment_one_ad import segment_one_ad


# =========================
# CONFIG
# =========================

VIDEO_INPUT = "data/screencap NU onderzoek (2).mp4"
SAM_MODEL_PATH = "sam2.1_s.pt"


# =========================
# OUTPUT DIRECTORY HELPER
# =========================

def get_output_dir(video_path):
    """Generate output directory based on video filename"""
    from pathlib import Path
    video_name = Path(video_path).stem
    clean_name = video_name.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = f"data/output/{clean_name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# Auto-generate paths based on video
OUTPUT_BASE = get_output_dir(VIDEO_INPUT)
ANNOTATIONS_FILE = os.path.join(OUTPUT_BASE, "ads_annotations.txt")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "detections")
MASTER_OUTPUT_FILE = os.path.join(OUTPUT_BASE, "all_detections.txt")


# =========================
# QUEUE PROCESSING
# =========================

class DetectionQueue:
    def __init__(self, video_path, annotations_path, output_dir, sam_model_path):
        self.video_path = video_path
        self.annotations_path = annotations_path
        self.output_dir = output_dir
        self.sam_model_path = sam_model_path
        
        # Load annotations
        self.annotations = self.load_annotations()
        
        # Track progress
        self.processed = []
        self.failed = []
        self.results = {}
    
    def load_annotations(self):
        """Load ad annotations"""
        if not os.path.exists(self.annotations_path):
            raise Exception(f"Annotations file not found: {self.annotations_path}")
        
        with open(self.annotations_path, 'r') as f:
            return json.load(f)
    
    def check_already_processed(self, ad_id):
        """Check if ad was already processed"""
        output_path = os.path.join(self.output_dir, f"{ad_id}_detections.json")
        return os.path.exists(output_path)
    
    def load_existing_detection(self, ad_id):
        """Load existing detection result"""
        output_path = os.path.join(self.output_dir, f"{ad_id}_detections.json")
        with open(output_path, 'r') as f:
            return json.load(f)
    
    def process_one_ad(self, ad_id, ad_data, skip_existing=True):
        """
        Process one ad.
        
        Args:
            ad_id: Ad identifier
            ad_data: Ad annotation data
            skip_existing: Skip if already processed
        
        Returns:
            (success, result_or_error)
        """
        # Check if already processed
        if skip_existing and self.check_already_processed(ad_id):
            print(f"⏭️  {ad_id} already processed, loading existing result")
            try:
                result = self.load_existing_detection(ad_id)
                return True, result
            except Exception as e:
                print(f"⚠️  Failed to load existing result: {e}")
                # Continue to reprocess
        
        # Process ad
        try:
            ad_annotation = ad_data.copy()
            ad_annotation['ad_id'] = ad_id
            
            result = segment_one_ad(
                self.video_path,
                ad_annotation,
                self.output_dir,
                self.sam_model_path
            )
            
            return True, result
        
        except Exception as e:
            print(f"❌ Failed to process {ad_id}: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def process_all(self, skip_existing=True, master_output_file=None):
        """
        Process all ads in the queue.
        
        Args:
            skip_existing: Skip ads that were already processed
            master_output_file: Path to save progress after each ad
        
        Returns:
            Summary dict
        """
        print("\n" + "="*60)
        print("BATCH DETECTION QUEUE")
        print("="*60)
        print(f"Video: {self.video_path}")
        print(f"Annotations: {self.annotations_path}")
        print(f"Total ads: {len(self.annotations)}")
        print(f"Output: {self.output_dir}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Process each ad
        for ad_id, ad_data in tqdm(self.annotations.items(), desc="Processing ads"):
            success, result = self.process_one_ad(ad_id, ad_data, skip_existing)
            
            if success:
                self.processed.append(ad_id)
                # Store only reference to output file, not full content
                output_file = os.path.join(self.output_dir, f"{ad_id}_detections.json")
                
                # If we loaded existing detection, get the annotation info from it
                if isinstance(result, dict) and 'annotation_frame' in result:
                    # Loaded from existing JSON
                    self.results[ad_id] = {
                        "status": "completed",
                        "output_file": output_file,
                        "annotation_frame": result.get('annotation_frame', ad_data['frame_idx']),
                        "annotation_bbox": result.get('annotation_bbox', ad_data['bbox']),
                    }
                else:
                    # Newly processed
                    self.results[ad_id] = {
                        "status": "completed",
                        "output_file": output_file,
                        "annotation_frame": ad_data['frame_idx'],
                        "annotation_bbox": ad_data['bbox'],
                    }
            else:
                self.failed.append({
                    "ad_id": ad_id,
                    "error": result
                })
            
            # Save progress after each ad
            if master_output_file:
                elapsed_time = time.time() - start_time
                summary = {
                    "video_path": self.video_path,
                    "annotations_path": self.annotations_path,
                    "output_directory": self.output_dir,
                    "total_ads": len(self.annotations),
                    "processed_ads": len(self.processed),
                    "failed_ads": len(self.failed),
                    "processing_time_seconds": elapsed_time,
                    "failed_list": self.failed,
                    "ads_registry": self.results
                }
                self.save_master_file(summary, master_output_file)
        
        elapsed_time = time.time() - start_time
        
        # Create summary
        summary = {
            "video_path": self.video_path,
            "annotations_path": self.annotations_path,
            "output_directory": self.output_dir,
            "total_ads": len(self.annotations),
            "processed_ads": len(self.processed),
            "failed_ads": len(self.failed),
            "processing_time_seconds": elapsed_time,
            "failed_list": self.failed,
            "ads_registry": self.results  # Registry with references, not full data
        }
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"✅ Processed: {len(self.processed)}/{len(self.annotations)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⏱️  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        print("="*60 + "\n")
        
        return summary
    
    def save_master_file(self, summary, output_path):
        """Save master detections file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Master detections saved to {output_path}")


# =========================
# RECONSTRUCT MASTER
# =========================

def reconstruct_master_from_jsons(annotations, output_dir):
    """
    Reconstruct master registry from existing detection JSONs.
    
    Args:
        annotations: Dict of annotations (ad_id -> annotation_data)
        output_dir: Directory where detection JSONs are stored
    
    Returns:
        Dict of ads_registry {ad_id: {status, output_file, ...}}
    """
    print("🔄 Reconstructing master from existing detections...")
    ads_registry = {}
    
    for ad_id in annotations.keys():
        json_path = os.path.join(output_dir, f"{ad_id}_detections.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    detection_data = json.load(f)
                
                ads_registry[ad_id] = {
                    "status": "completed",
                    "output_file": json_path,
                    "annotation_frame": detection_data.get('annotation_frame', 0),
                    "annotation_bbox": detection_data.get('annotation_bbox', [])
                }
            except Exception as e:
                print(f"⚠️  Could not load {ad_id}: {e}")
    
    return ads_registry


# =========================
# RESUME CAPABILITY
# =========================

def load_existing_progress(master_output_file):
    """Load existing progress from master file"""
    if os.path.exists(master_output_file):
        with open(master_output_file, 'r') as f:
            return json.load(f)
    return None


# =========================
# MAIN
# =========================

def main(video_path, annotations_path, output_dir, master_output_file, sam_model_path):
    """
    Main batch processing function.
    
    Args:
        video_path: Path to video
        annotations_path: Path to annotations file
        output_dir: Output directory for individual detections
        master_output_file: Path to master output file
        sam_model_path: Path to SAM2 model
    """
    # Create queue
    queue = DetectionQueue(video_path, annotations_path, output_dir, sam_model_path)
    
    # Reconstruct master from existing JSONs
    ads_registry = reconstruct_master_from_jsons(queue.annotations, output_dir)
    
    if ads_registry:
        print(f"📂 Found {len(ads_registry)}/{len(queue.annotations)} ads already processed")
        response = input("Skip already processed ads? (y/n): ")
        skip_existing = response.lower() == 'y'
    else:
        print("📂 No existing detections found, processing all ads")
        skip_existing = False
    
    # Process all
    summary = queue.process_all(skip_existing=skip_existing, master_output_file=master_output_file)
    
    # Save master file
    queue.save_master_file(summary, master_output_file)
    
    # Print final stats
    if summary['failed_ads'] > 0:
        print("\n⚠️  Failed ads:")
        for failed in summary['failed_list']:
            print(f"  - {failed['ad_id']}: {failed['error']}")
    
    return summary


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process all annotated ads")
    parser.add_argument("--video", default=VIDEO_INPUT, help="Path to video file")
    
    args = parser.parse_args()
    
    # Auto-generate paths
    output_base = get_output_dir(args.video)
    annotations_path = os.path.join(output_base, "ads_annotations.txt")
    output_dir = os.path.join(output_base, "detections")
    master_output = os.path.join(output_base, "all_detections.txt")
    
    print(f"Using annotations: {annotations_path}")
    print(f"Using output directory: {output_dir}")
    print(f"Using master output: {master_output}")
    
    # Run main
    main(
        args.video,
        annotations_path,
        output_dir,
        master_output,
        SAM_MODEL_PATH
    )

