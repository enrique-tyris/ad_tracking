"""
inference_on_queue.py

Batch processing pipeline for all annotated ads.
- Reads ads_annotations.txt
- Processes each ad with detect_one_ad.py
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

# Import our detect_one_ad module
from detect_one_ad import detect_one_ad


# =========================
# CONFIG
# =========================

SAM_MODEL_PATH = "sam2.1_s.pt"  # Default SAM model path


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
            
            result = detect_one_ad(
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
                        "name": result.get('name', ad_data.get('name')),
                    }
                else:
                    # Newly processed
                    self.results[ad_id] = {
                        "status": "completed",
                        "output_file": output_file,
                        "annotation_frame": ad_data['frame_idx'],
                        "annotation_bbox": ad_data['bbox'],
                        "name": ad_data.get('name'),
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
# PROCESS SINGLE VIDEO
# =========================

def process_single_video(video_path, sam_model_path=SAM_MODEL_PATH, skip_existing=None):
    """
    Process a single video with all its annotated ads.
    
    Args:
        video_path: Path to video
        sam_model_path: Path to SAM2 model (default: SAM_MODEL_PATH)
        skip_existing: If None, will ask user interactively. If True/False, will use that value.
    
    Returns:
        Summary dict
    """
    # Validate video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Auto-generate paths
    output_base = get_output_dir(video_path)
    annotations_path = os.path.join(output_base, "ads_annotations.txt")
    output_dir = os.path.join(output_base, "detections")
    master_output_file = os.path.join(output_base, "all_detections.txt")
    
    # Validate annotations exist
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            f"Annotations file not found: {annotations_path}\n"
            f"Please run annotate_video.py first to create annotations."
        )
    
    # Print info
    print(f"📹 Video: {video_path}")
    print(f"📄 Annotations: {annotations_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Master output: {master_output_file}\n")
    
    # Create queue
    queue = DetectionQueue(video_path, annotations_path, output_dir, sam_model_path)
    
    # Reconstruct master from existing JSONs
    ads_registry = reconstruct_master_from_jsons(queue.annotations, output_dir)
    
    # Determine skip_existing behavior
    if skip_existing is None:
        # Interactive mode - ask user
        if ads_registry:
            print(f"📂 Found {len(ads_registry)}/{len(queue.annotations)} ads already processed")
            response = input("Skip already processed ads? (y/n): ")
            skip_existing = response.lower() == 'y'
        else:
            print("📂 No existing detections found, processing all ads")
            skip_existing = False
    else:
        # Non-interactive mode - use provided value
        if ads_registry:
            action = "Skipping" if skip_existing else "Reprocessing"
            print(f"📂 Found {len(ads_registry)}/{len(queue.annotations)} ads already processed - {action}")
        else:
            print("📂 No existing detections found, processing all ads")
    
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
# BATCH MODE - Process all videos with annotations
# =========================

def find_all_annotated_videos(output_base_dir="data/output"):
    """
    Find all videos that have ads_annotations.txt in output directory.
    
    Returns:
        List of tuples: [(video_name, annotations_path, output_dir), ...]
    """
    videos_to_process = []
    
    if not os.path.exists(output_base_dir):
        return videos_to_process
    
    for video_dir in os.listdir(output_base_dir):
        video_dir_path = os.path.join(output_base_dir, video_dir)
        
        if not os.path.isdir(video_dir_path):
            continue
        
        annotations_path = os.path.join(video_dir_path, "ads_annotations.txt")
        
        if os.path.exists(annotations_path):
            # Try to find corresponding video in input directory
            # Assume video is in data/input/ with same name
            video_path = f"data/input/{video_dir}.mp4"
            
            if os.path.exists(video_path):
                output_dir = os.path.join(video_dir_path, "detections")
                master_output = os.path.join(video_dir_path, "all_detections.txt")
                
                videos_to_process.append({
                    'video_name': video_dir,
                    'video_path': video_path,
                    'annotations_path': annotations_path,
                    'output_dir': output_dir,
                    'master_output': master_output
                })
            else:
                print(f"⚠️  Found annotations but no video: {video_dir}")
    
    return videos_to_process


def process_all_videos(videos_list):
    """
    Process all videos in the list.
    
    Args:
        videos_list: List of video info dicts
    """
    # Validate we have videos to process
    if not videos_list:
        print("❌ No videos with annotations found in data/output/")
        print("   Please run annotate_video.py first to create annotations.")
        return []
    
    print("\n" + "="*60)
    print("BATCH MODE - PROCESSING ALL ANNOTATED VIDEOS")
    print("="*60)
    print(f"Found {len(videos_list)} videos with annotations")
    print("="*60 + "\n")
    
    # Ask once for all videos
    response = input("Skip already processed ads for all videos? (y/n): ")
    skip_existing = response.lower() == 'y'
    print()
    
    results = []
    
    for idx, video_info in enumerate(videos_list, 1):
        print(f"\n{'='*60}")
        print(f"VIDEO {idx}/{len(videos_list)}: {video_info['video_name']}")
        print(f"{'='*60}")
        
        try:
            summary = process_single_video(
                video_info['video_path'],
                skip_existing=skip_existing  # Pass the batch decision
            )
            
            results.append({
                'video_name': video_info['video_name'],
                'status': 'completed',
                'processed_ads': summary['processed_ads'],
                'failed_ads': summary['failed_ads']
            })
            
        except Exception as e:
            print(f"❌ Failed to process {video_info['video_name']}: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'video_name': video_info['video_name'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Print final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"✅ Completed: {completed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed videos:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['video_name']}: {r.get('error', 'Unknown error')}")
    
    print("="*60 + "\n")
    
    return results


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
 
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', '-v', type=str, help='Path to single video file')
    group.add_argument('--all', action='store_true', help='Process all videos with ads_annotations.txt in data/output/')
    args = parser.parse_args()
    
    if args.all:
        # Batch mode: process all videos with annotations
        videos_list = find_all_annotated_videos()
        process_all_videos(videos_list)
    
    else:
        # Single video mode
        process_single_video(args.video)


