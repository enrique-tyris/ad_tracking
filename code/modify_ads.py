"""
modify_ads.py

Interactive tool to modify existing ad detections.
- Navigate through all ads sequentially
- Delete and add bounding boxes within each ad's time window
- Save modifications to detection JSON files

⚠️  IMPORTANT: BACKUP YOUR FILES BEFORE USING THIS TOOL!
    - Backup: data/output/<video_name>/all_detections.txt
    - Backup: data/output/<video_name>/detections/ folder
    This tool modifies detection files directly and changes cannot be undone!

Usage:
    python code/modify_ads.py --video "data/input/video.mp4"
"""

import cv2
import json
import os
import argparse
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import simpledialog


# =========================
# OUTPUT DIRECTORY HELPER
# =========================

def get_output_dir(video_path):
    """Generate output directory based on video filename"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video_name = Path(video_path).stem
    output_dir = f"data/output/{video_name}"
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    return output_dir


# =========================
# DETECTION EDITOR
# =========================

class DetectionEditor:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Load all ads from master file
        self.master_file = os.path.join(output_dir, "all_detections.txt")
        self.master_data = self.load_master_file()
        
        # Get list of all ad_ids sorted by annotation_frame (chronological order)
        ads_with_frames = []
        for ad_id, ad_info in self.master_data['ads_registry'].items():
            annotation_frame = ad_info.get('annotation_frame', 0)
            ads_with_frames.append((ad_id, annotation_frame))
        
        # Sort by frame number
        ads_with_frames.sort(key=lambda x: x[1])
        self.ad_ids = [ad_id for ad_id, _ in ads_with_frames]
        
        if not self.ad_ids:
            raise Exception("No ads found in master file")
        
        # Current ad index
        self.current_ad_idx = 0
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # UI state
        self.p1 = None
        self.p2 = None
        self.bbox_confirmed = False
        self.current_frame = None
        
        # Window name (will be updated when loading ads)
        self.win_name = "Modify Detections"
        
        # Load first ad
        self.load_current_ad()
    
    def load_master_file(self):
        """Load master detections file"""
        if not os.path.exists(self.master_file):
            raise FileNotFoundError(f"Master file not found: {self.master_file}")
        
        with open(self.master_file, 'r') as f:
            return json.load(f)
    
    def load_current_ad(self):
        """Load detection data for current ad"""
        self.ad_id = self.ad_ids[self.current_ad_idx]
        ad_info = self.master_data['ads_registry'][self.ad_id]
        
        self.detection_file = ad_info['output_file']
        self.detection_data = self.load_detection_file(self.detection_file)
        
        # Get name from master registry (all_detections.txt)
        self.ad_name = ad_info.get('name')
        self.start_frame = self.detection_data['start_frame']
        self.end_frame = self.detection_data['end_frame']
        
        # Create frame index for quick lookup
        self.frame_to_detection = {}
        for idx, det in enumerate(self.detection_data['detections']):
            self.frame_to_detection[det['frame_idx']] = idx
        
        # Get sorted list of frames with detections
        self.detection_frames = sorted(self.frame_to_detection.keys())
        
        # Start at first frame of ad's time window
        self.current_frame_idx = self.start_frame
        
        # Track modifications for this ad
        self.modified = False
        
        # Track delete confirmation
        self.delete_confirmation_count = 0
        
        print(f"\n{'='*60}")
        print(f"Loading ad: {self.ad_id}{': ' + self.ad_name if self.ad_name else ''}")
        print(f"Ad {self.current_ad_idx + 1}/{len(self.ad_ids)}")
        print(f"Time window: frames {self.start_frame} - {self.end_frame}")
        print(f"Detections: {len(self.detection_frames)} frames")
        print(f"{'='*60}")
    
    def load_detection_file(self, detection_file):
        """Load detection JSON file"""
        if not os.path.exists(detection_file):
            raise FileNotFoundError(f"Detection file not found: {detection_file}")
        
        with open(detection_file, 'r') as f:
            return json.load(f)
    
    def save_detection_file(self):
        """Save modified detections"""
        # Update total_detections count
        self.detection_data['total_detections'] = len(self.detection_data['detections'])
        
        with open(self.detection_file, 'w') as f:
            json.dump(self.detection_data, f, indent=2)
        
        print(f"✅ Saved modifications to {self.detection_file}")
        self.modified = False
    
    def get_frame(self, frame_idx):
        """Read specific frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def frame_to_timestamp(self, frame_idx):
        """Convert frame index to timestamp string"""
        seconds = frame_idx / self.fps
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"
    
    def get_current_detection(self):
        """Get detection at current frame (if exists)"""
        if self.current_frame_idx in self.frame_to_detection:
            det_idx = self.frame_to_detection[self.current_frame_idx]
            return self.detection_data['detections'][det_idx]
        return None
    
    def draw_current_detection(self, frame):
        """Draw existing detection at current frame"""
        detection = self.get_current_detection()
        if detection:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            # Draw in blue (existing detection)
            color = (255, 0, 0)  # Blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Current", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def delete_current_detection(self):
        """Delete detection at current frame"""
        if self.current_frame_idx not in self.frame_to_detection:
            print("⚠️  No detection at current frame")
            return
        
        det_idx = self.frame_to_detection[self.current_frame_idx]
        del self.detection_data['detections'][det_idx]
        
        # Rebuild frame index
        self.frame_to_detection = {}
        for idx, det in enumerate(self.detection_data['detections']):
            self.frame_to_detection[det['frame_idx']] = idx
        
        self.detection_frames = sorted(self.frame_to_detection.keys())
        
        print(f"🗑️  Deleted detection at frame {self.current_frame_idx}")
        self.modified = True
    
    def edit_ad_name(self):
        """Edit ad name using popup dialog"""
        root = tk.Tk()
        root.withdraw()  # Hide root window
        
        current_name = self.ad_name if self.ad_name else ""
        answer = simpledialog.askstring(
            title=f"Edit name for {self.ad_id}",
            prompt="Enter ad name (optional):",
            initialvalue=current_name
        )
        
        root.destroy()
        
        if answer is not None:  # User didn't cancel
            new_name = answer.strip() if answer.strip() else None
            
            # Update in master registry
            self.master_data['ads_registry'][self.ad_id]['name'] = new_name
            
            # Save master file
            with open(self.master_file, 'w') as f:
                json.dump(self.master_data, f, indent=2)
            
            # Update local variable
            self.ad_name = new_name
            
            # Update window title
            self.update_window_title()
            
            name_str = f"'{new_name}'" if new_name else "null"
            print(f"✅ Updated name for {self.ad_id}: {name_str}")
    
    def delete_entire_ad(self):
        """Delete entire ad (requires 3 confirmations)"""
        self.delete_confirmation_count += 1
        
        if self.delete_confirmation_count < 3:
            remaining = 3 - self.delete_confirmation_count
            print(f"⚠️  Press 'x' {remaining} more time(s) to DELETE ENTIRE AD '{self.ad_id}'")
            return False
        
        # Confirmed - delete the ad
        print(f"\n🗑️  DELETING ENTIRE AD: {self.ad_id}")
        
        # Delete the detection file
        import os
        if os.path.exists(self.detection_file):
            os.remove(self.detection_file)
            print(f"   Deleted file: {self.detection_file}")
        
        # Remove from master registry
        if self.ad_id in self.master_data['ads_registry']:
            del self.master_data['ads_registry'][self.ad_id]
            self.master_data['processed_ads'] = len(self.master_data['ads_registry'])
            print(f"   Removed from master registry")
        
        # Save updated master file
        with open(self.master_file, 'w') as f:
            json.dump(self.master_data, f, indent=2)
        print(f"   Updated master file: {self.master_file}")
        
        # Remove from ad_ids list
        self.ad_ids.remove(self.ad_id)
        
        if not self.ad_ids:
            print("\n✅ No more ads to edit. Exiting.")
            return True  # Signal to exit
        
        # Move to next ad (or previous if this was the last)
        if self.current_ad_idx >= len(self.ad_ids):
            self.current_ad_idx = len(self.ad_ids) - 1
        
        self.load_current_ad()
        self.update_window_title()
        
        print(f"✅ Ad deleted successfully. Moved to {self.ad_id}")
        return False
    
    def add_detection(self):
        """Add new detection at current frame (only within ad's time window)"""
        if not self.bbox_confirmed or self.p1 is None or self.p2 is None:
            print("⚠️  No bbox defined (click twice first)")
            return
        
        # Check if current frame is within ad's time window
        if not (self.start_frame <= self.current_frame_idx <= self.end_frame):
            print(f"⚠️  Frame {self.current_frame_idx} is outside ad's time window ({self.start_frame}-{self.end_frame})")
            return
        
        x1, y1 = self.p1
        x2, y2 = self.p2
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        
        # Check if detection already exists at this frame
        if self.current_frame_idx in self.frame_to_detection:
            # Replace existing detection
            det_idx = self.frame_to_detection[self.current_frame_idx]
            self.detection_data['detections'][det_idx]['bbox'] = [xmin, ymin, xmax, ymax]
            print(f"✅ Updated detection at frame {self.current_frame_idx}")
        else:
            # Add new detection
            new_detection = {
                "frame_idx": self.current_frame_idx,
                "bbox": [xmin, ymin, xmax, ymax],
                "mask_area": (xmax - xmin) * (ymax - ymin)
            }
            self.detection_data['detections'].append(new_detection)
            
            # Sort detections by frame_idx
            self.detection_data['detections'].sort(key=lambda x: x['frame_idx'])
            
            # Rebuild frame index
            self.frame_to_detection = {}
            for idx, det in enumerate(self.detection_data['detections']):
                self.frame_to_detection[det['frame_idx']] = idx
            
            self.detection_frames = sorted(self.frame_to_detection.keys())
            
            print(f"✅ Added detection at frame {self.current_frame_idx}")
        
        self.modified = True
        self.p1 = self.p2 = None
        self.bbox_confirmed = False
    
    def next_ad(self):
        """Go to next ad"""
        if self.modified:
            print("\n⚠️  You have unsaved modifications for this ad!")
            response = input("Save before moving to next ad? (y/n): ")
            if response.lower() == 'y':
                self.save_detection_file()
        
        if self.current_ad_idx < len(self.ad_ids) - 1:
            self.current_ad_idx += 1
            self.load_current_ad()
            self.update_window_title()
            self.delete_confirmation_count = 0  # Reset delete confirmation
        else:
            print("⚠️  Already at last ad")
    
    def prev_ad(self):
        """Go to previous ad"""
        if self.modified:
            print("\n⚠️  You have unsaved modifications for this ad!")
            response = input("Save before moving to previous ad? (y/n): ")
            if response.lower() == 'y':
                self.save_detection_file()
        
        if self.current_ad_idx > 0:
            self.current_ad_idx -= 1
            self.load_current_ad()
            self.update_window_title()
            self.delete_confirmation_count = 0  # Reset delete confirmation
        else:
            print("⚠️  Already at first ad")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bbox drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.p1 is None:
                self.p1 = (x, y)
                self.p2 = None
                self.bbox_confirmed = False
            elif self.p2 is None:
                self.p2 = (x, y)
                self.bbox_confirmed = True
    
    def update_window_title(self):
        """Update window title with current ad info"""
        name_str = f": {self.ad_name}" if self.ad_name else ""
        title = f"Modify Detections - {self.ad_id}{name_str} ({self.current_ad_idx + 1}/{len(self.ad_ids)})"
        cv2.setWindowTitle(self.win_name, title)
    
    def run(self):
        """Main editing loop"""
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        self.update_window_title()
        
        print("\n" + "="*60)
        print("⚠️  BACKUP WARNING ⚠️")
        print("="*60)
        print("This tool modifies detection files directly!")
        print("Make sure you have backed up:")
        print(f"  - {os.path.dirname(self.detection_file)}/")
        print(f"  - all_detections.txt")
        print("="*60)
        print(f"Total ads: {len(self.ad_ids)}")
        print("="*60)
        print("Navigation:")
        print("  d / a       : Next/Previous frame")
        print("  → / ←       : Next/Previous AD")
        print("")
        print("Editing:")
        print("  Click x2    : Define bbox (yellow)")
        print("  ENTER       : Save bbox (blue)")
        print("  BACKSPACE   : Delete detection at current frame")
        print("  n           : Edit ad name (popup)")
        print("  x (3 times) : DELETE ENTIRE AD (requires 3 confirmations)")
        print("")
        print("  s           : Save modifications for current ad")
        print("  q           : Quit (will prompt to save if modified)")
        print("="*60 + "\n")
        
        while True:
            # Get current frame
            self.current_frame = self.get_frame(self.current_frame_idx)
            if self.current_frame is None:
                print("⚠️  Cannot read frame, going back")
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                continue
            
            # Create display
            display = self.current_frame.copy()
            
            # Draw existing detection (blue)
            display = self.draw_current_detection(display)
            
            # Draw new bbox being created (yellow)
            if self.p1 is not None:
                cv2.circle(display, self.p1, 5, (0, 255, 255), -1)
            if self.p1 is not None and self.p2 is not None:
                color = (0, 255, 255)  # Yellow
                cv2.rectangle(display, self.p1, self.p2, color, 3)
                cv2.putText(display, "Press ENTER to save", (self.p1[0], self.p1[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw info overlay (moved down and to the right)
            x_offset = 50
            y_start = 80
            line_height = 35
            
            # Line 1: Ad ID and name
            name_display = self.ad_name if self.ad_name else "no_name"
            ad_info = f"{self.ad_id}: {name_display}"
            cv2.putText(display, ad_info, (x_offset, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Line 2: Frame info and timestamp
            timestamp = self.frame_to_timestamp(self.current_frame_idx)
            frame_info = f"Frame {self.current_frame_idx} ({self.start_frame}-{self.end_frame}) | {timestamp}"
            cv2.putText(display, frame_info, (x_offset, y_start + line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Line 3: Ad counter
            ad_counter = f"Ad {self.current_ad_idx + 1}/{len(self.ad_ids)} | Detections: {len(self.detection_frames)}"
            cv2.putText(display, ad_counter, (x_offset, y_start + line_height * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Line 4: Modified indicator or delete confirmation
            if self.delete_confirmation_count > 0:
                remaining = 3 - self.delete_confirmation_count
                delete_msg = f"DELETE AD? Press 'x' {remaining} more time(s)"
                cv2.putText(display, delete_msg, (x_offset, y_start + line_height * 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red
            elif self.modified:
                cv2.putText(display, "MODIFIED (press 's' to save)", (x_offset, y_start + line_height * 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow(self.win_name, display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                if self.modified:
                    print("\n⚠️  You have unsaved modifications!")
                    response = input("Save before quitting? (y/n): ")
                    if response.lower() == 'y':
                        self.save_detection_file()
                break
            
            elif key == ord('d'):  # d - next frame
                # Stay within ad's time window
                if self.current_frame_idx < self.end_frame:
                    self.current_frame_idx += 1
                    self.delete_confirmation_count = 0  # Reset delete confirmation
                else:
                    print(f"⚠️  At end of ad's time window (frame {self.end_frame})")
            
            elif key == ord('a'):  # a - previous frame
                # Stay within ad's time window
                if self.current_frame_idx > self.start_frame:
                    self.current_frame_idx -= 1
                    self.delete_confirmation_count = 0  # Reset delete confirmation
                else:
                    print(f"⚠️  At start of ad's time window (frame {self.start_frame})")
            
            elif key == 83:  # Right arrow - next ad
                self.next_ad()
            
            elif key == 81:  # Left arrow - previous ad
                self.prev_ad()
            
            elif key == 13 or key == 10:  # ENTER
                self.add_detection()
            
            elif key == 127 or key == 8:  # BACKSPACE
                self.delete_current_detection()
            
            elif key == ord('s'):  # s - save
                self.save_detection_file()
                self.delete_confirmation_count = 0  # Reset delete confirmation
            
            elif key == ord('n'):  # n - edit ad name
                self.edit_ad_name()
                self.delete_confirmation_count = 0  # Reset delete confirmation
            
            elif key == ord('x'):  # x - delete entire ad (requires 3 confirmations)
                should_exit = self.delete_entire_ad()
                if should_exit:
                    break
        
        self.cap.release()
        cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive tool to modify existing ad detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  IMPORTANT: BACKUP YOUR FILES BEFORE USING THIS TOOL!
    - Backup: data/output/<video_name>/all_detections.txt
    - Backup: data/output/<video_name>/detections/ folder
    
This tool modifies detection files directly and changes cannot be undone!
        """
    )
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to the input video file')
    
    args = parser.parse_args()
    
    # Get paths
    output_dir = get_output_dir(args.video)
    
    # Check if detections exist
    detections_dir = os.path.join(output_dir, "detections")
    if not os.path.exists(detections_dir):
        print(f"❌ Detections directory not found: {detections_dir}")
        print(f"   Run inference_on_queue.py first to generate detections.")
        exit(1)
    
    # Run editor
    editor = DetectionEditor(args.video, output_dir)
    editor.run()

