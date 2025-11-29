"""
annotate_video.py

Interactive tool to manually annotate all ads in a video.
- Navigate through video frames
- Click to mark ad bounding boxes
- Save annotations to txt file
- Reload and visualize existing annotations

Output format (ads_annotations.txt):
{
    "ad_1": {
        "frame_idx": 150,
        "bbox": [x1, y1, x2, y2],  # full frame coordinates
        "timestamp": "00:05:00"
    },
    ...
}
"""

import cv2
import json
import os
from pathlib import Path


# =========================
# CONFIG
# =========================

VIDEO_INPUT = "data/input/screencap NU onderzoek (2).mp4"

# Auto-generate output directory based on video name
def get_output_dir(video_path):
    """Generate output directory based on video filename"""
    video_name = Path(video_path).stem  # Get filename without extension
    # Clean the name (replace spaces and special chars with _)
    clean_name = video_name.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = f"data/output/{clean_name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

OUTPUT_DIR = get_output_dir(VIDEO_INPUT)
ANNOTATIONS_FILE = os.path.join(OUTPUT_DIR, "ads_annotations.txt")


# =========================
# ANNOTATION STATE
# =========================

class AnnotationTool:
    def __init__(self, video_path, annotations_path):
        self.video_path = video_path
        self.annotations_path = annotations_path
        self.annotations = self.load_annotations()
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        
        # UI state
        self.p1 = None
        self.p2 = None
        self.bbox_confirmed = False  # True when bbox is ready to save (yellow)
        self.current_frame = None
        self.win_name = "Ad Annotation Tool"
        
        # Ad counter
        self.next_ad_id = self._get_next_ad_id()
    
    def load_annotations(self):
        """Load existing annotations or create empty dict"""
        if os.path.exists(self.annotations_path):
            with open(self.annotations_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_annotations(self):
        """Save annotations to file"""
        os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)
        with open(self.annotations_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"✅ Annotations saved: {len(self.annotations)} ads")
    
    def _get_next_ad_id(self):
        """Get next available ad ID"""
        if not self.annotations:
            return 1
        existing_ids = [int(k.split('_')[1]) for k in self.annotations.keys()]
        return max(existing_ids) + 1
    
    def frame_to_timestamp(self, frame_idx):
        """Convert frame index to timestamp string"""
        seconds = frame_idx / self.fps
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"
    
    def get_frame(self, frame_idx):
        """Read specific frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def draw_existing_annotations(self, frame, frame_idx):
        """Draw existing annotations at current frame only"""
        for ad_id, data in self.annotations.items():
            if data['frame_idx'] == frame_idx:
                bbox = data['bbox']
                x1, y1, x2, y2 = bbox
                # Draw in green (saved)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, ad_id, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bbox drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.p1 is None:
                self.p1 = (x, y)
                self.p2 = None
                self.bbox_confirmed = False
            elif self.p2 is None:
                self.p2 = (x, y)
                self.bbox_confirmed = True  # Bbox ready to save (yellow)
    
    def add_annotation(self):
        """Add current bbox as new annotation"""
        if not self.bbox_confirmed or self.p1 is None or self.p2 is None:
            print("⚠️  No bbox defined (click twice first)")
            return
        
        x1, y1 = self.p1
        x2, y2 = self.p2
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        
        ad_id = f"ad_{self.next_ad_id}"
        self.annotations[ad_id] = {
            "frame_idx": self.current_frame_idx,
            "bbox": [xmin, ymin, xmax, ymax],
            "timestamp": self.frame_to_timestamp(self.current_frame_idx)
        }
        
        print(f"✅ Added {ad_id} at frame {self.current_frame_idx}")
        self.next_ad_id += 1
        self.p1 = self.p2 = None
        self.bbox_confirmed = False
    
    def delete_annotation_at_frame(self):
        """Delete annotation at current frame"""
        to_delete = None
        for ad_id, data in self.annotations.items():
            if data['frame_idx'] == self.current_frame_idx:
                to_delete = ad_id
                break
        
        if to_delete:
            del self.annotations[to_delete]
            print(f"🗑️  Deleted {to_delete}")
        else:
            print("⚠️  No annotation at current frame")
    
    def run(self):
        """Main annotation loop"""
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("AD ANNOTATION TOOL")
        print("="*60)
        print("Navigation:")
        print("  d / a       : Next/Previous frame")
        print("  z / c       : Jump -30 / +30 frames")
        print("  ← / →       : Jump -100 / +100 frames")
        print("  SPACE       : Play/Pause")
        print("")
        print("Annotation:")
        print("  Click x2    : Define bbox (yellow)")
        print("  ENTER       : Save bbox (green)")
        print("  BACKSPACE   : Delete ad at current frame")
        print("")
        print("  q           : Quit and save")
        print("="*60 + "\n")
        
        playing = False
        
        while True:
            # Get current frame
            self.current_frame = self.get_frame(self.current_frame_idx)
            if self.current_frame is None:
                print("⚠️  Cannot read frame, going back")
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                continue
            
            # Create display
            display = self.current_frame.copy()
            
            # Draw existing annotations (green - saved)
            display = self.draw_existing_annotations(display, self.current_frame_idx)
            
            # Draw current bbox being created
            if self.p1 is not None:
                cv2.circle(display, self.p1, 5, (0, 255, 255), -1)  # Yellow point
            if self.p1 is not None and self.p2 is not None:
                # Yellow if not saved yet, will turn green after ENTER
                color = (0, 255, 255)  # Yellow (BGR)
                cv2.rectangle(display, self.p1, self.p2, color, 3)
                cv2.putText(display, "Press ENTER to save", (self.p1[0], self.p1[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw info overlay
            timestamp = self.frame_to_timestamp(self.current_frame_idx)
            info_text = f"Frame {self.current_frame_idx+1}/{self.total_frames} | {timestamp} | Ads: {len(self.annotations)}"
            cv2.putText(display, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if playing:
                cv2.putText(display, "PLAYING", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.win_name, display)
            
            # Handle keys (1ms = ~1000fps max, 33ms = ~30fps)
            wait_time = 1 if playing else 1
            key = cv2.waitKey(wait_time) & 0xFF
            
            # Debug: uncomment to see key codes
            # if key != 255:
            #     print(f"Key pressed: {key}")
            
            if key == ord('q') or key == 27:  # q or ESC
                self.save_annotations()
                break
            
            elif key == ord('d'):  # d - next frame
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
            
            elif key == ord('a'):  # a - previous frame
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
            
            elif key == ord('c'):  # c - jump +30 frames
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 30)
            
            elif key == ord('z'):  # z - jump -30 frames
                self.current_frame_idx = max(0, self.current_frame_idx - 30)
            
            elif key == 83:  # Right arrow - jump +100
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 100)
            
            elif key == 81:  # Left arrow - jump -100
                self.current_frame_idx = max(0, self.current_frame_idx - 100)
            
            elif key == ord(' '):  # SPACE
                playing = not playing
            
            elif key == 13 or key == 10:  # ENTER
                self.add_annotation()
            
            elif key == 127 or key == 8:  # BACKSPACE
                self.delete_annotation_at_frame()
            
            # Auto-advance if playing
            if playing:
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
                if self.current_frame_idx >= self.total_frames - 1:
                    playing = False
        
        self.cap.release()
        cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    tool = AnnotationTool(VIDEO_INPUT, ANNOTATIONS_FILE)
    tool.run()

