# Ad Tracking Pipeline

Automatic detection and tracking of advertisements in screen recordings using SAM2.

## Requirements

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Annotate Ads

Manually mark all advertisements in your video:

```bash
python code/annotate_video.py --video "data/input/your_video.mp4"
```

**Tips:**
- Set window to fullscreen for better visualization
- Mark the first frame where each ad appears (will be tracked forward based on seconds indicated in Step 2 config)
- Follow commands shown in terminal
- After confirming a bbox, a popup will ask for a name for the ad

**Output:** `data/output/{video_name}/ads_annotations.txt`

---

### 2. Process All Ads

Automatically segment all annotated ads with SAM2:

```bash
# Single video
python code/inference_on_queue.py --video "data/input/your_video.mp4"

# Batch mode (all videos with annotations)
python code/inference_on_queue.py --all
```

**Configuration (in `detect_one_ad.py`):**
```python
SAM_MODEL_PATH = "sam2.1_s.pt"  # small model, change 's' to 'b' for more precision
TIME_WINDOW_AFTER = 15.0  # seconds to track forward
```

**Tips:**
- If interrupted, the script can resume from where it stopped
- Answer 'y' to skip already processed ads, 'n' to reprocess all

The script will:
- Find existing processed ads
- Process ads with SAM2
- Save individual JSON and update master file after each ad

**Output:**
- `data/output/{video_name}/detections/ad_*_detections.json` (individual)
- `data/output/{video_name}/all_detections.txt` (master registry)

---

### 3. Visualize Results

Generate annotated video with all detections:

```bash
python code/visualize_final_result.py --video "data/input/your_video.mp4"
```

**Output:** `data/output/{video_name}/{video_name}_annotated.mp4`

---

### 4. Modify Detections (Optional)

Interactively review and edit detections:

```bash
python code/modify_ads.py --video "data/input/your_video.mp4"
```

⚠️ **Backup `all_detections.txt` and `detections/` folder before using!**

**Features:**
- Navigate between ads
- Modify and delete boxes in each frame
- Edit names and delete entire ads
- Changes only saved when pressing `s` or confirming on exit

**Output:** Modified files in `data/output/{video_name}/detections/` and `all_detections.txt`

---

## Directory Structure

```
ad_tracking/
├── data/
│   ├── input/
│   │   └── your_video.mp4          # Input video
│   └── output/
│       └── {video_name}/
│           ├── ads_annotations.txt       # Manual annotations
│           ├── detections/
│           │   ├── ad_1_detections.json  # Per-ad detections
│           │   ├── ad_2_detections.json
│           │   └── ...
│           ├── all_detections.txt        # Master registry
│           └── {video_name}_annotated.mp4  # Final video
├── code/
│   ├── annotate_video.py           # Step 1
│   ├── detect_one_ad.py            # Step 2 (used by inference_on_queue)
│   ├── inference_on_queue.py       # Step 2
│   ├── modify_ads.py               # Step 4 (optional)
│   └── visualize_final_result.py   # Step 3
└── sam2.1_s.pt                     # SAM2 model (small, use sam2.1_b.pt for base)
```
