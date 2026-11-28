# Pedestrian Detection (YOLO26m + OpenCV)

A simple, beginner-friendly project that detects pedestrians in a recorded video and shows a **Red / Yellow / Green** risk level depending on where each person is standing relative to a vehicle's path.

- Detects **only people** (the COCO `person` class)
- Draws bounding boxes with confidence scores
- Shows a danger zone in the lower-middle of the video
- Colors each pedestrian by risk level
- Shows a traffic-light indicator, pedestrian count, and overall risk
- Everything lives in a single file: `main.py`

## Project Structure

```
Pedestrian_Detection/
├── main.py
├── requirements.txt
└── README.md
```

## 1. Install

You need **Python 3.8 or newer** (3.10+ recommended).

It's a good idea to use a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install ultralytics opencv-python
```

> **Note:** Use `opencv-python`, not `opencv-python-headless`. The headless version cannot open the video window.

> **Note:** YOLO26 is a recent model family. If you get an error saying `yolo26m.pt` isn't recognized, upgrade Ultralytics with `pip install -U ultralytics`.

## 2. Get the Model (`yolo26m.pt`)

You usually don't need to do anything. When `main.py` runs `YOLO("yolo26m.pt")`, Ultralytics **downloads the file automatically** the first time (internet connection required) and reuses it afterwards.

If you'd like to download it ahead of time, run:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')"
```

If you already have the file, just place `yolo26m.pt` in the same folder as `main.py` (or set `MODEL_PATH` in `main.py` to its full path).

## 3. Run

1. Open `main.py` and set the path to your video near the top:

   ```python
   VIDEO_PATH = "path/to/your/video.mp4"
   ```

   On Windows, use forward slashes (`"C:/videos/street.mp4"`) or a raw string (`r"C:\videos\street.mp4"`).

2. Start the program:

   ```bash
   python main.py
   ```

3. A floating window opens with the processed video. Press **`Q`** to quit.

## 4. How the Risk Logic Works

### The danger zones

Two trapezoids are drawn in the lower-middle of the frame, like a road narrowing into the distance:

| Zone | Meaning | Color on screen |
|------|---------|-----------------|
| **Critical path** (inner) | The path the vehicle will drive through | Red |
| **Warning zone** (outer) | A wider area around the path | Yellow |

### Pedestrian risk

For each detected person, the program takes the **foot point**, the bottom-center of their bounding box, and checks where it falls:

| Risk | Color | Rule |
|------|-------|------|
| 🟢 **LOW** | Green | Foot point is **outside** both zones |
| 🟡 **MEDIUM** | Yellow | Foot point is inside the **warning zone**, but not the critical path (approaching the danger zone) |
| 🔴 **HIGH** | Red | Foot point is inside the **critical path** |

The foot point is used instead of the box center because feet show where a person is standing on the ground, which is what matters for a vehicle's path. It's drawn as a small dot on each box so you can see exactly what is being checked.

### Overall risk and the traffic light

The **overall risk** is simply the **highest** risk among all pedestrians in the frame:

- Any 🔴 pedestrian → overall **HIGH** (red light on)
- Otherwise, any 🟡 pedestrian → overall **MEDIUM** (yellow light on)
- Otherwise (including when nobody is detected) → overall **LOW** (green light on)

The traffic light in the top-right corner shows this overall risk. The top-left panel shows the pedestrian count and the overall risk as text.

> OpenCV can't draw emoji, so the video uses colored shapes and text instead of 🟢🟡🔴.

## 5. Easy Customization

All settings are at the top of `main.py`:

| Setting | What it does |
|---------|--------------|
| `VIDEO_PATH` | Path to your video file |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to count a detection (default `0.40`) |
| `FRAME_WIDTH` | Width every frame is resized to (default `1280`) |
| `CRITICAL_ZONE` | Corners of the inner (high-risk) zone |
| `WARNING_ZONE` | Corners of the outer (medium-risk) zone |

Zone corners are fractions of the frame: `(0, 0)` is the top-left and `(1, 1)` is the bottom-right. Adjust them until the red zone matches the lane or path in **your** video, since the right shape depends on your camera angle.

## Troubleshooting

- **"Could not open video"**: Check `VIDEO_PATH`. Try an absolute path and make sure the file exists.
- **It runs slowly**: `yolo26m.pt` is a medium-sized model, so it's much faster on a GPU. On a CPU you can switch to `yolo26s.pt` or `yolo26n.pt` by changing `MODEL_PATH`, or lower `FRAME_WIDTH`.
- **No window appears or an OpenCV GUI error**: Make sure `opencv-python` is installed (not the headless version). If both are installed, run `pip uninstall opencv-python-headless` then `pip install --force-reinstall opencv-python`.
- **Pedestrians are missed**: Lower `CONFIDENCE_THRESHOLD` (for example, `0.30`).

## Limitations

This is a simple educational demo. The risk level is based only on 2D position in the image, not real-world distance, speed, or direction of travel, so it should not be used as a real safety system.