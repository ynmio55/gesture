# Gesture Reaction Camera

Real-time webcam gesture/pose detection written in Python using OpenCV + MediaPipe.

## Features
- Webcam detection in real time
- Hand landmarks (up to 2 hands)
- Pose landmarks
- OPEN PALM, FIST, POINT, PEACE
- Left/right hand raised detection
- FPS overlay
- Windows / Linux support

## Run
```bash
git clone https://github.com/ynmio55/gesture.git
cd gesture
python -m venv .venv
```

Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python detect.py
```

Windows:
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python detect.py
```

Press **Q** or **Esc** to quit.

> This first version performs local landmark/gesture detection; no camera frames are uploaded by the app.
