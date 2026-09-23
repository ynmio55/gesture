from __future__ import annotations
import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"

HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

REACTIONS = {
    "OPEN PALM": ("STOP!", (40, 40, 220)),
    "FIST": ("FIST!", (180, 70, 30)),
    "POINT": ("POINT!", (30, 160, 240)),
    "PEACE": ("PEACE :)", (180, 80, 180)),
    "LEFT HAND UP": ("HEY!", (50, 190, 70)),
    "RIGHT HAND UP": ("HEY!", (50, 190, 70)),
}

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision


def ensure_model(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 100000:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...")
    urllib.request.urlretrieve(url, path)


def pt(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def classify_hand(lm):
    fingers = [
        finger_up(lm, 8, 6),
        finger_up(lm, 12, 10),
        finger_up(lm, 16, 14),
        finger_up(lm, 20, 18),
    ]
    n = sum(fingers)
    if n == 4:
        return "OPEN PALM"
    if fingers[0] and not any(fingers[1:]):
        return "POINT"
    if fingers[0] and fingers[1] and not any(fingers[2:]):
        return "PEACE"
    if n == 0:
        return "FIST"
    return "HAND"


def draw_chain(frame, lm, connections, color):
    h, w = frame.shape[:2]
    for c in connections:
        a = c.start if hasattr(c, "start") else c[0]
        b = c.end if hasattr(c, "end") else c[1]
        cv2.line(frame, pt(lm[a], w, h), pt(lm[b], w, h), color, 2, cv2.LINE_AA)
    for p in lm:
        x, y = pt(p, w, h)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)


def load_web_meme():
    """Download/cache a real internet reaction image (CC BY 3.0, Wikimedia Commons)."""
    meme_dir = Path(__file__).resolve().parent / "assets" / "memes"
    meme_dir.mkdir(parents=True, exist_ok=True)
    path = meme_dir / "surprised_face.jpg"
    if not path.exists():
        url = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Surprised%20Face.jpg"
        print("Downloading internet reaction image ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            print(f"Could not download meme: {exc}")
            return None
    return cv2.imread(str(path))


def make_reaction_panel(label, height, meme_image=None, width=420):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    title, color = REACTIONS.get(label, ("READY", (70, 70, 70)))
    panel[:] = (18, 18, 18)

    if label != "READY" and meme_image is not None:
        img = meme_image.copy()
        ih, iw = img.shape[:2]
        max_h = max(120, height - 150)
        scale = min(width / iw, max_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (width - nw) // 2
        y = max(0, (height - 100 - nh) // 2)
        panel[y:y+nh, x:x+nw] = img
    else:
        cv2.putText(panel, "SHOW A GESTURE", (35, height // 2),
                    cv2.FONT_HERSHEY_DUPLEX, .85, (220,220,220), 2, cv2.LINE_AA)

    cv2.rectangle(panel, (0, height-100), (width, height), color, -1)
    scale = 1.15 if len(title) < 10 else .82
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, scale, 3)[0][0]
    cv2.putText(panel, title, ((width-tw)//2, height-42),
                cv2.FONT_HERSHEY_DUPLEX, scale, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(panel, "Image: Wikimedia Commons / CC BY 3.0", (20,height-14),
                cv2.FONT_HERSHEY_SIMPLEX,.38,(245,245,245),1,cv2.LINE_AA)
    return panel


def main():
    ensure_model(HAND_MODEL, HAND_URL)
    ensure_model(POSE_MODEL, POSE_URL)
    meme_image = load_web_meme()

    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Camera not found. Check camera permission or try another /dev/video device.")

    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    prev = time.perf_counter()
    start = time.perf_counter()

    with vision.HandLandmarker.create_from_options(hand_opts) as hands, \
         vision.PoseLandmarker.create_from_options(pose_opts) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)

            hr = hands.detect_for_video(image, timestamp_ms)
            pr = pose.detect_for_video(image, timestamp_ms)
            labels = []

            for lm in hr.hand_landmarks:
                draw_chain(frame, lm, hand_connections, (0, 255, 80))
                labels.append(classify_hand(lm))

            for lm in pr.pose_landmarks:
                draw_chain(frame, lm, pose_connections, (255, 180, 0))
                # MediaPipe pose: 11/12 shoulders, 15/16 wrists.
                if lm[15].visibility > 0.6 and lm[15].y < lm[11].y:
                    labels.append("LEFT HAND UP")
                if lm[16].visibility > 0.6 and lm[16].y < lm[12].y:
                    labels.append("RIGHT HAND UP")

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            cv2.rectangle(frame, (12, 12), (570, 108), (20, 20, 20), -1)
            cv2.putText(frame, f"FPS {fps:4.1f}", (28, 49),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
            text = " | ".join(dict.fromkeys(labels)) or "READY"
            cv2.putText(frame, text[:55], (28, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

            # Keep a stable reaction briefly instead of flickering when landmarks disappear.
            active = next((x for x in labels if x in REACTIONS), "READY")
            reaction = make_reaction_panel(active, frame.shape[0], meme_image)
            combined = np.hstack((frame, reaction))
            cv2.imshow("Gesture Reaction Camera", combined)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
