from __future__ import annotations
import math
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"

HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

REACTIONS = {\n    "WAVE": ("WAVE", (50, 190, 70)),\n    "BOTH HANDS UP": ("HANDS UP", (30, 160, 240)),\n    "SHOCKED": ("SHOCKED", (40, 40, 220)),
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


MEME_URLS = {\n    "WAVE": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Waving_hand.jpg",\n    "BOTH HANDS UP": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Surprised%20Person.png",\n    "SHOCKED": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Openmouth.jpg",
    "OPEN PALM": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Baby%20surprised%20face.jpg",
    "FIST": "https://commons.wikimedia.org/wiki/Special:Redirect/file/SketchOfAConfusedHuman.jpg",
    "POINT": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Surprised%20Person.png",
    "PEACE": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Surprising%20face%20of%20manga%202022-08-30.png",
    "LEFT HAND UP": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Openmouth.jpg",
    "RIGHT HAND UP": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Openmouth.jpg",
}


def load_web_memes():
    """Download/cache real reaction images. A failed image never disables the others."""
    meme_dir = Path(__file__).resolve().parent / "assets" / "memes"
    meme_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, url in MEME_URLS.items():
        ext = ".png" if url.lower().endswith(".png") else ".jpg"
        path = meme_dir / (label.lower().replace(" ", "_") + ext)
        try:
            if not path.exists() or path.stat().st_size < 2000:
                print(f"Downloading reaction: {label} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "gesture-reaction-camera/1.0"})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(req, timeout=20) as response, open(path, "wb") as out:
                            out.write(response.read())
                        break
                    except urllib.error.HTTPError as http_exc:
                        if http_exc.code != 429 or attempt == 2:
                            raise
                        wait = 2.0 + attempt * 3.0 + random.random()
                        print(f"Rate limited; retrying {label} in {wait:.1f}s ...")
                        time.sleep(wait)
                time.sleep(1.2 + random.random() * 0.8)
            img = cv2.imread(str(path))
            if img is not None:
                result[label] = img
            else:
                print(f"Warning: invalid image for {label}: {path}")
        except Exception as exc:
            print(f"Warning: could not download {label}: {exc}")
    print(f"Loaded {len(result)}/{len(MEME_URLS)} reaction images")
    return result


def choose_reaction(hand_labels, pose_landmarks):
    """Choose a reaction category from the combined hand + body pose."""
    left_up = right_up = False
    both_near_head = False
    if pose_landmarks:
        lm = pose_landmarks[0]
        left_up = lm[15].visibility > .6 and lm[15].y < lm[11].y
        right_up = lm[16].visibility > .6 and lm[16].y < lm[12].y
        # wrists close to ears/head: classic shocked / hands-on-head pose
        both_near_head = (
            lm[15].visibility > .6 and lm[16].visibility > .6 and
            abs(lm[15].x - lm[7].x) < .18 and abs(lm[15].y - lm[7].y) < .22 and
            abs(lm[16].x - lm[8].x) < .18 and abs(lm[16].y - lm[8].y) < .22
        )

    if both_near_head:
        return "SHOCKED"
    if left_up and right_up:
        return "BOTH HANDS UP"
    if "PEACE" in hand_labels:
        return "PEACE"
    if "POINT" in hand_labels:
        return "POINT"
    if "FIST" in hand_labels:
        return "FIST"
    if "OPEN PALM" in hand_labels and (left_up or right_up):
        return "WAVE"
    if "OPEN PALM" in hand_labels:
        return "OPEN PALM"
    if left_up:
        return "LEFT HAND UP"
    if right_up:
        return "RIGHT HAND UP"
    return "READY"


def make_reaction_panel(label, height, meme_images=None, width=None):
    """Reference-style right pane: a large real meme image, no synthetic card UI."""
    width = width or int(height * 0.90)
    panel = np.full((height, width, 3), 245, dtype=np.uint8)

    if label == "READY":
        cv2.putText(panel, "Show a gesture", (35, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,80), 2, cv2.LINE_AA)
        return panel

    img = None
    if meme_images:
        img = meme_images.get(label)
        if img is None and meme_images:
            # Never leave the right side blank merely because one URL was rate-limited.
            img = next(iter(meme_images.values()))

    if img is None:
        cv2.putText(panel, label, (35, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,50,50), 2, cv2.LINE_AA)
        return panel

    ih, iw = img.shape[:2]
    scale = min(width / max(iw, 1), height / max(ih, 1))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (width - nw) // 2, (height - nh) // 2
    panel[y:y+nh, x:x+nw] = img
    return panel


def main():
    ensure_model(HAND_MODEL, HAND_URL)
    ensure_model(POSE_MODEL, POSE_URL)
    meme_images = load_web_memes()

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
            labels = []\n            hand_labels = []

            for lm in hr.hand_landmarks:
                draw_chain(frame, lm, hand_connections, (0, 255, 80))
                hand_label = classify_hand(lm)\n                hand_labels.append(hand_label)\n                labels.append(hand_label)

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
            active = choose_reaction(hand_labels, pr.pose_landmarks)
            reaction = make_reaction_panel(
                active, frame.shape[0], meme_images,
                width=max(420, frame.shape[1] // 2)
            )
            combined = np.hstack((frame, reaction))
            cv2.imshow("Gesture Reaction Camera", combined)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
