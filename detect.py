from __future__ import annotations
import time
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def classify_hand(lm):
    fingers = [finger_up(lm,8,6), finger_up(lm,12,10), finger_up(lm,16,14), finger_up(lm,20,18)]
    n=sum(fingers)
    if n == 4: return "OPEN PALM"
    if fingers[0] and not any(fingers[1:]): return "POINT"
    if fingers[0] and fingers[1] and not any(fingers[2:]): return "PEACE"
    if n == 0: return "FIST"
    return "HAND"

def main():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    if not cap.isOpened(): raise RuntimeError("Camera not found")
    prev=time.perf_counter()
    with mp_hands.Hands(model_complexity=1,max_num_hands=2,min_detection_confidence=.6,min_tracking_confidence=.6) as hands, mp_pose.Pose(model_complexity=1,min_detection_confidence=.6,min_tracking_confidence=.6) as pose:
        while True:
            ok, frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            hr=hands.process(rgb); pr=pose.process(rgb)
            labels=[]
            if hr.multi_hand_landmarks:
                for h in hr.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame,h,mp_hands.HAND_CONNECTIONS)
                    labels.append(classify_hand(h.landmark))
            if pr.pose_landmarks:
                mp_draw.draw_landmarks(frame,pr.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                lm=pr.pose_landmarks.landmark
                if lm[15].visibility>.6 and lm[15].y < lm[11].y: labels.append("LEFT HAND UP")
                if lm[16].visibility>.6 and lm[16].y < lm[12].y: labels.append("RIGHT HAND UP")
            now=time.perf_counter(); fps=1/max(now-prev,1e-6); prev=now
            cv2.rectangle(frame,(12,12),(430,105),(20,20,20),-1)
            cv2.putText(frame,f"FPS {fps:4.1f}",(28,48),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,120),2)
            cv2.putText(frame," | ".join(dict.fromkeys(labels)) or "READY",(28,84),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,120),2)
            cv2.imshow("Gesture Reaction Camera",frame)
            if cv2.waitKey(1)&0xFF in (27,ord('q')): break
    cap.release(); cv2.destroyAllWindows()
if __name__=="__main__": main()
