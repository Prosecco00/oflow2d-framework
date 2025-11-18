
import cv2, numpy as np

def load_video_frames(video_path, max_frames=None, stride=1):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Impossibile aprire {video_path}'
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames: break
        i += 1
    cap.release()
    if len(frames) < 2: raise RuntimeError("Servono almeno 2 frame nel video.")
    return frames

def pairwise(frames):
    for i in range(len(frames)-1):
        yield frames[i], frames[i+1]
