"""Triage candidate clips for the pipeline: people per frame, whole body in frame, camera motion."""
import glob, sys
import cv2, numpy as np
from ultralytics import YOLO
m = YOLO("/home/sli3/GVHMR/inputs/checkpoints/yolo/yolov8x.pt")
for f in sorted(glob.glob("*.mp4")):
    c = cv2.VideoCapture(f); fps = c.get(5); n = int(c.get(7)); W, H = int(c.get(3)), int(c.get(4))
    step = max(1, int(fps / 2)); persons, cut, shifts, prev = [], 0, [], None
    for i in range(0, n, step):
        c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = c.read()
        if not ok: break
        small = cv2.resize(fr, (480, int(480 * H / W)))
        r = m(small, classes=[0], conf=0.5, verbose=False)[0]
        boxes = [b for b in r.boxes.xyxy.cpu().numpy() if (b[2]-b[0])*(b[3]-b[1]) > 0.02*small.shape[0]*small.shape[1]]
        persons.append(len(boxes))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if boxes:
            b = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            if b[3] > small.shape[0] - 3 or b[1] < 3: cut += 1        # head or feet touching the frame edge
            x0, y0, x1, y1 = map(int, b); g[y0:y1, x0:x1] = g.mean()   # mask the person: measure the background only
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g); shifts.append(np.hypot(dx, dy) / small.shape[1] * fps / step)
        prev = g
    p = np.array(persons)
    print(f"{f:42s} {W}x{H} {n/fps:5.1f}s  one-person {np.mean(p==1)*100:4.0f}%  body-cut {cut/len(p)*100:4.0f}%  "
          f"camera {np.median(shifts)*100:5.1f}%/s (p90 {np.percentile(shifts,90)*100:5.1f})")
