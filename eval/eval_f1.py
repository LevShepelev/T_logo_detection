#!/usr/bin/env python3
import argparse, os, glob, math
import numpy as np
from ultralytics import YOLO

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua>0 else 0.0

def load_gt(label_path, w, h):
    boxes=[]
    if os.path.isfile(label_path):
        for line in open(label_path):
            parts = line.strip().split()
            if len(parts)!=5: continue
            _, cx, cy, bw, bh = parts
            cx,cy,bw,bh = map(float, (cx,cy,bw,bh))
            x1 = (cx - bw/2)*w; y1 = (cy - bh/2)*h
            x2 = (cx + bw/2)*w; y2 = (cy + bh/2)*h
            boxes.append([x1,y1,x2,y2])
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/tbank_detector.pt")
    ap.add_argument("--data-root", default="data/yolo")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf-grid", default="0.10,0.60,0.02")  # start, stop, step
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    start, stop, step = map(float, args.conf_grid.split(","))
    grid = np.arange(start, stop+1e-9, step)

    model = YOLO(args.weights)

    val_imgs = sorted(glob.glob(os.path.join(args.data_root, "images/val/*")))
    val_labs = os.path.join(args.data_root, "labels/val")

    best = (0.0, 0.0, 0.0, 0.0)  # F1, P, R, conf

    for conf in grid:
        TP=FP=FN=0
        for ip in val_imgs:
            # предсказания
            r = model.predict(ip, imgsz=args.imgsz, conf=conf, iou=0.5, verbose=False)[0]
            preds = r.boxes.xyxy.cpu().numpy().tolist()
            # GT
            import PIL.Image as Image, PIL.ImageOps as ImageOps
            im = ImageOps.exif_transpose(Image.open(ip).convert("RGB"))
            w,h = im.size
            base = os.path.splitext(os.path.basename(ip))[0]
            gt = load_gt(os.path.join(val_labs, base + ".txt"), w, h)

            matched = set()
            for p in preds:
                hit=False
                for j,g in enumerate(gt):
                    if j in matched: continue
                    if iou(p,g) >= args.iou_thr:
                        matched.add(j); hit=True; break
                if hit: TP+=1
                else: FP+=1
            FN += (len(gt)-len(matched))

        P = TP/(TP+FP) if TP+FP>0 else 0.0
        R = TP/(TP+FN) if TP+FN>0 else 0.0
        F1 = 2*P*R/(P+R) if P+R>0 else 0.0
        if F1 > best[0]:
            best = (F1, P, R, conf)

    F1,P,R,conf = best
    print(f"Best F1@0.5 = {F1:.4f} @ conf={conf:.2f}  (P={P:.4f}, R={R:.4f})")

if __name__ == "__main__":
    main()
