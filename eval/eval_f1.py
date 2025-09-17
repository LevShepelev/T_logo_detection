#!/usr/bin/env python3
# eval/eval_f1.py  — robust F1@0.5 sweep with progress & checkpoint
import argparse, os, glob, csv, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from tqdm import tqdm

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def load_gt(label_path: Path, w: int, h: int):
    boxes = []
    if label_path.is_file():
        for line in open(label_path, "r", encoding="utf-8", errors="ignore"):
            parts = line.strip().split()
            if len(parts) != 5: continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw/2)*w; y1 = (cy - bh/2)*h
            x2 = (cx + bw/2)*w; y2 = (cy + bh/2)*h
            boxes.append([x1, y1, x2, y2])
    return boxes

def resolve_device(device: str):
    import torch
    if device == "auto":
        return "0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    return device

def pil_open(path: str):
    # Use PIL so we don’t rely on cv2’s imread in Ultralytics
    im = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(im)

def safe_predict(model, items, imgsz, conf, device, batch):
    """Try batched predict with PIL images, fall back per-image if something explodes."""
    results = []
    skipped = 0
    try:
        # items are PIL Images
        res = model.predict(items, imgsz=imgsz, conf=conf, iou=0.5,
                            verbose=False, device=device, batch=batch)
        return res, skipped
    except Exception as e:
        # fallback per-image
        for it in items:
            try:
                r = model.predict(it, imgsz=imgsz, conf=conf, iou=0.5,
                                  verbose=False, device=device)
                results.append(r[0])
            except Exception:
                skipped += 1
                results.append(None)
        return results, skipped

def write_row(csv_path: Path, row: dict, header: list[str]):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser("Evaluate F1@0.5 with progress, checkpoint, and robust loading")
    ap.add_argument("--weights", default="models/tbank_detector.pt")
    ap.add_argument("--data-root", default="data/yolo")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf-grid", default="0.10,0.60,0.02")  # start, stop, step
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-csv", default="eval/f1_conf_sweep.csv")
    args = ap.parse_args()

    device = resolve_device(args.device)

    # Collect val images
    iv = Path(args.data_root) / "images" / "val"
    lv = Path(args.data_root) / "labels" / "val"
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(str(iv / e))
    img_paths = sorted(img_paths)
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]
    if not img_paths:
        raise FileNotFoundError(f"No images found in {iv}")

    # Preindex sizes and GT
    print(f"[prep] indexing {len(img_paths)} images & GT …")
    meta = {}
    for ip in tqdm(img_paths, total=len(img_paths), desc="Indexing", dynamic_ncols=True):
        im = pil_open(ip)               # PIL read here catches many bad files early
        w, h = im.size
        lab = lv / (Path(ip).stem + ".txt")
        gt = load_gt(lab, w, h)
        meta[ip] = (w, h, gt)

    # Load model once
    model = YOLO(args.weights)

    # Conf grid
    start, stop, step = map(float, args.conf_grid.split(","))
    grid = np.arange(start, stop + 1e-9, step)

    # Prepare CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["conf","TP","FP","FN","P","R","F1","processed","skipped","time_sec"]

    best = (0.0, 0.0, 0.0, 0.0)  # F1,P,R,conf

    outer = tqdm(grid, desc="Conf sweep", dynamic_ncols=True)
    for conf in outer:
        t0 = time.time()
        TP = FP = FN = 0
        processed = 0
        skipped_total = 0

        # Iterate batches with PIL images
        inner = tqdm(total=len(img_paths), desc=f"conf={conf:.2f}", leave=False, dynamic_ncols=True)
        for i in range(0, len(img_paths), args.batch):
            batch_paths = img_paths[i : i + args.batch]
            batch_imgs = [pil_open(p) for p in batch_paths]

            res, skipped = safe_predict(model, batch_imgs, args.imgsz, conf, device, args.batch)
            skipped_total += skipped

            for ip, r in zip(batch_paths, res):
                if r is None:
                    continue  # counted in skipped
                preds = r.boxes.xyxy.cpu().numpy().tolist() if r and r.boxes is not None else []
                w, h, gt = meta[ip]
                matched = set()
                for p in preds:
                    hit = False
                    for j, g in enumerate(gt):
                        if j in matched: continue
                        if iou(p, g) >= args.iou_thr:
                            matched.add(j); hit = True; break
                    if hit: TP += 1
                    else:   FP += 1
                FN += (len(gt) - len(matched))
                processed += 1

            inner.update(len(batch_paths))
        inner.close()

        P = TP/(TP+FP) if TP+FP>0 else 0.0
        R = TP/(TP+FN) if TP+FN>0 else 0.0
        F1 = 2*P*R/(P+R) if P+R>0 else 0.0
        dt = time.time() - t0

        # checkpoint to CSV each conf
        row = {"conf": f"{conf:.4f}", "TP": TP, "FP": FP, "FN": FN,
               "P": f"{P:.6f}", "R": f"{R:.6f}", "F1": f"{F1:.6f}",
               "processed": processed, "skipped": skipped_total, "time_sec": f"{dt:.2f}"}
        write_row(out_csv, row, header)

        if F1 > best[0]:
            best = (F1, P, R, conf)
        outer.set_postfix_str(f"bestF1={best[0]:.4f}@{best[3]:.2f}, skip={skipped_total}")

        # free GPU mem between steps
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    F1, P, R, conf = best
    print(f"\nBest F1@0.5 = {F1:.4f} @ conf={conf:.2f}  (P={P:.4f}, R={R:.4f})")
    print(f"Per-conf results saved to: {out_csv.resolve()}")
    print("Tip: open it in a spreadsheet to inspect the curve.")
if __name__ == "__main__":
    main()
