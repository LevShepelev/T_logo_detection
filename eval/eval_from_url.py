#!/usr/bin/env python3
import argparse, os, sys, hashlib, tempfile, shutil, urllib.request, zipfile, tarfile
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps

def download(url: str, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        read = 0
        block = 1 << 16
        while True:
            chunk = r.read(block)
            if not chunk: break
            f.write(chunk)
            read += len(chunk)
            if total:
                done = int(50 * read / total)
                sys.stdout.write("\r[{}{}] {}/{}".format("#"*done, "."*(50-done), read, total))
                sys.stdout.flush()
    sys.stdout.write("\n")
    return dst_path

def extract(archive: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z: z.extractall(out_dir)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as t: t.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")
    return out_dir

def find_yolo_val_root(root: Path):
    """
    Accept common layouts:
      - images/val, labels/val (recommended)
      - val/images, val/labels
    Return (images_val_dir, labels_val_dir)
    """
    cands = [
        (root/"images"/"val", root/"labels"/"val"),
        (root/"val"/"images", root/"val"/"labels"),
    ]
    for iv, lv in cands:
        if iv.is_dir() and lv.is_dir():
            return iv, lv
    # search one level deeper
    for p in root.rglob("*"):
        if p.is_dir() and (p/"images"/"val").is_dir() and (p/"labels"/"val").is_dir():
            return p/"images"/"val", p/"labels"/"val"
    raise FileNotFoundError("Could not locate YOLO val dirs (images/val, labels/val) in extracted archive")

def load_gt(label_path: Path, w: int, h: int):
    boxes=[]
    if label_path.is_file():
        for line in open(label_path):
            parts=line.strip().split()
            if len(parts)!=5: continue
            _, cx, cy, bw, bh = parts
            cx,cy,bw,bh = map(float,(cx,cy,bw,bh))
            x1=(cx-bw/2)*w; y1=(cy-bh/2)*h
            x2=(cx+bw/2)*w; y2=(cy+bh/2)*h
            boxes.append([x1,y1,x2,y2])
    return boxes

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def main():
    ap = argparse.ArgumentParser("Evaluate F1@0.5 on a remote val archive")
    ap.add_argument("--weights", default="models/tbank_detector.pt")
    ap.add_argument("--val-url", help="Public URL to ZIP/TAR.GZ with YOLO val (images/val, labels/val)")
    ap.add_argument("--gdrive-id", help="Alternative: Google Drive file id (use if you can't provide direct URL)")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf-grid", default="0.10,0.60,0.02")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    tmp_root = Path(tempfile.mkdtemp(prefix="tbank_val_"))
    try:
        # Download
        if args.gdrive-id:
            print("Please install gdown: poetry run pip install gdown")
            print("Then run: gdown --fuzzy 'https://drive.google.com/uc?id={}' -O {}".format(args.gdrive_id, tmp_root/"val.zip"))
            return
        if not args.val_url:
            raise SystemExit("--val-url or --gdrive-id is required")

        arch = tmp_root / ("val" + (".zip" if args.val_url.endswith(".zip") else ".tar.gz"))
        print(f"Downloading val to {arch} ...")
        download(args.val_url, arch)

        # Extract
        extract_dir = tmp_root / "unzipped"
        print(f"Extracting to {extract_dir} ...")
        extract(arch, extract_dir)

        # Find val folders
        iv, lv = find_yolo_val_root(extract_dir)
        print("Found val dirs:", iv, "|", lv)

        # Build list of images
        from glob import glob
        img_paths = sorted(sum([glob(str(iv/"*.jpg")), glob(str(iv/"*.jpeg")), glob(str(iv/"*.png")), glob(str(iv/"*.bmp")), glob(str(iv/"*.webp"))], []))
        if not img_paths:
            raise FileNotFoundError("No images found in extracted val/images")

        # Load model
        model = YOLO(args.weights)

        # Conf sweep
        start, stop, step = map(float, args.conf_grid.split(","))
        grid = np.arange(start, stop+1e-9, step)
        best = (0.0, 0.0, 0.0, 0.0)  # F1,P,R,conf

        for conf in grid:
            TP=FP=FN=0
            for ip in img_paths:
                r = model.predict(ip, imgsz=args.imgsz, conf=conf, iou=0.5, verbose=False)[0]
                preds = r.boxes.xyxy.cpu().numpy().tolist()
                im = ImageOps.exif_transpose(Image.open(ip).convert("RGB"))
                w,h = im.size
                lab = lv / (Path(ip).stem + ".txt")
                gt = load_gt(lab, w, h)
                matched=set()
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
                best = (F1,P,R,conf)

        print(f"Best F1@0.5 = {best[0]:.4f} @ conf={best[3]:.2f}  (P={best[1]:.4f}, R={best[2]:.4f})")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
