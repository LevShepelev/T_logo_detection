from __future__ import annotations
import os, glob
from typing import Iterable
from PIL import Image, ImageOps
import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(folder: str) -> list[str]:
    out = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(out)

def load_pil(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))

def save_yolo(path_img: str, boxes_xyxy, out_dir: str) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path_img))[0]
    out_txt = os.path.join(out_dir, base + ".txt")
    if len(boxes_xyxy) == 0:
        if os.path.exists(out_txt):
            os.remove(out_txt)
        return False
    pil = load_pil(path_img)
    w, h = pil.size
    from .postprocess import xyxy_to_yolo
    with open(out_txt, "w") as f:
        for (x1, y1, x2, y2) in boxes_xyxy:
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return True

def save_viz(path_img: str, boxes_xyxy, scores, out_dir: str) -> str | None:
    if len(boxes_xyxy) == 0:
        return None
    os.makedirs(out_dir, exist_ok=True)
    pil = load_pil(path_img)
    im = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        if scores is not None:
            cv2.putText(im, f"{scores[i]:.2f}", (int(x1), max(0,int(y1)-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)
    out_path = os.path.join(out_dir, os.path.basename(path_img))
    cv2.imwrite(out_path, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
