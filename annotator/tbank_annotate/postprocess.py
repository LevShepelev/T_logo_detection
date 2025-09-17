from __future__ import annotations
import numpy as np
import torch
from torchvision.ops import nms

def nms_np(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores
    keep = nms(torch.tensor(boxes_xyxy, dtype=torch.float32),
               torch.tensor(scores, dtype=torch.float32),
               iou_thr).cpu().numpy()
    return boxes_xyxy[keep], scores[keep]

def filter_shape(boxes_xyxy: np.ndarray,
                 scores: np.ndarray | None,
                 w: int, h: int,
                 min_side: int = 16,
                 aspect_min: float = 0.6,
                 aspect_max: float = 1.4) -> tuple[np.ndarray, np.ndarray | None]:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        bw, bh = (x2 - x1), (y2 - y1)
        if bw < min_side or bh < min_side:
            continue
        ar = bw / max(bh, 1e-6)
        if not (aspect_min <= ar <= aspect_max):
            continue
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        keep.append(i)
    boxes = boxes_xyxy[keep]
    scores = scores[keep] if scores is not None else None
    return boxes, scores

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter / union, 0.0)

def subtract_negatives(pos: np.ndarray, neg: np.ndarray, iou_thr: float = 0.4) -> np.ndarray:
    if len(pos) == 0 or len(neg) == 0:
        return pos
    m = iou_matrix(pos, neg)
    keep = (m < iou_thr).all(axis=1)
    return pos[keep]

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    return cx / w, cy / h, bw / w, bh / h
