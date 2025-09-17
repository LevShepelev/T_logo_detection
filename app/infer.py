import torch, numpy as np
from PIL import Image, ImageOps
import cv2

class TBankDetector:
    def __init__(self, weights="models/tbank_detector.pt", device=None, imgsz=896, score=0.30, iou=0.50):
        from ultralytics import YOLO
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.model.to(self.device)
        self.imgsz = imgsz
        self.score = score   # из лучшего на валиде
        self.iou = iou

    @torch.no_grad()
    def detect(self, img_bytes: bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.score, iou=self.iou, verbose=False, device=0 if self.device=="cuda" else None)
        boxes = []
        if len(res) and len(res[0].boxes):
            for b in res[0].boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2 = map(int, b[:4])
                boxes.append((x1,y1,x2,y2))
        return boxes
