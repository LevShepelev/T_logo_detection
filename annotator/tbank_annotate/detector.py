from __future__ import annotations
import os, glob
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OWLDetector:
    def __init__(self,
                 model_id: str = "google/owlv2-large-patch14-ensemble",
                 device: str | None = None,
                 fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()


        # Hard sanity check: fail fast if anything stayed on CPU accidentally
        p = next(self.model.parameters())
        if self.device == "cuda" and p.device.type != "cuda":
            raise RuntimeError(
                f"Model is not on CUDA after .to(): param device={p.device}"
            )

        self.fp16 = bool(fp16 and self.device == "cuda")
    @torch.no_grad()
    def detect(
        self,
        pil_img: Image.Image,
        texts: list[str],
        score_thresh: float = 0.24,
        nms_iou: float = 0.5,
        top_k: int = 150,
        shape_min_side: int = 16,
        shape_ar_min: float = 0.6,
        shape_ar_max: float = 1.4,
    ):
        # Prepare inputs on the same device as the model
        w, h = pil_img.size
        inputs = self.processor(
            text=[texts],
            images=[pil_img],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Forward
        outputs = self.model(**inputs)

        # Post-process to absolute xyxy on CPU-friendly tensors
        target_sizes = torch.tensor([[h, w]], device=self.device)
        r = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes
        )[0]
        boxes = r["boxes"].detach().cpu().numpy()   # [N,4] xyxy (pixels)
        scores = r["scores"].detach().cpu().numpy() # [N]

        # Threshold / Top-K / NMS / Shape filters
        keep = np.where(scores >= score_thresh)[0]
        boxes, scores = boxes[keep], scores[keep]

        if len(scores) > top_k:
            top = scores.argsort()[::-1][:top_k]
            boxes, scores = boxes[top], scores[top]

        from .postprocess import nms_np, filter_shape
        boxes, scores = nms_np(boxes, scores, nms_iou)
        boxes, scores = filter_shape(
            boxes, scores, w, h,
            min_side=shape_min_side,
            aspect_min=shape_ar_min,
            aspect_max=shape_ar_max,
        )
        return boxes, scores
