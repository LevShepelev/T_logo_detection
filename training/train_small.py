#!/usr/bin/env python3
import argparse, os
from ultralytics import YOLO
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/configs/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")  # auto|cpu|0|0,1
    ap.add_argument("--project", default="runs/train_tbank")
    ap.add_argument("--name", default="y8s_896")
    ap.add_argument("--save-best", action="store_true")
    args = ap.parse_args()

    # Resolve device='auto'
    dev = args.device
    if dev == "auto":
        dev = "0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"[train] Using device={dev} (cuda_available={torch.cuda.is_available()}, count={torch.cuda.device_count()})")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=dev,                    # <- pass resolved device
        project=args.project, name=args.name,
        workers=4,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
        patience=30,
        save=True,
        save_period=-1
    )

    best = results.save_dir / "weights" / "best.pt"
    os.makedirs("models", exist_ok=True)
    if best.exists():
        import shutil
        shutil.copy2(best, "models/tbank_detector.pt")
        print("Saved best -> models/tbank_detector.pt")
    else:
        print("No best.pt found at", best)

if __name__ == "__main__":
    main()
