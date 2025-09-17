from __future__ import annotations
import argparse, os, time
from tqdm import tqdm
from PIL import Image
from .detector import OWLDetector
from .prompts import SETS
from .postprocess import subtract_negatives
from .utils import list_images, load_pil, save_yolo, save_viz

def build_parser():
    p = argparse.ArgumentParser("tbank-annotate: zero-shot labeling for T-Bank logo")
    p.add_argument("--input", required=True, help="folder with images")
    p.add_argument("--out-labels", default="labels_yolo", help="output folder for YOLO txt")
    p.add_argument("--out-viz", default="viz_pos", help="output folder for positive previews")
    p.add_argument("--model", default="google/owlv2-large-patch14",
                   help="HF model id, e.g. google/owlv2-large-patch14 or google/owlvit-large-patch14")
    p.add_argument("--pos-set", choices=["strict","balanced"], default="strict", help="positive prompts set")
    p.add_argument("--use-neg", action="store_true", help="run negative prompts and subtract by IoU")
    p.add_argument("--neg-iou", type=float, default=0.4, help="IoU threshold to subtract negatives")
    # thresholds
    p.add_argument("--score", type=float, default=0.26, help="score threshold for positives")
    p.add_argument("--neg-score", type=float, default=0.28, help="score threshold for negatives")
    p.add_argument("--nms", type=float, default=0.50, help="NMS IoU")
    p.add_argument("--topk", type=int, default=150, help="top-K candidates before NMS")
    p.add_argument("--min-side", type=int, default=16, help="min side in px")
    p.add_argument("--ar-min", type=float, default=0.6, help="min aspect ratio w/h")
    p.add_argument("--ar-max", type=float, default=1.4, help="max aspect ratio w/h")
    p.add_argument("--device", default=None, help="cuda or cpu (auto by default)")
    p.add_argument("--fp32", action="store_true", help="force FP32 even on CUDA")
    return p

def main():
    args = build_parser().parse_args()

    det = OWLDetector(model_id=args.model,
                      device=args.device,
                      fp16=(not args.fp32))

    pos_prompts = SETS[args.pos_set]
    neg_prompts = SETS["neg"]

    imgs = list_images(args.input)
    print(f"Found images: {len(imgs)}")
    os.makedirs(args.out_labels, exist_ok=True)
    os.makedirs(args.out_viz, exist_ok=True)

    t0 = time.time()
    pos_count = 0

    for p in tqdm(imgs):
        try:
            pil = load_pil(p)
            # positives
            pos_boxes, pos_scores = det.detect(
                pil, pos_prompts,
                score_thresh=args.score,
                nms_iou=args.nms,
                top_k=args.topk,
                shape_min_side=args.min_side,
                shape_ar_min=args.ar_min,
                shape_ar_max=args.ar_max
            )

            # negatives (optional subtraction)
            if args.use_neg and len(pos_boxes) > 0:
                neg_boxes, _ = det.detect(
                    pil, neg_prompts,
                    score_thresh=args.neg_score,
                    nms_iou=args.nms,
                    top_k=args.topk,
                    shape_min_side=max(8, args.min_side-4),  # чуть мягче
                    shape_ar_min=0.2, shape_ar_max=5.0       # шире покрытие негативов
                )
                pos_boxes = subtract_negatives(pos_boxes, neg_boxes, iou_thr=args.neg_iou)

            # save only positives
            if len(pos_boxes) > 0:
                if save_yolo(p, pos_boxes, args.out_labels):
                    save_viz(p, pos_boxes, pos_scores, args.out_viz)
                    pos_count += 1
            else:
                # чистим пустой лейбл, если существует
                base = os.path.splitext(os.path.basename(p))[0]
                fp = os.path.join(args.out_labels, base + ".txt")
                if os.path.exists(fp):
                    os.remove(fp)
        except Exception as e:
            print("Error on", p, "->", repr(e))
            continue

    dt = time.time() - t0
    print(f"Done. Positive previews: {pos_count} | Time: {dt:.1f}s")
    print("Labels:", os.path.abspath(args.out_labels))
    print("Viz:", os.path.abspath(args.out_viz))

if __name__ == "__main__":
    main()
