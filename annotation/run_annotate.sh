#!/usr/bin/env bash
set -e

INPUT_DIR="${1:-data/images}"
OUT_LAB="data/labels_yolo"
OUT_VIZ="data/viz_pos"

# pip install -e .[train]   # если нужно ultralytics позже
# pip install -e .          # чтобы появился tbank-annotate

tbank-annotate \
  --input "$INPUT_DIR" \
  --out-labels "$OUT_LAB" \
  --out-viz "$OUT_VIZ" \
  --model google/owlv2-large-patch14 \
  --pos-set strict \
  --use-neg \
  --score 0.26 --neg-score 0.30 \
  --nms 0.5 --topk 150 \
  --min-side 16 --ar-min 0.6 --ar-max 1.4 \
  --neg-iou 0.4
