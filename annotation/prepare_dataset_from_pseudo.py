#!/usr/bin/env python3
import argparse, os, shutil, random, glob

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

def list_images(folder):
    out=[]
    for ext in IMG_EXTS:
        out += glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True)
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels_yolo")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    imgs = list_images(args.images)
    random.shuffle(imgs)

    train_n = int(len(imgs)*args.train_ratio)
    splits = [("train", imgs[:train_n]), ("val", imgs[train_n:])]

    for split, items in splits:
        img_dir = os.path.join(args.out, "images", split)
        lab_dir = os.path.join(args.out, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        moved = 0
        for ip in items:
            base = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(args.labels, base + ".txt")
            # можно копировать даже без лейбла (негатив), но лучше требовать файл
            if not os.path.isfile(lp):
                # негативные кадры допустимы — создаём пустой .txt
                open(lp, "a").close()
            shutil.copy2(ip, os.path.join(img_dir, os.path.basename(ip)))
            shutil.copy2(lp, os.path.join(lab_dir, base + ".txt"))
            moved += 1
        print(f"{split}: {moved} samples -> {img_dir}, {lab_dir}")

if __name__ == "__main__":
    main()
