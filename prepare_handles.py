"""Prepare the photos of our own handles (handles.zip -> data/handles).

1. Resize: phone photos are 4080x3072; we store them EXIF-rotated at 1280px
   long side and split them 70/15/15 into train/valid/test (fixed seed).
2. Pre-label (optional): draft boxes from a trained detector, e.g. the stage1
   model. Drafts go into data/handles/*/labels and MUST be reviewed by hand
   (Roboflow / CVAT / Label Studio) before stage2 training.
3. Negatives (optional): resize + split photos with no handle in them
   straight into data/handles_neg (no labels needed - a missing label file
   is a negative), seeding that folder from data/handles on first use.

Usage:
  python prepare_handles.py                                         # resize + split only
  python prepare_handles.py --weights runs/stage1/weights/best.pt   # + draft labels
  python prepare_handles.py --zero-shot                             # + YOLOE "door handle" drafts (poor, see README)
  python prepare_handles.py --negatives new/                        # negatives -> data/handles_neg
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "handles.zip"
RAW = ROOT / "data/handles_raw"
OUT = ROOT / "data/handles"
HANDLES_NEG = ROOT / "data/handles_neg"
LONG_SIDE = 1280


def unzip():
    if not RAW.exists():
        RAW.mkdir(parents=True)
        with zipfile.ZipFile(ZIP) as z:
            for name in z.namelist():
                if name.lower().endswith(".jpg"):
                    (RAW / Path(name).name).write_bytes(z.read(name))


def resize_all():
    images = sorted(RAW.glob("*.jpg"))
    random.Random(0).shuffle(images)
    n_test = n_val = max(1, round(len(images) * 0.15))
    splits = {"test": images[:n_test], "valid": images[n_test:n_test + n_val],
              "train": images[n_test + n_val:]}
    for split, files in splits.items():
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = OUT / split / "images" / f.name
            if dst.exists():
                continue
            im = ImageOps.exif_transpose(Image.open(f))
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
            im.save(dst, quality=92)
        print(f"{split}: {len(files)} images")


def prepare_negatives(src):
    if not HANDLES_NEG.exists():
        shutil.copytree(OUT, HANDLES_NEG)

    images = sorted(p for p in Path(src).glob("*.jpg"))
    random.Random(0).shuffle(images)
    n_test = n_val = max(1, round(len(images) * 0.15))
    splits = {"test": images[:n_test], "valid": images[n_test:n_test + n_val],
              "train": images[n_test + n_val:]}
    for split, files in splits.items():
        dst_dir = HANDLES_NEG / split / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for f in files:
            dst = dst_dir / f.name
            if dst.exists():
                continue
            im = ImageOps.exif_transpose(Image.open(f))
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
            im.save(dst, quality=92)
            added += 1
        print(f"handles_neg/{split}: +{added} negatives ({len(files)} in this split)")


def prelabel(weights, zero_shot, conf):
    from ultralytics import YOLO, YOLOE

    if zero_shot:
        model = YOLOE("yoloe-11s-seg.pt")
        names = ["door handle"]
        model.set_classes(names, model.get_text_pe(names))
    else:
        model = YOLO(weights)

    for img in sorted(OUT.glob("*/images/*.jpg")):
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        if label.exists():  # never overwrite reviewed labels
            continue
        r = model.predict(img, conf=conf, imgsz=1280, verbose=False)[0]
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in r.boxes.xywhn.tolist()]
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{img.name}: {len(lines)} boxes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="detector for draft labels, e.g. runs/stage1/weights/best.pt")
    ap.add_argument("--zero-shot", action="store_true", help="draft labels from open-vocab YOLOE")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--negatives", help="dir of raw photos with no handle -> data/handles_neg")
    args = ap.parse_args()
    if args.negatives:
        prepare_negatives(args.negatives)
    else:
        unzip()
        resize_all()
        if args.weights or args.zero_shot:
            prelabel(args.weights, args.zero_shot, args.conf)
