"""Two-stage door handle detector training.

  stage1: COCO-pretrained YOLO -> general door-handle dataset (Roboflow)
  stage2: stage1 weights -> our handles (data/handles), mixed with general data
          so the model doesn't forget other handle types
  export: TensorRT/ONNX for the G1's Jetson Orin

Usage:
  python train.py stage1 [--model yolo11s.pt]
  python train.py stage2
  python train.py eval --dataset handles_mix        # test split of our handles
  python train.py predict --source handle.mp4       # eyeball results on photos/video
  python train.py export --format engine            # run ON the Jetson for TensorRT
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT / "Door handle.v1-door_handle_one_ann.yolov8"
HANDLES = ROOT / "data/handles"
RUNS = ROOT / "runs"


def write_yaml(name, train, val, test=None):
    path = ROOT / "data" / f"{name}.yaml"
    path.parent.mkdir(exist_ok=True)
    fmt = lambda v: "[" + ", ".join(f"'{p}'" for p in v) + "]" if isinstance(v, list) else f"'{v}'"
    text = f"train: {fmt(train)}\nval: {fmt(val)}\n"
    if test:
        text += f"test: {fmt(test)}\n"
    path.write_text(text + "nc: 1\nnames: ['Door-handle']\n")
    return path


def stage1(args):
    data = write_yaml("general", GENERAL / "train/images", GENERAL / "valid/images",
                      GENERAL / "test/images")
    YOLO(args.model).train(
        data=data, epochs=args.epochs or 100, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, patience=25, project=RUNS, name="stage1", exist_ok=True,
        # handles are small; scale/mosaic augmentation helps far-away doors
        scale=0.5, mosaic=1.0, close_mosaic=10,
    )


def stage2(args):
    # Our handles listed `oversample` times so they dominate the mix (~70%+).
    data = write_yaml(
        "handles_mix",
        train=[str(HANDLES / "train/images")] * args.oversample + [str(GENERAL / "train/images")],
        val=HANDLES / "valid/images",
        test=HANDLES / "test/images",
    )
    YOLO(args.weights or RUNS / "stage1/weights/best.pt").train(
        data=data, epochs=args.epochs or 60, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, lr0=0.001, warmup_epochs=1, freeze=args.freeze, patience=20,
        project=RUNS, name="stage2", exist_ok=True,
        scale=0.5, degrees=5, close_mosaic=10,
    )


def evaluate(args):
    model = YOLO(args.weights or RUNS / "stage2/weights/best.pt")
    data = ROOT / "data" / f"{args.dataset}.yaml"
    m = model.val(data=data, split="test", imgsz=args.imgsz, batch=args.batch,
                  project=RUNS, name=f"eval_{args.dataset}", exist_ok=True)
    print(f"{args.dataset} test: mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} "
          f"P={m.box.mp:.3f} R={m.box.mr:.3f}")


def predict(args):
    model = YOLO(args.weights or RUNS / "stage2/weights/best.pt")
    for source in args.source:  # one at a time: ultralytics can't mix images and videos
        model.predict(source, imgsz=args.imgsz, conf=0.25, save=True, stream=False,
                      project=RUNS, name="predict", exist_ok=True, verbose=False)
    print(f"saved to {RUNS / 'predict'}")


def export(args):
    model = YOLO(args.weights or RUNS / "stage2/weights/best.pt")
    # Run this ON the Jetson for format=engine: TensorRT engines are GPU-specific.
    model.export(format=args.format, imgsz=args.imgsz, half=args.format == "engine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2", "eval", "predict", "export"])
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--weights")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto-fit to GPU memory")
    ap.add_argument("--workers", type=int, default=8, help="dataloader processes; lower if workers crash (Windows RAM)")
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers in stage2 (10 = backbone)")
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--dataset", default="handles_mix", help="eval: data/<name>.yaml")
    ap.add_argument("--source", nargs="+", default=["close.jpg", "far.jpg", "handle.mp4"])
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    args = ap.parse_args()
    {"stage1": stage1, "stage2": stage2, "eval": evaluate, "predict": predict,
     "export": export}[args.cmd](args)
