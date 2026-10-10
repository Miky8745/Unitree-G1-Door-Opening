# G1 door opening — handle detector

Two-stage YOLO11 detector for door handles (step 1 of the G1 door-opening pipeline).

## Data
| Path | What |
|---|---|
| `Door handle.v1-door_handle_one_ann.yolov8/` | General handles (Roboflow, CC BY 4.0). 397/114/57 train/valid/test, boxes only, 640x480 stretched |
| `handles.zip` → `data/handles/` | 70 photos of our own handles. **Not labeled yet.** |
| `close.jpg`, `far.jpg`, `handle.mp4` | Sanity-check media for `predict` |

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv ultralytics
source .venv/bin/activate
```

## Workflow
```bash
# 1. General handles (COCO-pretrained yolo11s). ~100 epochs.
python train.py stage1

# 2. Our photos: resize + split, then draft labels with the stage1 model
python prepare_handles.py --weights runs/stage1/weights/best.pt

# 3. REVIEW the drafts in data/handles/{train,valid,test}/labels:
#    fix boxes, add missed handles (especially small/far ones). Upload
#    data/handles to Roboflow/CVAT as a YOLO dataset, fix, export back.
#    Optional: python viz_labels.py sheet.jpg data/handles/train/images/*.jpg

# 4. Fine-tune on our handles (oversampled 4x, mixed with general data)
python train.py stage2

# 5. Evaluate on our held-out test photos, and look at the video
python train.py eval --dataset handles_mix
python train.py predict                 # close.jpg, far.jpg, handle.mp4 -> runs/predict

# 6. Deploy: export on the Jetson itself (TensorRT engines are GPU-specific)
python train.py export --format engine
```

## Notes
- **GPU memory.** On the Quadro M2000M laptop (4 GB, with only about 1 GB free
  after the desktop and browser) yolo11s ran out of memory at batch 8, and
  Ultralytics fell back to batch 1. Train on a bigger GPU. On a small card use `--batch -1` (auto-fit to
  free memory) or `--model yolo11n.pt`.
- **Small handles.** In far shots the handle is about 1% of the image width. If
  `predict` misses far handles, retrain stage 2 with `--imgsz 960`. Note that this
  costs FPS on the Jetson.
- **Zero-shot labels don't work here.** YOLOE prompted with "door handle"
  (`prepare_handles.py --zero-shot`) missed close-up handles and boxed whole doors
  and cabinets. That's why the drafts come from the stage1 model instead.
- **Test set.** The handles test split is phone photos. The real test is frames
  from the G1 head camera; add some of those to `data/handles/test` as soon as you
  can capture them.
- **Next step.** To grasp a handle you need more than a box: you need its pivot
  and tip, plus depth. Consider re-labeling the handles set as keypoints and
  training a YOLO-pose model.
