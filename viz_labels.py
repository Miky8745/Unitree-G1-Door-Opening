"""Draw YOLO labels onto images: python viz_labels.py <out.jpg> <img> [<img> ...]"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

tiles = []
for p in map(Path, sys.argv[2:]):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
    for line in (lbl.read_text().split("\n") if lbl.exists() else []):
        if line.strip():
            _, x, y, w, h = map(float, line.split()[:5])
            d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                        outline="red", width=max(2, W // 200))
    im.thumbnail((400, 400))
    tiles.append(im)
cols = 5
rows = -(-len(tiles) // cols)
sheet = Image.new("RGB", (cols * 404, rows * 404), "white")
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 404, (i // cols) * 404))
sheet.save(sys.argv[1])
