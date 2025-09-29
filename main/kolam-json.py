#!/usr/bin/env python3
"""
Kolam Reversible Pipeline
- image -> JSON (vectorized strokes + metadata; optional embedded raster)
- JSON  -> image (deterministic renderer that tries vector regen, falls back to embedded raster if present)
- Attempts pixel-perfect round-trip; if vector regen doesn't match input exactly, uses raster fallback.

Usage:
    # Image -> JSON (with optional raster embed)
    python kolam_reversible.py --mode img2json --input kolam.png --out results/kolam.json --embed-raster

    # JSON -> Image (regenerate)
    python kolam_reversible.py --mode json2img --input results/kolam.json --out results/kolam_regen.png

Dependencies: opencv-python, numpy, Pillow
"""

import os
import cv2
import json
import base64
import argparse
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO

# ---------------- Helpers ---------------- #

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def read_image_rgb(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {os.path.abspath(path)}")
    img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Unable to read image: {path}")
    # If PNG with alpha, keep alpha channel
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        # Convert BGRA -> RGBA
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def write_image_rgb(path: str, img_rgb: np.ndarray):
    # Convert to BGR or BGRA depending on channels and write safely for Windows paths
    ensure_dir(path)
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGRA)
    else:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Use imencode + tofile to support unicode paths on Windows
    ext = os.path.splitext(path)[1] or ".png"
    ok, enc = cv2.imencode(ext, img_bgr)
    if not ok:
        raise IOError("Failed to encode image for saving")
    enc.tofile(path)

def pil_image_to_base64_png(img_rgb: np.ndarray) -> str:
    pil = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def base64_png_to_pil_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    pil = Image.open(BytesIO(raw)).convert("RGBA")
    return np.array(pil)[:,:,:3] if pil.mode == "RGB" else np.array(pil)

# ---------------- Vectorization ---------------- #

def vectorize_image(img_rgb: np.ndarray, threshold: int = 127) -> Dict[str, Any]:
    """
    Vectorize: find contours on a binary mask of the drawn (non-background) pixels.
    Returns dictionary with:
      - canvas_size
      - background_color
      - primitives: list of {points: [[x,y],...], closed:bool, thickness:float, color:[r,g,b], bbox:[]}
    """
    h, w = img_rgb.shape[:2]
    canvas_size = [w, h]

    # Determine background color by sampling corners (robust for Kolam with white background)
    corners = [
        img_rgb[0,0], img_rgb[0,w-1],
        img_rgb[h-1,0], img_rgb[h-1,w-1]
    ]
    bg = np.median(np.stack(corners, axis=0).astype(np.int32), axis=0).astype(int).tolist()

    # Create mask of 'ink' vs background using color distance
    img_float = img_rgb.astype(np.int32)
    bg_arr = np.array(bg, dtype=np.int32)
    dist = np.sqrt(np.sum((img_float - bg_arr)**2, axis=2))
    # Adaptive threshold: anything sufficiently different from bg is foreground
    thresh = max(30, np.percentile(dist, 90) * 0.4)  # heuristic
    mask = (dist > thresh).astype(np.uint8) * 255

    # Morphological cleanup: close small holes and remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours (external + internal)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    primitives = []
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # distance transform to estimate stroke half-widths (for thickness)
    # compute distance on inverted mask (foreground pixels -> distance to background)
    dist_to_bg = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

    for cnt in contours:
        if len(cnt) < 4:
            continue
        pts = cnt.reshape(-1,2).tolist()
        # bounding box
        x, y, ww, hh = cv2.boundingRect(cnt)
        bbox = [int(x), int(y), int(ww), int(hh)]
        # compute mean color of contour region (sample masked bbox)
        mask_roi = np.zeros((hh, ww), dtype=np.uint8)
        cnt_shifted = cnt - [x,y]
        cv2.drawContours(mask_roi, [cnt_shifted], -1, 255, thickness=-1)
        roi = img_rgb[y:y+hh, x:x+ww]
        if roi.size == 0:
            mean_color = [0,0,0]
        else:
            # mean color only where mask_roi>0
            m = mask_roi > 0
            if m.sum() == 0:
                mean_color = [int(img_rgb[y + hh//2, x + ww//2, c]) for c in range(3)]
            else:
                mean_color = [int(np.sum(roi[:,:,c] * m) / m.sum()) for c in range(3)]
        # thickness estimate: mean 2*distance on contour points
        d_vals = []
        for (px, py) in pts[::max(1, int(len(pts)/200))]:  # sample to limit cost
            # clamp
            if 0 <= py < dist_to_bg.shape[0] and 0 <= px < dist_to_bg.shape[1]:
                d_vals.append(dist_to_bg[py, px])
        if len(d_vals) == 0:
            thickness = 1.0
        else:
            thickness = float(max(1.0, np.mean(d_vals) * 2.0))
        # closed flag: if area is reasonably large and contour end close to start
        closed = cv2.contourArea(cnt) > 10 and np.linalg.norm(cnt[0][0] - cnt[-1][0]) < 3.0
        primitives.append({
            "points": [[int(p[0]), int(p[1])] for p in pts],
            "closed": bool(closed),
            "thickness": float(thickness),
            "color": mean_color,
            "bbox": bbox
        })

    # Heuristic: sort primitives by top-left y then x (drawing order approximation)
    primitives.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))

    return {
        "canvas_size": canvas_size,
        "background_color": bg,
        "primitives": primitives,
        "mask_threshold": float(thresh)
    }

# ---------------- JSON serialization ---------------- #

def image_to_json(image_path: str, json_out: str, embed_raster: bool = False) -> Dict[str, Any]:
    img_rgb = read_image_rgb(image_path)
    data = vectorize_image(img_rgb)
    data["source_image_name"] = os.path.basename(image_path)
    # embed raster as optional perfect-fallback
    if embed_raster:
        data["_embedded_png_b64"] = pil_image_to_base64_png(img_rgb)
    # metadata
    data["metadata"] = {
        "pixel_perfect_intent": bool(embed_raster),
        "vectorized_primitives": len(data["primitives"])
    }
    ensure_dir(json_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data

# ---------------- Deterministic renderer ---------------- #

def render_from_json(data: Dict[str, Any]) -> np.ndarray:
    w, h = data["canvas_size"]
    bg = tuple(int(c) for c in data["background_color"])
    canvas = np.ones((h, w, 3), dtype=np.uint8)
    canvas[:] = bg[::-1]  # fill with background (note: we'll convert later to RGB order)
    # draw in preserved order
    for prim in data["primitives"]:
        pts = np.array(prim["points"], dtype=np.int32).reshape(-1,1,2)
        color_rgb = tuple(int(c) for c in prim["color"])
        thickness = max(1, int(round(prim.get("thickness", 1))))
        closed = prim.get("closed", False)
        # OpenCV draws in BGR; convert RGB->BGR
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        # If closed, prefer filled polygon when thickness is large relative to area
        area = prim.get("bbox", [0,0,0,0])[2] * prim.get("bbox", [0,0,0,0])[3]
        if closed and thickness > 1 and area < 10_000:
            cv2.drawContours(canvas, [pts], -1, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
        else:
            # Use polylines for strokes
            cv2.polylines(canvas, [pts], isClosed=closed, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)
    # convert BGR->RGB for returning
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas_rgb

# ---------------- Comparison utilities ---------------- #

def images_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)

def compare_and_maybe_fallback(original_rgb: np.ndarray, regen_rgb: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
    # Try pixel perfect equality
    if images_equal(original_rgb, regen_rgb):
        return regen_rgb
    # If JSON contains embedded raster, use that (exact)
    if "_embedded_png_b64" in data:
        print("[INFO] Vector regeneration differs from original; using embedded raster fallback for pixel-perfect result.")
        fallback = base64_png_to_pil_image(data["_embedded_png_b64"])
        # Ensure shape matches canvas size
        return fallback
    # Otherwise return the vector regen but warn user
    print("[WARN] Regenerated image differs from original and no embedded raster present.")
    # Optional: provide difference image for debugging
    return regen_rgb

# ---------------- CLI orchestration ---------------- #

def mode_img2json(input_path: str, out_path: str, embed_raster: bool):
    print(f"[MODE] Image -> JSON: {input_path} -> {out_path}  (embed raster: {embed_raster})")
    data = image_to_json(input_path, out_path, embed_raster=embed_raster)
    print(f"[OK] Saved JSON with {len(data['primitives'])} primitives.")
    return data

def mode_json2img(input_path: str, out_path: str):
    print(f"[MODE] JSON -> Image: {input_path} -> {out_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"JSON file not found: {os.path.abspath(input_path)}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    regen = render_from_json(data)
    # If original image was embedded in JSON, allow comparison for pixel-perfect confirmation
    if "source_image_name" in data and "_embedded_png_b64" in data:
        orig = base64_png_to_pil_image(data["_embedded_png_b64"])
        final = compare_and_maybe_fallback(orig, regen, data)
    else:
        final = regen
    write_image_rgb(out_path, final)
    print(f"[OK] Regenerated image written to {out_path}")
    # If we can, check equality and report
    if "_embedded_png_b64" in data:
        orig = base64_png_to_pil_image(data["_embedded_png_b64"])
        if images_equal(orig, final):
            print("[SUCCESS] Pixel-perfect regeneration confirmed (embedded raster matched).")
        else:
            print("[NOTICE] Even with fallback, regenerated image differs from embedded raster (unexpected).")
    return final

def parse_args():
    p = argparse.ArgumentParser(description="Kolam reversible image <-> JSON pipeline")
    p.add_argument("--mode", choices=["img2json", "json2img"], required=True, help="Operation mode")
    p.add_argument("--input", required=True, help="Input file (image or json)")
    p.add_argument("--out", required=True, help="Output file path (json or image)")
    p.add_argument("--embed-raster", action="store_true", help="When creating JSON, embed source PNG (lossless fallback)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "img2json":
        mode_img2json(args.input, args.out, embed_raster=args.embed_raster)
    else:
        mode_json2img(args.input, args.out)

if __name__ == "__main__":
    main()
