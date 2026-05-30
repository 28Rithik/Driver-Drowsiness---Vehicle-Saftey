from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove colored bounding-box overlays from NTHU dataset images using HSV masking and inpainting."
    )
    parser.add_argument("--source-root", type=Path, default=Path("NTHU_Dataset"), help="Raw dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cleaned_nthu_dataset"),
        help="Destination folder for cleaned images and copied labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many files would be processed without writing anything.",
    )
    parser.add_argument("--inpaint-radius", type=int, default=2, help="Inpainting radius in pixels.")
    parser.add_argument(
        "--inpaint-method",
        choices=["telea", "ns", "median"],
        default="median",
        help="Cleanup method. median is usually sharpest for grayscale driver frames.",
    )
    parser.add_argument(
        "--label-padding",
        type=int,
        default=3,
        help="Extra pixels added around label boxes before masking.",
    )
    parser.add_argument(
        "--label-thickness",
        type=int,
        default=6,
        help="Thickness of the masked box outline in pixels.",
    )
    return parser.parse_args()


def collect_images(source_root: Path) -> list[Path]:
    return [path for path in sorted(source_root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def build_overlay_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue_channel, green_channel, red_channel = cv2.split(image_bgr)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    hsv_colored_pixels = (saturation > 70) & (value > 50)
    red_pixels = (hue <= 12) | (hue >= 168)
    green_pixels = (hue >= 40) & (hue <= 85)
    blue_pixels = (hue >= 100) & (hue <= 140)
    dominance_pixels = (
        ((blue_channel > 90) & (blue_channel > green_channel + 20) & (blue_channel > red_channel + 20))
        | ((green_channel > 90) & (green_channel > blue_channel + 20) & (green_channel > red_channel + 20))
        | ((red_channel > 90) & (red_channel > green_channel + 20) & (red_channel > blue_channel + 20))
    )

    mask = (hsv_colored_pixels & (red_pixels | green_pixels | blue_pixels)) | dominance_pixels
    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def build_label_mask(
    image_path: Path,
    image_shape: tuple[int, int, int],
    *,
    padding: int,
    thickness: int,
) -> np.ndarray:
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return np.zeros(image_shape[:2], dtype=np.uint8)

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        _, center_x, center_y, box_width, box_height = map(float, parts)
        x1 = max(0, int(round((center_x - box_width / 2) * width)) - padding)
        y1 = max(0, int(round((center_y - box_height / 2) * height)) - padding)
        x2 = min(width - 1, int(round((center_x + box_width / 2) * width)) + padding)
        y2 = min(height - 1, int(round((center_y + box_height / 2) * height)) + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=thickness)

    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def clean_image(
    image_path: Path,
    output_path: Path,
    *,
    inpaint_radius: int,
    inpaint_method: str,
    label_padding: int,
    label_thickness: int,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    mask = build_overlay_mask(image_bgr)
    label_mask = build_label_mask(
        image_path,
        image_bgr.shape,
        padding=label_padding,
        thickness=label_thickness,
    )
    mask = cv2.bitwise_or(mask, label_mask)
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if inpaint_method == "median":
        smoothed = cv2.medianBlur(grayscale, 5)
        cleaned_gray = grayscale.copy()
        cleaned_gray[mask > 0] = smoothed[mask > 0]
    else:
        method = cv2.INPAINT_NS if inpaint_method == "ns" else cv2.INPAINT_TELEA
        cleaned_gray = cv2.inpaint(grayscale, mask, inpaint_radius, method)
    cleaned = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cleaned):
        raise RuntimeError(f"Unable to write image: {output_path}")


def copy_label(image_path: Path, source_root: Path, output_root: Path) -> None:
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return

    relative_parent = image_path.parent.relative_to(source_root)
    destination = output_root / relative_parent / label_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, destination)


def main() -> None:
    args = parse_args()
    if not args.source_root.exists():
        raise SystemExit(f"Source root does not exist: {args.source_root}")

    images = collect_images(args.source_root)
    summary = {"images": len(images), "labels": 0 if args.dry_run else len(images)}

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    if args.output_root.exists():
        raise SystemExit(f"Output root already exists: {args.output_root}. Remove it or choose a new output path.")

    for image_path in images:
        relative_parent = image_path.parent.relative_to(args.source_root)
        output_image_path = args.output_root / relative_parent / image_path.name
        clean_image(
            image_path,
            output_image_path,
            inpaint_radius=args.inpaint_radius,
            inpaint_method=args.inpaint_method,
            label_padding=args.label_padding,
            label_thickness=args.label_thickness,
        )
        copy_label(image_path, args.source_root, args.output_root)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()