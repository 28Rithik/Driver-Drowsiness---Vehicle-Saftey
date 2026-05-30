from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe-style JSON annotations into text files for drowsiness dataset training."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("prepared_nthu_dataset/annotations/json"),
        help="Folder that contains JSON annotation files.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("prepared_nthu_dataset/labels"),
        help="Folder where .txt label files will be written.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy the matching images into the output folder next to the text labels.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("prepared_nthu_dataset/images"),
        help="Folder that contains the matching image files.",
    )
    return parser.parse_args()


def convert_json_file(json_path: Path, text_dir: Path) -> bool:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    output_path = text_dir / f"{json_path.stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for shape in data.get("shapes", []):
        shape_type = shape.get("shape_type", "polygon")
        label = shape.get("label", "unknown")
        points = shape.get("points", [])

        if shape_type == "polygon" and points:
            normalized_points: list[str] = []
            for x, y in points:
                normalized_points.extend([f"{x / image_width:.6f}", f"{y / image_height:.6f}"])
            lines.append(f"{label} {' '.join(normalized_points)}")
        elif shape_type == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height
            box_width = abs(x2 - x1) / image_width
            box_height = abs(y2 - y1) / image_height
            lines.append(f"{label} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True


def copy_matching_image(json_path: Path, image_root: Path, text_dir: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = data.get("imagePath")
    if not image_name:
        return

    source_image = json_path.parent / image_name
    if not source_image.exists():
        source_image = image_root / image_name

    if source_image.exists():
        target_image = text_dir / source_image.name
        target_image.write_bytes(source_image.read_bytes())


def main() -> None:
    args = parse_args()

    if not args.json_dir.exists():
        raise SystemExit(f"JSON folder does not exist: {args.json_dir}")

    json_files = sorted(args.json_dir.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in: {args.json_dir}")

    args.text_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for json_path in json_files:
        if convert_json_file(json_path, args.text_dir):
            converted += 1
            if args.copy_images:
                copy_matching_image(json_path, args.image_root, args.text_dir)

    print(f"Converted {converted} JSON files into text labels in {args.text_dir}")


if __name__ == "__main__":
    main()