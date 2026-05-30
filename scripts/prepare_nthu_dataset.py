from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path
    relative_parent: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce and split the NTHU drowsiness dataset into a training-ready folder layout."
    )
    parser.add_argument("--source-root", type=Path, default=Path("NTHU_Dataset"), help="Raw dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared_nthu_dataset"),
        help="Destination folder for the reduced dataset.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=1000,
        help="Maximum number of images to keep from each source group.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned counts without writing files.")
    return parser.parse_args()


def collect_items(source_root: Path) -> dict[str, list[DatasetItem]]:
    grouped: dict[str, list[DatasetItem]] = {}
    for image_path in sorted(source_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parent = image_path.parent.relative_to(source_root)
        group_name = relative_parent.as_posix()
        grouped.setdefault(group_name, []).append(
            DatasetItem(
                image_path=image_path,
                label_path=image_path.with_suffix(".txt"),
                relative_parent=relative_parent,
            )
        )
    return grouped


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = round(train_ratio + val_ratio + test_ratio, 6)
    if total != 1:
        raise SystemExit("Split ratios must add up to 1.0.")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise SystemExit("Split ratios must be non-negative.")


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if test_count < 0:
        test_count = 0

    assigned = train_count + val_count + test_count
    while assigned < total:
        remainder_candidates = [
            (total * train_ratio - train_count, "train"),
            (total * val_ratio - val_count, "val"),
            (total * test_ratio - test_count, "test"),
        ]
        remainder_candidates.sort(reverse=True)
        target = remainder_candidates[0][1]
        if target == "train":
            train_count += 1
        elif target == "val":
            val_count += 1
        else:
            test_count += 1
        assigned += 1

    return train_count, val_count, test_count


def copy_or_move(source: Path, destination: Path, *, move: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)


def write_empty_file(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("", encoding="utf-8")


def build_selection(
    grouped_items: dict[str, list[DatasetItem]],
    *,
    max_per_group: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, dict[str, object]], dict[str, list[DatasetItem]]]:
    rng = random.Random(seed)
    selected_plan: dict[str, dict[str, object]] = {}
    split_buckets: dict[str, list[DatasetItem]] = {"train": [], "val": [], "test": []}

    for group_name, items in sorted(grouped_items.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        selected = shuffled[: min(len(shuffled), max_per_group)]
        train_count, val_count, test_count = split_counts(len(selected), train_ratio, val_ratio, test_ratio)

        selected_plan[group_name] = {
            "available": len(items),
            "selected": len(selected),
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }

        split_buckets["train"].extend(selected[:train_count])
        split_buckets["val"].extend(selected[train_count : train_count + val_count])
        split_buckets["test"].extend(
            selected[train_count + val_count : train_count + val_count + test_count]
        )

    return selected_plan, split_buckets


def serialize_summary(plan: dict[str, dict[str, object]], splits: dict[str, list[DatasetItem]]) -> dict[str, object]:
    return {
        "groups": {
            group_name: {
                key: value
                for key, value in stats.items()
                if key != "items"
            }
            for group_name, stats in plan.items()
        },
        "splits": {split_name: len(items) for split_name, items in splits.items()},
    }


def main() -> None:
    args = parse_args()
    if not args.source_root.exists():
        raise SystemExit(f"Source root does not exist: {args.source_root}")

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    grouped_items = collect_items(args.source_root)
    selected_plan, split_buckets = build_selection(
        grouped_items,
        max_per_group=args.max_per_group,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    summary = serialize_summary(selected_plan, split_buckets)

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    if args.output_root.exists():
        raise SystemExit(
            f"Output root already exists: {args.output_root}. Remove it or choose a new output path."
        )

    for split_name, split_items in split_buckets.items():
        for item in split_items:
            image_destination = args.output_root / "images" / split_name / item.relative_parent / item.image_path.name
            label_destination = args.output_root / "labels" / split_name / item.relative_parent / item.label_path.name
            copy_or_move(item.image_path, image_destination, move=args.move)
            if item.label_path.exists():
                copy_or_move(item.label_path, label_destination, move=args.move)
            else:
                write_empty_file(label_destination)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.output_root / "selection_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["group", "available", "selected", "train", "val", "test"])
        for group_name, stats in sorted(selected_plan.items()):
            writer.writerow(
                [
                    group_name,
                    stats["available"],
                    stats["selected"],
                    stats["train"],
                    stats["val"],
                    stats["test"],
                ]
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()