from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract still images from driver-monitoring videos for drowsiness training."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("NTHU_Dataset"),
        help="Folder that contains the source videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_nthu_dataset/video_frames"),
        help="Folder where extracted frames will be saved.",
    )
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=0.25,
        help="Save one frame every N seconds.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Optional width to resize frames to. Use 0 to keep original size.",
    )
    return parser.parse_args()


def iter_videos(input_dir: Path) -> list[Path]:
    return [path for path in sorted(input_dir.rglob("*")) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]


def extract_frames(video_path: Path, output_dir: Path, frame_interval_seconds: float, resize_width: int) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * frame_interval_seconds)))

    relative_video_path = video_path.stem
    video_output_dir = output_dir / relative_video_path
    video_output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if resize_width > 0:
            height, width = frame.shape[:2]
            resize_height = int(height * resize_width / width)
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        if frame_index % frame_interval == 0:
            frame_name = f"{video_path.stem}_frame_{saved_count:06d}.jpg"
            frame_path = video_output_dir / frame_name
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Unable to write frame: {frame_path}")
            saved_count += 1

        frame_index += 1

    capture.release()
    return saved_count


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {args.input_dir}")

    videos = iter_videos(args.input_dir)
    if not videos:
        raise SystemExit(f"No videos found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for video_path in videos:
        total_saved += extract_frames(
            video_path,
            args.output_dir,
            frame_interval_seconds=args.frame_interval_seconds,
            resize_width=args.resize_width,
        )

    print(f"Processed {len(videos)} videos and saved {total_saved} frames to {args.output_dir}")


if __name__ == "__main__":
    main()