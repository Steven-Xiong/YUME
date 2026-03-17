#!/usr/bin/env python3
"""
Extract the first frame from each video.
All output frames are saved flat in OUTPUT_DIR.

Structure:
  video/{name}.mp4  ->  first_frame/{name}.jpg

Usage:
  python extract_first_frames.py --video-dir /path/to/video
  python extract_first_frames.py --video-dir /path/to/video --output-dir /path/to/first_frame
  python extract_first_frames.py  # uses default hardcoded paths
"""

import argparse
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_DEFAULT_VIDEO_DIR = "/mnt/bn/voyager-sg-l3/zhexiao.xiong/zhexiao.xiong/data/seadance2_yume_test_12class_simple/video"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract first frame from each video.")
    parser.add_argument("--video-dir", default=_DEFAULT_VIDEO_DIR, help="Directory containing .mp4 videos")
    parser.add_argument("--output-dir", default=None, help="Directory to save extracted frames (default: sibling 'first_frame' dir)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    return parser.parse_args()


def extract_first_frame(video_path: Path, output_path: Path) -> tuple[str, bool, str]:
    """Extract the first frame of a video and save as JPEG. Returns (video_name, success, message)."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return video_path.name, False, "Cannot open video"
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return video_path.name, False, "Cannot read frame"
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return video_path.name, True, ""
    except Exception as e:
        return video_path.name, False, str(e)


def collect_tasks(video_dir: Path, output_dir: Path):
    """Collect (video_path, output_path) pairs for all videos."""
    tasks = []
    for video_path in sorted(video_dir.glob("*.mp4")):
        output_path = output_dir / (video_path.stem + ".jpg")
        tasks.append((video_path, output_path))
    return tasks


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir) if args.output_dir else video_dir.parent / "first_frame"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(video_dir, output_dir)
    print(f"Found {len(tasks)} videos in {video_dir}")

    success_count = 0
    fail_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_first_frame, vp, op): (vp, op)
            for vp, op in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
            name, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append(f"  {name}: {msg}")

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}")
    if errors:
        print("Errors:")
        for e in errors:
            print(e)

    count = len(list(output_dir.glob("*.jpg")))
    print(f"\nOutput: {output_dir}  ({count} frames)")


if __name__ == "__main__":
    main()
