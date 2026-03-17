"""
Convert vipe output (pose npz + rgb mp4) into YUME 1.5 training format.

Input:  vipe_results/
            pose/{video_name}.npz   (N,4,4) c2w matrices
            rgb/{video_name}.mp4    original video (H264)

Output: {output_dir}/
            mp4_frame/Keys_{keys}_Mouse_{mouse}/
                {video_id}_{seg_start}_{seg_end}_frames_{start:05d}-{end:05d}.mp4
                {video_id}_{seg_start}_{seg_end}_frames_{start:05d}-{end:05d}.txt
                {video_id}_{seg_start}_{seg_end}_frames_{start:05d}-{end:05d}.npy
            Sekai/{video_id}/
                {video_id}_{seg_start}_{seg_end}.mp4   (symlink to original)

File naming follows the Sekai convention so that YUME's dataset loader
can resolve full_mp4 paths for frame packing (v2v training mode).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# ── Arrow symbol ↔ English folder name mapping ──────────────────────────
MOUSE_SYMBOL_TO_FOLDER = {
    "→": "Right",
    "←": "Left",
    "↑": "Up",
    "↓": "Down",
    "↑→": "Up_Right",
    "↑←": "Up_Left",
    "↓→": "Down_Right",
    "↓←": "Down_Left",
    "·": "·",
}

KEYS_TO_FOLDER = {
    "W": "W",
    "S": "S",
    "A": "A",
    "D": "D",
    "W+A": "W_A",
    "W+D": "W_D",
    "S+A": "S_A",
    "S+D": "S_D",
    "None": "None",
}


def decode_camera_controls(cam_c2w, stride=1,
                           translation_threshold=0.0001,
                           rotation_threshold=0.001):
    """Decode c2w sequence into per-frame WASD + mouse direction labels."""
    c2w_matrices = cam_c2w[::stride]
    t_thresh = translation_threshold * stride
    r_thresh = rotation_threshold * stride

    controls = []
    for i in range(len(c2w_matrices) - 1):
        T_rel = np.linalg.inv(c2w_matrices[i]) @ c2w_matrices[i + 1]
        t_rel = T_rel[:3, 3]
        R_rel = T_rel[:3, :3]

        keys = []
        x_move, _, z_move = t_rel
        if z_move > t_thresh:
            keys.append("W")
        if z_move < -t_thresh:
            keys.append("S")
        if x_move > t_thresh:
            keys.append("D")
        if x_move < -t_thresh:
            keys.append("A")
        key_cmd = "+".join(keys) if keys else "None"

        roc = Rotation.from_matrix(R_rel).as_euler("xyz", degrees=False)
        mouse_h = None
        mouse_v = None
        if roc[1] > r_thresh:
            mouse_h = "→"
        elif roc[1] < -r_thresh:
            mouse_h = "←"
        if roc[0] > r_thresh:
            mouse_v = "↑"
        elif roc[0] < -r_thresh:
            mouse_v = "↓"

        if mouse_h and mouse_v:
            mouse_dir = mouse_v + mouse_h
        else:
            mouse_dir = mouse_h or mouse_v or "·"

        controls.append({"frame": i, "keys": key_cmd, "mouse": mouse_dir})
    return controls


def majority_vote(controls):
    """Return the dominant (keys, mouse) label from a list of per-frame controls."""
    keys_label = Counter(c["keys"] for c in controls).most_common(1)[0][0]
    mouse_label = Counter(c["mouse"] for c in controls).most_common(1)[0][0]
    return keys_label, mouse_label


def get_video_info(video_path):
    """Return (num_frames, fps, width, height) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height",
        "-of", "default=noprint_wrappers=1", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or "nb_frames" not in result.stdout:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {result.stderr.strip()}"
        )

    info = {}
    for line in result.stdout.strip().split("\n"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k] = v

    missing = [k for k in ("nb_frames", "r_frame_rate", "width", "height") if k not in info]
    if missing:
        raise RuntimeError(
            f"ffprobe output missing fields {missing} for {video_path}. "
            f"stdout: {result.stdout!r}  stderr: {result.stderr!r}"
        )

    num_frames = int(info["nb_frames"])
    fps_parts = info["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1])
    width = int(info["width"])
    height = int(info["height"])
    return num_frames, fps, width, height


def extract_clip_ffmpeg(src_video, dst_video, start_frame, num_frames, fps):
    """Extract a clip from src_video using ffmpeg.

    Writes to /tmp first then moves, to avoid 'moov atom not found' errors
    on network filesystems that don't support certain seek operations.
    """
    start_time = start_frame / fps
    duration = num_frames / fps

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start_time:.6f}",
            "-i", str(src_video),
            "-t", f"{duration:.6f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            tmp_path,
        ]
        subprocess.run(cmd, check=True)
        shutil.move(tmp_path, str(dst_video))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def setup_sekai_dir(video_name, video_path, num_frames, output_dir):
    """Create Sekai/{video_id}/{video_id}_{start}_{end}.mp4 that points to the
    original full video, matching the naming convention YUME's loader expects.

    The loader resolves full_mp4 as:
        Sekai/{video_id}/{video_id}_{seg_start}_{seg_end}.mp4
    where video_id = clip_basename.split('_')[:-2] joined by '_',
    and seg part    = clip_basename.split('_frames_')[0].

    We use video_name as video_id and fake segment bounds 0000000_{num_frames:07d}.
    """
    seg_start = "0000000"
    seg_end = f"{num_frames:07d}"
    seg_name = f"{video_name}_{seg_start}_{seg_end}"

    sekai_dir = Path(output_dir) / "Sekai" / video_name
    sekai_dir.mkdir(parents=True, exist_ok=True)

    dst = sekai_dir / f"{seg_name}.mp4"
    if not dst.exists():
        src = Path(video_path).resolve()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(str(src), str(dst))

    return seg_name


def process_one_video(video_name, vipe_dir, output_dir, clip_length, clip_stride,
                      translation_threshold, rotation_threshold, rgb_dir=None):
    """Process a single video: load pose, slice clips, decode controls, write output."""
    pose_path = Path(vipe_dir) / "pose" / f"{video_name}.npz"
    if rgb_dir is not None:
        video_path = Path(rgb_dir) / f"{video_name}.mp4"
    else:
        video_path = Path(vipe_dir) / "rgb" / f"{video_name}.mp4"

    if not pose_path.exists():
        print(f"[SKIP] pose not found: {pose_path}")
        return 0
    if not video_path.exists():
        print(f"[SKIP] video not found: {video_path}")
        return 0

    pose_data = np.load(pose_path)
    c2w_all = pose_data["data"]       # (N, 4, 4)
    inds_all = pose_data["inds"]      # (N,)

    try:
        num_frames, fps, width, height = get_video_info(video_path)
    except RuntimeError as e:
        print(f"[SKIP] broken video {video_path}: {e}")
        return 0
    total_pose_frames = len(c2w_all)

    print(f"[{video_name}] frames={num_frames}, poses={total_pose_frames}, "
          f"fps={fps}, res={width}x{height}")

    seg_name = setup_sekai_dir(video_name, video_path, num_frames, output_dir)

    clip_count = 0
    start = 0
    while start + clip_length <= total_pose_frames:
        end = start + clip_length
        c2w_clip = c2w_all[start:end]
        inds_clip = inds_all[start:end]

        controls = decode_camera_controls(
            c2w_clip, stride=1,
            translation_threshold=translation_threshold,
            rotation_threshold=rotation_threshold,
        )

        if not controls:
            start += clip_stride
            continue

        keys_label, mouse_label = majority_vote(controls)

        keys_folder = KEYS_TO_FOLDER.get(keys_label, keys_label.replace("+", "_"))
        mouse_folder = MOUSE_SYMBOL_TO_FOLDER.get(mouse_label, mouse_label)
        folder_name = f"Keys_{keys_folder}_Mouse_{mouse_folder}"

        clip_dir = Path(output_dir) / "mp4_frame" / folder_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Naming: {video_id}_{seg_start}_{seg_end}_frames_{clipStart}-{clipEnd}
        # so YUME's loader can do split('_frames_')[0] → seg_name,
        # then split('_')[:-2] → video_id → Sekai/{video_id}/
        base_name = f"{seg_name}_frames_{int(inds_clip[0]):05d}-{int(inds_clip[-1]):05d}"
        clip_mp4 = clip_dir / f"{base_name}.mp4"
        clip_txt = clip_dir / f"{base_name}.txt"
        clip_npy = clip_dir / f"{base_name}.npy"

        extract_clip_ffmpeg(video_path, clip_mp4, int(inds_clip[0]), clip_length, fps)

        with open(clip_txt, "w") as f:
            f.write(f"Start Frame: {int(inds_clip[0])}\n")
            f.write(f"End Frame: {int(inds_clip[-1])}\n")
            f.write(f"Duration: {clip_length} frames\n")
            f.write(f"Keys: {keys_label}\n")
            f.write(f"Mouse: {mouse_label}\n")

        np.save(clip_npy, c2w_clip.astype(np.float64))

        clip_count += 1
        start += clip_stride

    return clip_count


def print_stats(output_dir):
    """Print distribution of clips across categories."""
    mp4_frame = Path(output_dir) / "mp4_frame"
    if not mp4_frame.exists():
        return
    print("\n" + "=" * 60)
    print("Category distribution:")
    print("=" * 60)
    total = 0
    for d in sorted(mp4_frame.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.mp4")))
            if n > 0:
                print(f"  {d.name}: {n} clips")
                total += n
    print(f"\nTotal: {total} clips")


def main():
    parser = argparse.ArgumentParser(
        description="Convert vipe output to YUME 1.5 training format")
    parser.add_argument("--vipe_dir", type=str, required=True,
                        help="Path to vipe_results/ directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for YUME training data")
    parser.add_argument("--clip_length", type=int, default=49,
                        help="Number of frames per clip (default: 49, = 4k+1 for HunyuanVideo VAE)")
    parser.add_argument("--clip_stride", type=int, default=16,
                        help="Stride between clip starts (default: 16, ~50%% overlap)")
    parser.add_argument("--translation_threshold", type=float, default=0.0001,
                        help="Threshold for WASD movement detection (default: 0.0001)")
    parser.add_argument("--rotation_threshold", type=float, default=0.001,
                        help="Threshold for mouse rotation detection (default: 0.001)")
    parser.add_argument("--rgb_dir", type=str, default=None,
                        help="Optional separate directory containing source RGB videos "
                             "(overrides {vipe_dir}/rgb/). Use when source videos live "
                             "outside the vipe output tree (e.g. on a readable FS).")
    args = parser.parse_args()

    pose_dir = Path(args.vipe_dir) / "pose"
    if not pose_dir.exists():
        print(f"ERROR: pose directory not found: {pose_dir}")
        sys.exit(1)

    video_names = sorted([
        p.stem for p in pose_dir.glob("*.npz")
    ])
    print(f"Found {len(video_names)} videos with pose data: {video_names}")

    total_clips = 0
    for name in video_names:
        n = process_one_video(
            name, args.vipe_dir, args.output_dir,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            translation_threshold=args.translation_threshold,
            rotation_threshold=args.rotation_threshold,
            rgb_dir=args.rgb_dir,
        )
        print(f"  -> {n} clips extracted from {name}")
        total_clips += n

    print_stats(args.output_dir)


if __name__ == "__main__":
    main()
