"""Generate an HTML visualization of batch inference results.

For each sample, shows: first-frame image | TSV prompt | longest generated video (inline playable).
Videos are base64-encoded so the HTML is self-contained and works offline.

Usage:
    python fastvideo/sample/visualize_batch_results.py \
        --video_dir  ./outputs/stage1_val_batch1000step_autocaption_3.7 \
        --tsv_path   .../world_model_action12_inference_240.tsv \
        --first_frame_dir .../first_frame \
        --output     ./outputs/stage1_val_batch1000step_autocaption_3.7/results.html
"""

import argparse
import base64
import csv
import os
import re
from collections import defaultdict


def encode_file_b64(path, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def find_longest_videos(video_dir):
    """Group mp4 files by sample id (everything before the last _N.mp4) and pick the one with the largest step number."""
    pat = re.compile(r'^(.+?)_(\d+)\.mp4$')
    groups = defaultdict(list)
    for fname in os.listdir(video_dir):
        if not fname.endswith('.mp4'):
            continue
        m = pat.match(fname)
        if m:
            base_id, step = m.group(1), int(m.group(2))
            groups[base_id].append((step, fname))

    result = {}
    for base_id, items in groups.items():
        items.sort(key=lambda x: x[0])
        longest_fname = items[-1][1]
        result[base_id] = longest_fname
    return result


def load_tsv_prompts(tsv_path):
    """Return {video_stem: prompt} mapping from TSV."""
    prompts = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            stem = os.path.splitext(row['video_name'])[0]
            prompts[stem] = row['prompt']
    return prompts


def build_html(video_dir, tsv_path, first_frame_dir, output_path):
    longest = find_longest_videos(video_dir)
    prompts = load_tsv_prompts(tsv_path)

    video_stem_to_base = {}
    for base_id, fname in longest.items():
        parts = base_id.split('_')
        for i in range(1, len(parts)):
            candidate = '_'.join(parts[:i])
            if candidate in prompts:
                video_stem_to_base[base_id] = candidate
                break

    rows_html = []
    seen_stems = set()

    for base_id in sorted(longest.keys()):
        video_fname = longest[base_id]
        video_path = os.path.join(video_dir, video_fname)

        stem = video_stem_to_base.get(base_id, base_id)
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        prompt = prompts.get(stem, "N/A")
        img_path = os.path.join(first_frame_dir, stem + '.jpg')
        if not os.path.isfile(img_path):
            img_path = os.path.join(first_frame_dir, stem + '.png')

        caption_path = os.path.join(video_dir, os.path.splitext(video_fname)[0] + '_caption.txt')
        caption_text = ""
        if os.path.isfile(caption_path):
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_text = f.read()

        img_b64 = encode_file_b64(img_path, "image/jpeg") if os.path.isfile(img_path) else ""
        vid_b64 = encode_file_b64(video_path, "video/mp4")

        caption_block = ""
        if caption_text:
            escaped = caption_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            caption_block = f'<details><summary>Generated Caption ({len(caption_text.splitlines())} steps)</summary><pre>{escaped}</pre></details>'

        rows_html.append(f"""
        <tr>
            <td class="img-cell">
                {'<img src="' + img_b64 + '">' if img_b64 else '<span class="na">No image</span>'}
            </td>
            <td class="prompt-cell">
                <div class="sample-id">{stem}</div>
                <div class="prompt">{prompt.replace('<', '&lt;').replace('>', '&gt;')}</div>
                {caption_block}
            </td>
            <td class="video-cell">
                <video controls preload="metadata">
                    <source src="{vid_b64}" type="video/mp4">
                </video>
                <div class="fname">{video_fname}</div>
            </td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Batch Inference Results</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }}
    h1 {{ text-align: center; margin-bottom: 8px; font-size: 22px; color: #fff; }}
    .meta {{ text-align: center; margin-bottom: 24px; color: #888; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #1a1a1a; padding: 12px 16px; text-align: left; font-size: 13px; color: #aaa; border-bottom: 2px solid #333; }}
    tr {{ border-bottom: 1px solid #222; }}
    tr:hover {{ background: #1a1a1a; }}
    td {{ padding: 16px; vertical-align: top; }}
    .img-cell img {{ width: 240px; border-radius: 4px; }}
    .prompt-cell {{ max-width: 520px; }}
    .sample-id {{ font-weight: 700; font-size: 14px; color: #7cb3ff; margin-bottom: 6px; }}
    .prompt {{ font-size: 13px; line-height: 1.5; color: #ccc; }}
    .video-cell video {{ width: 420px; border-radius: 4px; }}
    .fname {{ font-size: 11px; color: #666; margin-top: 4px; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; font-size: 12px; color: #888; }}
    pre {{ font-size: 11px; color: #999; white-space: pre-wrap; margin-top: 4px; max-height: 200px; overflow-y: auto; background: #151515; padding: 8px; border-radius: 4px; }}
    .na {{ color: #555; }}
</style>
</head>
<body>
<h1>Batch Inference Results</h1>
<div class="meta">{len(rows_html)} samples &bull; {video_dir}</div>
<table>
<thead><tr><th>First Frame</th><th>Prompt</th><th>Generated Video (longest)</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML saved to {output_path} ({len(rows_html)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory containing generated .mp4 files")
    parser.add_argument("--tsv_path", required=True, help="TSV file with action_class/prompt/video_name columns")
    parser.add_argument("--first_frame_dir", required=True, help="Directory containing first-frame .jpg files")
    parser.add_argument("--output", default=None, help="Output HTML path (default: video_dir/results.html)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.video_dir, "results.html")

    build_html(args.video_dir, args.tsv_path, args.first_frame_dir, args.output)
