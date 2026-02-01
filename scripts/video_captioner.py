#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


def extract_frames(video_path: str, interval: float, tmpdir: str):
    fps = 1.0 / interval
    out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
    print(f"[INFO] extracting frames from {os.path.basename(video_path)} every {interval}s...")
    cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", out_pattern]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def caption_images(
    frames,
    processor,
    model,
    device,
    interval,
    min_length,
    max_length,
    num_beams,
    use_fp16=False,
):
    captions = []
    print(
        f"[INFO] captioning {len(frames)} frames (min_length={min_length}, max_length={max_length}, num_beams={num_beams})..."
    )
    for i, frame_path in enumerate(frames):
        time_pos = i * interval
        image = Image.open(frame_path).convert("RGB")

        # Move inputs to device with appropriate dtype
        if use_fp16:
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
            )
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"  [frame {i + 1}/{len(frames)}] @{time_pos:.0f}s: {caption}")
        captions.append(
            {
                "frame": os.path.basename(frame_path),
                "time_sec": time_pos,
                "caption": caption,
            }
        )
    return captions


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for video frames using BLIP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python video_captioner.py video.mp4
  
  # Custom frame interval (extract frames every 60 seconds)
  python video_captioner.py video.mp4 --interval 60
  
  # Shorter captions (faster)
  python video_captioner.py video.mp4 --min-length 15 --max-length 50
  
  # Longer, more detailed captions
  python video_captioner.py video.mp4 --min-length 30 --max-length 120
  
  # Use beam search for higher quality (slower)
  python video_captioner.py video.mp4 --num-beams 3
  
  # Use FP16 for faster processing (10% faster, 50% less memory)
  python video_captioner.py video.mp4 --fp16
""",
    )

    parser.add_argument("video_file", type=str, help="Path to video file")
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Frame extraction interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=25,
        help="Minimum caption length in tokens (default: 25, ~20 words)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum caption length in tokens (default: 100, ~80 words)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1=greedy, faster)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 (half precision) for faster inference and lower memory usage. "
        "About 10%% faster and 50%% less memory on Apple Silicon. May have minor quality differences.",
    )

    args = parser.parse_args()

    video_file = Path(args.video_file)
    if not video_file.is_file():
        print(f"[ERROR] File not found: {video_file}")
        sys.exit(1)

    start_time = time.time()
    process_timestamp = datetime.now().isoformat()

    print("[INFO] initializing model on device...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[INFO] using device: {device}")

    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)

    # Load model with optional FP16
    if args.fp16:
        print("[INFO] loading model in FP16 (half precision)")
        model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
    else:
        print("[INFO] loading model in FP32 (full precision)")
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device).eval()

    print(f"[INFO] Processing video: {video_file.name}")
    print(
        f"[INFO] Settings: interval={args.interval}s, min_length={args.min_length}, max_length={args.max_length}, num_beams={args.num_beams}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        extract_frames(str(video_file), args.interval, tmp)
        frame_paths = sorted([os.path.join(tmp, p) for p in os.listdir(tmp) if p.startswith("frame_")])
        caps = caption_images(
            frame_paths,
            processor,
            model,
            device,
            args.interval,
            args.min_length,
            args.max_length,
            args.num_beams,
            use_fp16=args.fp16,
        )

    elapsed_time = time.time() - start_time
    video_out = video_file.parent / (video_file.stem + "_captions.json")
    result = {
        "video": str(video_file),
        "interval_sec": args.interval,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "fp16": args.fp16,
        "processed_at": process_timestamp,
        "processing_time_sec": round(elapsed_time, 2),
        "captions": caps,
    }
    with open(video_out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[SUCCESS] Saved to {video_out} (took {elapsed_time:.1f}s)")


if __name__ == "__main__":
    main()
