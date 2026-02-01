#!/usr/bin/env python3
"""
Add AI-generated video summaries to caption JSON files.
Reads frame captions and generates a concise summary using a local LLM.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def format_time(seconds):
    """Format time position as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def create_summary_prompt(captions_data):
    """Create a concise summary prompt from frame captions"""
    captions = captions_data["captions"]
    num_frames = len(captions)

    # Determine paragraph count based on video length
    if num_frames <= 20:
        paragraph_guide = "Write 2-3 sentences in a single paragraph"
    elif num_frames <= 60:
        paragraph_guide = "Write 1-2 short paragraphs (3-5 sentences total)"
    else:
        paragraph_guide = "Write 2-3 short paragraphs (5-8 sentences total)"

    prompt = f"""You are analyzing a video. Below are frame descriptions captured every 30 seconds.

Write a brief, concise summary of what happens in this video. Focus on the key people, activities, and events. Be factual and efficient - this summary will help organize videos in a library.

IMPORTANT: {paragraph_guide}. Do NOT write detailed descriptions. Just the essential information.

Frame descriptions:
"""

    for cap in captions:
        time_str = format_time(cap["time_sec"])
        prompt += f"[{time_str}] {cap['caption']}\n"

    prompt += "\nSummary:"

    return prompt


def generate_summary(prompt, model_name):
    """Generate summary using Ollama"""
    print(f"[INFO] Generating summary with {model_name}...")

    start_time = time.time()

    # Call Ollama with the prompt
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "--verbose"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"[ERROR] Ollama failed: {result.stderr}")
            return None, 0

        summary = result.stdout.strip()
        elapsed = time.time() - start_time

        print(f"[INFO] Summary generated in {elapsed:.2f}s ({len(summary.split())} words)")

        return summary, elapsed

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Model {model_name} timed out after 5 minutes")
        return None, 0
    except FileNotFoundError:
        print("[ERROR] Ollama not found. Please install Ollama first.")
        return None, 0
    except Exception as e:
        print(f"[ERROR] Failed to generate summary: {e}")
        return None, 0


def main():
    parser = argparse.ArgumentParser(
        description="Add AI-generated summaries to video caption JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video caption file (uses llama3.2 by default)
  python add_video_summary.py video_captions.json
  
  # Use a different model
  python add_video_summary.py video_captions.json --model llama3.1
  
  # Process all caption JSON files in current directory
  python add_video_summary.py *.json
  
Available models (install via Ollama):
  llama3.2 (default) - Fast, concise (3B params, ~5s per video)
  llama3.1          - More detailed (8B params, ~13s per video)
  phi3              - Alternative (3.8B params, ~8s per video)
  mistral           - Alternative (7B params, ~15s per video)
  qwen2.5           - Alternative (7B params, ~21s per video)
""",
    )

    parser.add_argument("json_files", nargs="+", type=str, help="Caption JSON file(s) to process")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="LLM model to use for summary generation (default: llama3.2)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing summary if present")

    args = parser.parse_args()

    # Process each JSON file
    success_count = 0
    skip_count = 0
    error_count = 0
    total_time = 0

    for json_path_str in args.json_files:
        json_path = Path(json_path_str)

        if not json_path.exists():
            print(f"[WARN] File not found: {json_path}")
            error_count += 1
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing: {json_path.name}")
        print(f"{'=' * 80}")

        # Load caption JSON
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read JSON: {e}")
            error_count += 1
            continue

        # Check if summary already exists
        if "summary" in data and not args.overwrite:
            print("[SKIP] Summary already exists (use --overwrite to regenerate)")
            skip_count += 1
            continue

        # Generate summary
        prompt = create_summary_prompt(data)
        summary, elapsed = generate_summary(prompt, args.model)

        if summary is None:
            error_count += 1
            continue

        # Add summary to JSON
        data["summary"] = summary
        data["summary_model"] = args.model
        data["summary_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Save updated JSON
        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[SUCCESS] Updated {json_path.name}")
            success_count += 1
            total_time += elapsed
        except Exception as e:
            print(f"[ERROR] Failed to write JSON: {e}")
            error_count += 1

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Processed: {success_count} files")
    print(f"Skipped:   {skip_count} files (already have summaries)")
    print(f"Errors:    {error_count} files")
    if success_count > 0:
        print(f"Total time: {total_time:.1f}s")
        print(f"Average:    {total_time / success_count:.1f}s per video")
    print()


if __name__ == "__main__":
    main()
