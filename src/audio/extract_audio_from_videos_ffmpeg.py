import os
import sys
sys.path.append('src')

import subprocess
from tqdm import tqdm
from typing import List, Dict, Any


from audio.utils.utils import load_config


def extract_audio_ffmpeg(video_path: str, audio_path: str, config: Dict[str, Any]) -> None:
    """
    Extract audio from a video file using ffmpeg.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    # Choose audio codec based on desired format
    codec = "pcm_s16le" if config["audio_format"] == "wav" else "libmp3lame"

    # ffmpeg command
    command = [
        "ffmpeg",
        "-i", video_path,             # input file
        "-async", str(1),                  # sync audio and video
        "-vn",                        # no video
        "-acodec", codec,             # audio codec
        "-ar", str(config["sample_rate"]),  # sample rate
        "-ac", str(config["channels"]),     # number of channels
        audio_path,                   # output file
        "-y"                          # overwrite without prompt
    ]

    # print(' '.join(command))
    # Run the command silently
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Failed to extract from: {video_path}")


def get_all_video_files(config: Dict[str, Any]) -> List[str]:
    """
    Recursively find all video files in a directory with specified extensions.
    """
    video_files = []
    src_root = config["source_dir"]
    splits = config["splits"]
    valid_exts = tuple(config["video_extensions"])

    for dataset in os.listdir(src_root):
        for split in splits:
            target_dir = os.path.join(src_root, dataset, split)
            if not os.path.isdir(target_dir):
                continue
            for dirpath, _, filenames in os.walk(target_dir):
                for file in filenames:
                    if file.lower().endswith(valid_exts):
                        full_path = os.path.join(dirpath, file)
                        video_files.append(full_path)
    
    return video_files


def process_videos(config: Dict[str, Any]) -> None:
    """
    Process all video files according to configuration and extract audio.
    """
    # Load configuration values
    src_root = config["source_dir"]
    dst_root = config["output_audio_dir"]
    audio_ext = config["audio_format"]

    # Gather all video files
    video_files = get_all_video_files(config)

    # Process with a progress bar
    for video_path in tqdm(video_files, desc="Extracting audio"):
        # Maintain directory structure for output
        rel_path = os.path.relpath(video_path, src_root)
        rel_audio_path = os.path.splitext(rel_path)[0] + f".{audio_ext}"
        audio_path = os.path.join(dst_root, rel_audio_path)

        # Extract audio
        extract_audio_ffmpeg(video_path, audio_path, config)


if __name__ == "__main__":
    config = load_config("config.yaml")
    process_videos(config)