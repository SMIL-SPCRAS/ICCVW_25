#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import glob
import argparse
import subprocess
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
from utils.global_variables import EMOTIONS_EXT, EMO_FOLDER_MAP


def extract_audio(video_path: str, wav_path: str):
    """Extract 16 kHz mono WAV from video if not already present."""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1", "-ar", "16000",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def find_video(video_name: str, search_dirs):
    """
    Recursively search for a file whose base name matches 'video_name'
    under any of the directories in 'search_dirs'. Returns the first match
    or None if not found.
    """
    for base_dir in search_dirs:
        for dirpath, _, files in os.walk(base_dir):
            for fname in files:
                name, ext = os.path.splitext(fname)
                # Compare without extension or with, to be robust
                if name == video_name or fname == video_name:
                    return os.path.join(dirpath, fname)
    return None

def process_csv(csv_path: str, base_folders, asr_pipe, model_name):
    """
    Read the CSV, replace the 'text' column by running ASR on each video’s audio.
    For each row, use emotion one-hot columns to narrow search to subfolder if possible.
    Saves a new CSV named "<original>_<model_name>.csv" next to original.
    """
    print(f"Processing CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Prepare an output-audio directory next to the CSV
    audio_output_dir = os.path.join(os.path.dirname(csv_path), "extracted_audio")
    os.makedirs(audio_output_dir, exist_ok=True)

    # For each row
    for idx, row in tqdm(df.iterrows(),
                         total=len(df),
                         desc=f"  Rows in {os.path.basename(csv_path)}"):
        vid_name = str(row.get("video_name", "")).strip()
        if not vid_name:
            tqdm.write(f"!!! Empty video_name at row {idx}, skipping.")
            continue

        # Determine emotion from one-hot columns
        folder_name = None
        # Check among the seven primary emotions
        for emo in EMOTIONS_EXT:
            try:
                val = float(row.get(emo, 0))
            except:
                val = 0.0
            if emo != "Other" and val == 1.0:
                # pick this emotion
                folder_name = EMO_FOLDER_MAP.get(emo, None)
                # Break on first found; if multiple ==1, this picks first in EMOTIONS_EXT order
                break
        # If folder_name determined, set candidate dirs under each base folder
        video_path = None
        if folder_name:
            candidate_dirs = [os.path.join(base, folder_name) for base in base_folders]
            # Try finding within the emotion subfolders
            video_path = find_video(vid_name, candidate_dirs)
            if video_path is None:
                # Not found in narrowed subfolder; fallback to broad search
                video_path = find_video(vid_name, base_folders)
        else:
            # No specific emotion or only Other=1: fallback broad search
            video_path = find_video(vid_name, base_folders)

        if video_path is None:
            tqdm.write(f"!!! video not found for '{vid_name}' (emotion folder: {folder_name}) — skipping row {idx}")
            continue

        # Build wav path
        wav_path = os.path.join(audio_output_dir, vid_name + ".wav")
        # Extract audio
        try:
            extract_audio(video_path, wav_path)
        except Exception as e:
            tqdm.write(f"!!! Failed to extract audio for '{video_path}': {e}")
            continue

        # Run ASR
        try:
            result = asr_pipe(wav_path)
            # result may be dict {"text":...} or list of strings/objects
            if isinstance(result, dict):
                text = result.get("text", "")
            elif isinstance(result, list):
                # take first item
                first = result[0]
                if isinstance(first, dict):
                    text = first.get("text", "")
                elif hasattr(first, "text"):
                    text = first.text
                else:
                    text = str(first)
            else:
                # Unexpected format
                text = str(result)
            df.at[idx, "text"] = text
        except Exception as e:
            tqdm.write(f"ASR failed on {wav_path}: {e}")

    # Build new CSV path
    safe_model = model_name.replace("/", "_")
    new_csv_path = csv_path[:-4] + f"_{safe_model}.csv"
    df.to_csv(new_csv_path, index=False)
    print(f"Saved updated CSV to {new_csv_path}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Batch ASR from videos listed in CSVs under emotion subfolders."
    )
    parser.add_argument(
        "--folders", "-f",
        nargs="+", required=True,
        help="One or more root folders containing emotion subfolders (e.g. Train_AFEW, Val_AFEW)."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model ID for ASR or 'parakeet'."
    )
    parser.add_argument(
        "--csvs", "-c",
        nargs="+",
        help="One or more CSV file paths to process. If omitted, globbing '*.csv' in CWD is used."
    )
    parser.add_argument(
        "--device", "-d",
        type=int, default=-1,
        help="CUDA device ID (e.g. 0) or -1 for CPU."
    )
    args = parser.parse_args()

    # Validate base folders
    base_folders = []
    for fld in args.folders:
        if os.path.isdir(fld):
            base_folders.append(os.path.abspath(fld))
        else:
            print(f"!!! Warning: folder '{fld}' does not exist or is not a directory; skipping.")
    if not base_folders:
        print("No valid folders provided; exiting.")
        return

    # Initialize the ASR pipeline once
    print(f"Loading ASR model {args.model} on device {args.device}…")
    if "parakeet" not in args.model:
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model,
            device=args.device,
            chunk_length_s=30,
            stride_length_s=(5, 5),
            torch_dtype=torch_dtype
        )
    else:
        # NeMo Parakeet setup
        print("Loading NVIDIA Parakeet-TDT-0.6B-V2 via NeMo...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        asr_model.change_attention_model("rel_pos_local_attn", [128, 128])
        asr_model.change_subsampling_conv_chunking_factor(1)
        if args.device is not None and args.device >= 0 and torch.cuda.is_available():
            asr_model = asr_model.to(f"cuda:{args.device}")
        else:
            asr_model = asr_model.to("cpu")
        def asr_pipe(wav_path: str):
            out = asr_model.transcribe([wav_path])
            return out

    # Determine CSV paths
    if args.csvs:
        csv_paths = []
        for p in args.csvs:
            if os.path.isfile(p):
                csv_paths.append(p)
            else:
                print(f"!!! Warning: CSV '{p}' not found; skipping.")
        if not csv_paths:
            print("No valid CSV files provided; exiting.")
            return
    else:
        csv_paths = glob.glob("*.csv")
        if not csv_paths:
            print("No CSV files found in current directory; exiting.")
            return

    # Process each CSV
    for csv_path in csv_paths:
        process_csv(csv_path, base_folders, asr_pipe, args.model)


if __name__ == "__main__":
    main()

