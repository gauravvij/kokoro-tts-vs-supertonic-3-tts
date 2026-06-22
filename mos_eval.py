"""
Automated MOS (Mean Opinion Score) evaluation for the TTS benchmark
====================================================================
Scores every saved audio sample with UTMOS (utmos22_strong), an objective
neural naturalness predictor, so audio quality gets a number instead of a
purely subjective listen.

Reads:  results/audio_samples/*.wav
Writes: results/mos_results.csv   (config_name, text_length_label, mos)

UTMOS is loaded from torch.hub (tarepan/SpeechMOS). The predictor handles
resampling internally, so we pass the waveform at its native sample rate.
"""

import csv
import os

import numpy as np
import soundfile as sf
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio_samples")
MOS_CSV = os.path.join(RESULTS_DIR, "mos_results.csv")

LENGTH_ORDER = ["tiny", "short", "medium", "long", "paragraph", "extended"]


def parse_filename(stem: str):
    """'Kokoro_PyTorch_medium' -> ('Kokoro-PyTorch', 'medium').

    Audio files are saved as <config_with_underscores>_<length>.wav by
    benchmark.py. Length labels are single tokens, so the last underscore-
    separated token is the length and the rest is the (hyphenated) config.
    """
    parts = stem.split("_")
    label = parts[-1]
    if label not in LENGTH_ORDER:
        return None, None
    config_name = "-".join(parts[:-1])
    return config_name, label


def load_mono(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def main():
    print("UTMOS MOS evaluation")
    print(f"Reading audio from: {AUDIO_DIR}")

    print("Loading UTMOS (utmos22_strong) from torch.hub...")
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    predictor.eval()

    wavs = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"Found {len(wavs)} WAV files")

    rows = []
    for fname in wavs:
        stem = os.path.splitext(fname)[0]
        config_name, label = parse_filename(stem)
        if config_name is None:
            print(f"  [SKIP] unrecognized filename: {fname}")
            continue
        audio, sr = load_mono(os.path.join(AUDIO_DIR, fname))
        with torch.no_grad():
            wave = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
            score = float(predictor(wave, sr).squeeze().item())
        rows.append({
            "config_name": config_name,
            "text_length_label": label,
            "mos": round(score, 4),
        })
        print(f"  {config_name:<18} {label:<10} MOS={score:.3f}")

    if not rows:
        raise RuntimeError("No audio samples scored — did benchmark.py run?")

    with open(MOS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config_name", "text_length_label", "mos"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {MOS_CSV}")

    # Mean MOS per config
    by_config = {}
    for r in rows:
        by_config.setdefault(r["config_name"], []).append(r["mos"])
    print("\nMean MOS per config (higher = better, scale ~1-5):")
    for cfg, vals in sorted(by_config.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {cfg:<18} {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
