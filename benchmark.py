"""
TTS CPU Benchmark: Supertonic 3 vs Kokoro 82M
==============================================
4 configs × 6 text lengths × 5 reps = 120 timed runs (CPU-only)
"""

import os
import sys
import time
import csv
import traceback
import numpy as np
import soundfile as sf

# Force CPU-only for all ONNX sessions
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_ALL_PROVIDERS_EXCEPT_CPU"] = "1"

# ── Text corpus ────────────────────────────────────────────────────────────────
TEXT_CORPUS = {
    "tiny": "Hello there.",
    "short": "The quick brown fox jumps over the lazy dog near the river.",
    "medium": (
        "Artificial intelligence is transforming the way we interact with technology. "
        "From voice assistants to autonomous vehicles, machine learning models are "
        "becoming an integral part of our daily lives."
    ),
    "long": (
        "Text-to-speech synthesis has advanced dramatically over the past decade. "
        "Modern neural TTS systems can produce natural-sounding speech that is nearly "
        "indistinguishable from human recordings. These systems rely on deep learning "
        "architectures such as Transformers and flow-matching models to generate "
        "high-quality audio waveforms directly from text input. The key challenge "
        "remains achieving real-time performance on commodity hardware without "
        "sacrificing audio quality or naturalness."
    ),
    "paragraph": (
        "The development of efficient text-to-speech systems has been a long-standing "
        "goal in the field of speech synthesis. Early systems relied on concatenative "
        "methods that stitched together pre-recorded phoneme segments, producing "
        "robotic-sounding output. The introduction of parametric synthesis improved "
        "flexibility but still fell short of natural human speech. The deep learning "
        "revolution brought neural TTS systems like Tacotron, WaveNet, and FastSpeech, "
        "which dramatically improved naturalness. Today, state-of-the-art systems such "
        "as VITS, StyleTTS2, and flow-matching models can produce speech that rivals "
        "professional voice actors. The remaining frontier is real-time CPU inference, "
        "which is critical for edge deployment, privacy-preserving applications, and "
        "low-latency interactive systems where GPU resources are unavailable or "
        "cost-prohibitive."
    ),
    "extended": (
        "Speech synthesis technology has undergone a remarkable transformation over "
        "the past several decades. In the early days of computing, text-to-speech "
        "systems were primitive rule-based engines that produced highly artificial, "
        "robotic-sounding output. These systems worked by applying phonetic rules to "
        "convert text into a sequence of phonemes, which were then rendered using "
        "simple formant synthesizers. While functional, the output was immediately "
        "recognizable as synthetic and lacked the prosodic richness of natural speech. "
        "\n"
        "The advent of concatenative synthesis marked a significant improvement. By "
        "recording large databases of speech segments and intelligently selecting and "
        "joining them, researchers could produce more natural-sounding output. However, "
        "this approach required enormous storage and was inflexible when it came to "
        "expressing different emotions or speaking styles. The parametric approach "
        "offered more flexibility by modeling the statistical properties of speech, "
        "but the output still sounded somewhat muffled and unnatural. "
        "\n"
        "The deep learning era changed everything. Neural network-based TTS systems "
        "like Google's Tacotron demonstrated that end-to-end learning from raw text "
        "to mel spectrograms was not only feasible but could produce remarkably natural "
        "speech. Subsequent work on WaveNet, WaveGlow, and HiFi-GAN addressed the "
        "vocoder component, enabling high-fidelity waveform generation. More recent "
        "architectures like VITS, NaturalSpeech, and StyleTTS2 have pushed the "
        "boundaries further, achieving human-level naturalness on standard benchmarks. "
        "The challenge now is making these powerful models fast enough for real-time "
        "CPU deployment, which is exactly what this benchmark aims to measure."
    ),
}

TEXT_LENGTH_ORDER = ["tiny", "short", "medium", "long", "paragraph", "extended"]
REPS = 5
WARMUP_TEXT_KEY = "medium"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio_samples")
CSV_PATH = os.path.join(RESULTS_DIR, "raw_results.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def audio_duration(samples, sample_rate):
    return len(samples) / sample_rate


def save_audio(samples, sample_rate, path):
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    # Normalize to [-1, 1] if needed
    max_val = np.abs(samples).max()
    if max_val > 1.0:
        samples = samples / max_val
    sf.write(path, samples, sample_rate)


# ── Config runners ─────────────────────────────────────────────────────────────

class SupertonicRunner:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.name = f"Supertonic-{total_steps}step"
        self._model = None
        self._voice_style = None

    def load(self):
        from supertonic import TTS
        self._model = TTS()
        self._voice_style = self._model.get_voice_style('F1')
        print(f"  [{self.name}] Model loaded. SR={self._model.sample_rate}")

    def synthesize(self, text: str):
        """Returns (samples_np, sample_rate)
        synthesize() returns tuple[np.ndarray, np.ndarray]:
          result[0] shape=(1, N) — audio samples
          result[1] shape=(1,)  — ignored (duration or metadata)
        """
        result = self._model.synthesize(
            text, self._voice_style, total_steps=self.total_steps
        )
        samples = np.array(result[0], dtype=np.float32).flatten()
        sr = self._model.sample_rate  # 44100
        return samples, sr


class KokoroPyTorchRunner:
    def __init__(self):
        self.name = "Kokoro-PyTorch"
        self._pipeline = None

    def load(self):
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code='a')
        print(f"  [{self.name}] Model loaded.")

    def synthesize(self, text: str):
        """Returns (samples_np, sample_rate)"""
        all_samples = []
        sr = 24000
        generator = self._pipeline(text, voice='af_heart', speed=1.0)
        for _, _, audio in generator:
            if audio is not None:
                arr = np.array(audio, dtype=np.float32).flatten()
                all_samples.append(arr)
        if not all_samples:
            raise RuntimeError("Kokoro PyTorch produced no audio")
        return np.concatenate(all_samples), sr


class KokoroONNXRunner:
    def __init__(self):
        self.name = "Kokoro-ONNX"
        self._model = None

    def load(self):
        from kokoro_onnx import Kokoro
        onnx_path = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
        voices_path = os.path.join(MODELS_DIR, "voices-v1.0.bin")
        self._model = Kokoro(onnx_path, voices_path)
        # Two compatibility shims for kokoro-onnx 0.5.0 against this input_ids-style
        # v1.0 ONNX export, applied at the session boundary so we don't edit the
        # installed package:
        #   1. "speed" is fed as int32 but the graph expects tensor(float).
        #   2. The graph returns audio shaped (1, N); create() concatenates per-chunk
        #      outputs and assumes 1-D, so long (multi-chunk) texts crash. Squeeze
        #      the batch axis so concatenation works.
        _orig_run = self._model.sess.run

        def _patched_run(output_names, input_feed, *args, **kwargs):
            if "speed" in input_feed:
                input_feed = {**input_feed,
                              "speed": np.asarray(input_feed["speed"], dtype=np.float32)}
            outputs = _orig_run(output_names, input_feed, *args, **kwargs)
            if (outputs and isinstance(outputs[0], np.ndarray)
                    and outputs[0].ndim == 2 and outputs[0].shape[0] == 1):
                outputs = [outputs[0][0]] + list(outputs[1:])
            return outputs

        self._model.sess.run = _patched_run
        print(f"  [{self.name}] Model loaded.")

    def synthesize(self, text: str):
        """Returns (samples_np, sample_rate)"""
        samples, sr = self._model.create(text, voice='af_heart', speed=1.0, lang='en-us')
        return np.array(samples, dtype=np.float32).flatten(), int(sr)


class InflectNanoRunner:
    """Inflect-Nano-v1 — 4.63M-param FastSpeech + Snake HiFi-GAN, 24kHz, single male voice.

    Ships as a HuggingFace git repo (no pip package). We add the cloned repo to
    sys.path and call its importable synthesize() helper, which returns a numpy
    waveform directly. See models/Inflect-Nano-v1/inference.py.
    """

    def __init__(self):
        self.name = "Inflect-Nano"
        self._repo_dir = os.path.join(MODELS_DIR, "Inflect-Nano-v1")
        self._inf = None
        self._acoustic = None
        self._vocoder = None
        self._speakers = None
        self._device = None

    def load(self):
        import torch
        # The repo's inference.py inserts REPO_ROOT and the vendored frontend onto
        # sys.path at import time, so importing it is enough to wire up its packages.
        if self._repo_dir not in sys.path:
            sys.path.insert(0, self._repo_dir)
        import inference as inflect_inference
        self._inf = inflect_inference
        self._device = torch.device("cpu")
        acoustic_path = os.path.join(self._repo_dir, "weights", "inflect_nano_v1_acoustic.pt")
        vocoder_path = os.path.join(self._repo_dir, "weights", "inflect_nano_v1_vocoder.pt")
        self._acoustic, self._speakers, acoustic_params = inflect_inference.load_acoustic(
            acoustic_path, self._device
        )
        self._vocoder, vocoder_params = inflect_inference.load_vocoder(vocoder_path, self._device)
        print(f"  [{self.name}] Model loaded. "
              f"params: acoustic={acoustic_params:,} vocoder={vocoder_params:,} "
              f"total={acoustic_params + vocoder_params:,}  speakers={self._speakers}")

    def synthesize(self, text: str):
        """Returns (samples_np, sample_rate)"""
        audio = self._inf.synthesize(
            text, self._acoustic, self._vocoder, self._speakers, self._device,
            length_scale=1.0, pitch_scale=1.0, energy_scale=1.0,
        )
        return np.asarray(audio, dtype=np.float32).flatten(), 24000


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run_config(runner, rows: list):
    config_name = runner.name
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    # Load model
    try:
        runner.load()
    except Exception as e:
        print(f"  [ERROR] Failed to load {config_name}: {e}")
        traceback.print_exc()
        return

    # Warmup run (discarded)
    warmup_text = TEXT_CORPUS[WARMUP_TEXT_KEY]
    print(f"  Warmup on '{WARMUP_TEXT_KEY}' ({len(warmup_text)} chars)...")
    try:
        t0 = time.perf_counter()
        runner.synthesize(warmup_text)
        t1 = time.perf_counter()
        print(f"  Warmup done in {t1-t0:.2f}s")
    except Exception as e:
        print(f"  [WARN] Warmup failed: {e}")
        traceback.print_exc()

    # Timed runs
    for label in TEXT_LENGTH_ORDER:
        text = TEXT_CORPUS[label]
        n_chars = len(text)
        print(f"\n  Text: {label} ({n_chars} chars)")
        rep_times = []
        rep_durations = []

        for rep in range(1, REPS + 1):
            try:
                t0 = time.perf_counter()
                samples, sr = runner.synthesize(text)
                t1 = time.perf_counter()

                wall_time = t1 - t0
                audio_dur = audio_duration(samples, sr)
                rtf = wall_time / audio_dur if audio_dur > 0 else float('nan')
                chars_per_sec = n_chars / wall_time if wall_time > 0 else float('nan')

                rep_times.append(wall_time)
                rep_durations.append(audio_dur)

                print(f"    Rep {rep}: wall={wall_time:.3f}s  audio={audio_dur:.2f}s  RTF={rtf:.4f}  chars/s={chars_per_sec:.1f}")

                rows.append({
                    "config_name": config_name,
                    "text_length_label": label,
                    "text_length_chars": n_chars,
                    "rep": rep,
                    "wall_time_sec": round(wall_time, 6),
                    "audio_duration_sec": round(audio_dur, 6),
                    "rtf": round(rtf, 6),
                    "chars_per_sec": round(chars_per_sec, 4),
                })

                # Save audio sample for rep 1 only
                if rep == 1:
                    safe_config = config_name.replace("-", "_").replace(" ", "_")
                    audio_path = os.path.join(AUDIO_DIR, f"{safe_config}_{label}.wav")
                    save_audio(samples, sr, audio_path)
                    print(f"    Saved audio: {audio_path}")

            except Exception as e:
                print(f"    [ERROR] Rep {rep} failed: {e}")
                traceback.print_exc()
                rows.append({
                    "config_name": config_name,
                    "text_length_label": label,
                    "text_length_chars": n_chars,
                    "rep": rep,
                    "wall_time_sec": float('nan'),
                    "audio_duration_sec": float('nan'),
                    "rtf": float('nan'),
                    "chars_per_sec": float('nan'),
                })

        if rep_times:
            mean_rtf = (sum(t / d for t, d in zip(rep_times, rep_durations) if d > 0)) / len(rep_times)
            print(f"  → {label}: mean_wall={sum(rep_times)/len(rep_times):.3f}s  mean_RTF={mean_rtf:.4f}")


def main():
    print("TTS CPU Benchmark: Supertonic 3 vs Kokoro 82M vs Inflect-Nano-v1")
    print(f"Text lengths: {TEXT_LENGTH_ORDER}")
    print(f"Reps per cell: {REPS}")
    print(f"Total planned runs: {len(TEXT_LENGTH_ORDER) * 5 * REPS} (5 configs)")
    print()

    # Check CPU info
    try:
        with open("/proc/cpuinfo") as f:
            cpu_info = f.read()
        model_lines = [l for l in cpu_info.split('\n') if 'model name' in l]
        if model_lines:
            print(f"CPU: {model_lines[0].split(':')[1].strip()}")
    except Exception:
        pass
    print(f"CPU cores: {os.cpu_count()}")
    print()

    configs = [
        SupertonicRunner(total_steps=2),
        SupertonicRunner(total_steps=5),
        KokoroPyTorchRunner(),
        KokoroONNXRunner(),
        InflectNanoRunner(),
    ]

    rows = []
    for runner in configs:
        run_config(runner, rows)

    # Save CSV
    if rows:
        fieldnames = ["config_name", "text_length_label", "text_length_chars",
                      "rep", "wall_time_sec", "audio_duration_sec", "rtf", "chars_per_sec"]
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{'='*60}")
        print(f"Results saved to: {CSV_PATH}")
        print(f"Total rows written: {len(rows)}")

        # Quick validation
        valid_rows = [r for r in rows if not (
            r['rtf'] != r['rtf'] or  # NaN check
            r['rtf'] <= 0
        )]
        print(f"Valid rows (RTF > 0, no NaN): {len(valid_rows)}")
        print(f"Audio samples saved to: {AUDIO_DIR}")
        print(f"Audio sample count: {len(os.listdir(AUDIO_DIR))}")
    else:
        print("ERROR: No rows collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
