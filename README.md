# Kokoro 82M vs Supertonic 3 vs Inflect-Nano-v1: CPU TTS Benchmark

<a href="https://heyneo.com/" target="_blank"><img src="https://img.shields.io/badge/Built%20with-Neo%20AI%20Engineer-black?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSIzIiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="Built with Neo AI Engineer" /></a>

This TTS model benchmark was designed, implemented, and executed end-to-end by [Neo](https://heyneo.com/) — an autonomous AI engineering agent. No manual coding or configuration was required.

---

A complete CPU-only benchmark comparing **Kokoro 82M**, **Supertonic 3**, and **Inflect-Nano-v1** across speed, latency, throughput, and audio quality — with an objective UTMOS score for every sample. All five configurations were measured on the same machine for an apples-to-apples comparison.

## Results Summary

> **What are MOS and UTMOS?** *MOS (Mean Opinion Score)* is the standard way speech quality is measured: human listeners rate how natural a clip sounds on a 1–5 scale (1 = bad/robotic, 5 = indistinguishable from a real person) and the scores are averaged. Running a real listening panel is slow and expensive, so this benchmark uses *UTMOS* — a neural network trained on thousands of human MOS ratings that **predicts** that same 1–5 score automatically from the audio. Every sample below is scored by UTMOS, giving one consistent quality number across all configs without a human panel. Treat it as a strong proxy, not ground truth: as the Inflect-Nano-v1 row shows, UTMOS can over-rate clean-but-unnatural vocoders, so we also cross-check by ear. (`RTF` is *Real-Time Factor* — wall-clock seconds to synthesize one second of audio, so lower is faster and `1/RTF` is the real-time multiple.)

| Config | Mean RTF | vs Real-Time | Mean MOS (UTMOS) | Params |
|--------|----------|--------------|------------------|--------|
| Inflect-Nano-v1 | 0.1376 | 7.3× | 3.48† (buzzy/robotic) | **4.6M** |
| Supertonic-3 (2-step) | 0.1781 | 5.6× | 1.53 (robotic) | ~100M |
| Supertonic-3 (5-step) | 0.3164 | 3.2× | 4.37 (good) | ~100M |
| Kokoro-82M (ONNX) | 0.5711 | 1.8× | 4.44 (human-like) | 82M |
| Kokoro-82M (PyTorch) | 0.7865 | 1.3× | 4.45 (human-like) | 82M |

**Quality winner:** Kokoro-82M (MOS ~4.45, Apache-2.0)
**Best speed + quality balance:** Supertonic-3 (5-step) (MOS 4.37 at 3.2× real-time)
**Tiny + fast, but robotic:** Inflect-Nano-v1 — 4.6M params, 7.3× real-time, but buzzy/robotic by ear and caps output at ~15s (UTMOS 3.48 over-rates it)
**Fastest but unusable:** Supertonic-3 (2-step) — MOS 1.53

†*UTMOS over-rates Inflect-Nano-v1: its 3.48 looks mid-pack, but human listening finds it buzzy and robotic. The metric flatters small HiFi-GAN vocoders that are clean but not natural.*

![Quality vs Speed](results/charts/quality_vs_speed.png)
![RTF Comparison](results/charts/rtf_comparison.png)

Read the full analysis: [results/benchmark_report.md](results/benchmark_report.md)
Read the blog post: [blog_post.md](blog_post.md)

> **Caveat on Inflect-Nano-v1:** its acoustic model has `max_frames = 1400`, capping synthesis at ~14.93s of audio. Inputs longer than that (`long`/`paragraph`/`extended`) are silently truncated, so its RTF/throughput on those rows are inflated. Treat the `tiny`/`short`/`medium` rows as the fair comparison; chunk long text yourself for real long-form use.

## Files

```
benchmark.py                    # Full benchmark harness (150 timed runs, 5 configs)
mos_eval.py                     # UTMOS quality scoring → results/mos_results.csv
report.py                       # Report and chart generator
results/
  raw_results.csv               # 150 rows of raw timing data
  mos_results.csv               # 30 rows of per-sample MOS
  benchmark_report.md           # Full analysis report with quality assessment
  charts/
    rtf_comparison.png
    latency_vs_length.png
    quality_vs_speed.png        # MOS vs RTF scatter
  audio_samples/                # 30 WAV files (1 per config x text length)
results_amd_epyc/               # Original AMD EPYC 7763 results (reference)
blog_post.md                    # Full blog post writeup
```

## Hardware

Intel Xeon Platinum 8272CL, 4 cores, 15.6GB RAM, no GPU. Python 3.12.

## Reproduce

```bash
# Install system dependency
apt install espeak-ng

# Create venv and install packages
python -m venv venv && source venv/bin/activate
pip install supertonic kokoro kokoro-onnx onnxruntime soundfile matplotlib pandas numpy torch torchaudio numba g2p_en transformers

# Download Kokoro ONNX model + voices
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx
curl -L -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Clone Inflect-Nano-v1 (needs git-lfs, or fetch the .pt/.pickle LFS files separately)
git clone https://huggingface.co/owensong/Inflect-Nano-v1 models/Inflect-Nano-v1

# Run benchmark, score quality, generate report
python benchmark.py
python mos_eval.py
python report.py
```

## Notes on reproducibility

- All five configs were measured in a single session on the same CPU. `CUDA_VISIBLE_DEVICES=''` enforced CPU-only.
- `kokoro-onnx 0.5.0` has a dtype/shape bug against the v1.0 ONNX export; `benchmark.py` patches it at the session boundary (`KokoroONNXRunner.load`).
- Inflect-Nano-v1's LFS weights (`weights/*.pt`) and `third_party/.../cmudict_cache.pickle` must be fetched via git-lfs (or directly from the HF `resolve/main` URLs) if your clone leaves them as pointer files.
