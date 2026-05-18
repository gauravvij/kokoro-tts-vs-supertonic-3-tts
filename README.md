# Kokoro 82M vs Supertonic 3: CPU TTS Benchmark

<a href="https://heyneo.com/" target="_blank"><img src="https://img.shields.io/badge/Built%20with-Neo%20AI%20Engineer-black?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSIzIiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="Built with Neo AI Engineer" /></a>

This TTS model benchmark was designed, implemented, and executed end-to-end by [Neo](https://heyneo.com/) — an autonomous AI engineering agent — from a single prompt. No manual coding or configuration was required.

---

A complete CPU-only benchmark comparing Kokoro 82M and Supertonic 3 across speed, latency, throughput, and audio quality.

## Results Summary

| Config | Mean RTF | Quality |
|--------|----------|---------|
| Supertonic-3 (2-step) | 0.165 | Poor (robotic) |
| Supertonic-3 (5-step) | 0.313 | Good (clear) |
| Kokoro-82M (PyTorch) | 0.469 | Excellent (human-like) |
| Kokoro-82M (ONNX) | 0.509 | Excellent (human-like) |

**Speed winner:** Supertonic-3 (2-step) at 6.1x real-time  
**Quality + speed balance:** Supertonic-3 (5-step) at 3.2x real-time  
**Quality winner:** Kokoro-82M at 2.0-2.1x real-time

Read the full analysis: [results/benchmark_report.md](results/benchmark_report.md)  
Read the blog post: [blog_post.md](blog_post.md)

## Files

```
benchmark.py                    # Full benchmark harness (120 timed runs)
report.py                       # Report and chart generator
results/
  raw_results.csv               # 120 rows of raw timing data
  benchmark_report.md           # Full analysis report with quality assessment
  charts/
    rtf_comparison.png
    latency_vs_length.png
  audio_samples/                # 24 WAV files (1 per config x text length)
blog_post.md                    # Full blog post writeup
```

## Hardware

AMD EPYC 7763, 4 cores, 15.6GB RAM, no GPU. Python 3.11.

## Reproduce

```bash
# Install system dependency
apt install espeak-ng

# Create venv and install packages
python -m venv venv && source venv/bin/activate
pip install supertonic kokoro kokoro-onnx onnxruntime soundfile matplotlib pandas numpy torch

# Download Kokoro ONNX model files
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx
curl -L -o models/voices-v1.0.bin https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices.bin

# Run benchmark
python benchmark.py

# Generate report and charts
python report.py
```
