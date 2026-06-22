# TTS CPU Benchmark Report: Kokoro 82M vs Supertonic 3 vs Inflect-Nano-v1

*Generated: 2026-06-22 11:14:33 UTC*

---

## Executive Summary

This report presents a rigorous CPU-only benchmark comparing **Kokoro 82M**, **Supertonic 3**, and **Inflect-Nano-v1** across 6 text lengths (12–1712 characters), 5 configurations, and 5 repetitions each (150 total timed runs). All inference was performed on CPU with no GPU acceleration. Audio quality is reported as an objective UTMOS predicted MOS (neural naturalness estimate, ~1–5 scale).

| Config | Overall Mean RTF | vs Real-Time | Mean MOS (UTMOS) |
|--------|-----------------|--------------|------------------|
| Supertonic-3 (2-step) | **0.1120** | 8.9× faster than real-time | **1.57** |
| Supertonic-3 (5-step) | **0.1951** | 5.1× faster than real-time | **4.38** |
| Kokoro-82M (PyTorch) | **0.5351** | 1.9× faster than real-time | **4.45** |
| Kokoro-82M (ONNX) | **0.5643** | 1.8× faster than real-time | **4.44** |
| Inflect-Nano-v1 (4.6M) | **0.1331** | 7.5× faster than real-time | **3.48** |

> **Winner (lowest RTF):** Supertonic-3 (2-step) with mean RTF = 0.1120

---

## Hardware & Environment

| Property | Value |
|----------|-------|
| CPU Model | Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz |
| CPU Cores | 4 |
| RAM | 15.6 GB |
| OS | Linux-6.17.0-1018-azure-x86_64-with-glibc2.39 |
| Python | 3.12.3 |
| supertonic | 1.3.1 |
| kokoro | 0.9.4 |
| kokoro-onnx | unknown |
| onnxruntime | 1.27.0 |
| torch | 2.12.1+cu130 |

---

## Methodology

### Configurations Tested

| Config | Model | Backend | Steps/Mode |
|--------|-------|---------|------------|
| Supertonic-3 (2-step) | Supertone/supertonic-3 | ONNX Runtime (CPU) | total_steps=2 (speed mode) |
| Supertonic-3 (5-step) | Supertone/supertonic-3 | ONNX Runtime (CPU) | total_steps=5 (default quality) |
| Kokoro-82M (PyTorch) | hexgrad/Kokoro-82M | PyTorch CPU | Default |
| Kokoro-82M (ONNX) | onnx-community/Kokoro-82M-v1.0-ONNX | ONNX Runtime (CPU) | Full precision |
| Inflect-Nano-v1 (4.6M) | owensong/Inflect-Nano-v1 | PyTorch CPU | FastSpeech + Snake HiFi-GAN, single male voice |

### Text Corpus

| Label | Characters | Description |
|-------|-----------|-------------|
| tiny | 12 | Single short greeting |
| short | 59 | One sentence (pangram) |
| medium | 196 | 2–3 sentences on AI |
| long | 483 | Paragraph on neural TTS |
| paragraph | 851 | Multi-sentence technical paragraph |
| extended | 1712 | Multi-paragraph essay (~1700 chars) |

### Protocol

- **CPU-only**: `CUDA_VISIBLE_DEVICES=''` set for all runs; ONNX sessions use `CPUExecutionProvider` only
- **Warmup**: 1 discarded warmup run per config on the 'medium' text before timing begins
- **Repetitions**: 5 timed runs per (config × text_length) cell
- **Timing**: `time.perf_counter()` wall-clock, measuring synthesis only (not model load)
- **Metrics**:
  - **RTF** = wall_time / audio_duration (lower = faster; <1.0 = real-time capable)
  - **Latency** = wall-clock seconds per synthesis call
  - **Throughput** = input_chars / wall_time (chars/sec)
- **Voice**: Supertonic voice 'F1' (female); Kokoro voice 'af_heart' (female); Inflect-Nano-v1 default voice 'mark' (male, single-speaker)
- **Audio saved**: 1 WAV sample per (config × text_length) for quality verification

---

## Results

### Mean RTF by Config and Text Length

*(Lower RTF = faster; RTF < 1.0 = faster than real-time)*

| Config | Tiny | Short | Medium | Long | Paragraph | Extended | **Mean** |
|--------|-------|-------|-------|-------|-------|-------|---------|
| Supertonic-3 (2-step) | 0.1854±0.0050 | 0.1142±0.0050 | 0.0974±0.0015 | 0.0954±0.0071 | 0.0912±0.0019 | 0.0884±0.0016 | **0.1120** |
| Supertonic-3 (5-step) | 0.3205±0.0053 | 0.1910±0.0021 | 0.1642±0.0019 | 0.1663±0.0016 | 0.1636±0.0045 | 0.1649±0.0021 | **0.1951** |
| Kokoro-82M (PyTorch) | 0.4622±0.0311 | 0.4191±0.0098 | 0.5000±0.0141 | 0.6280±0.0056 | 0.6254±0.0034 | 0.5757±0.0095 | **0.5351** |
| Kokoro-82M (ONNX) | 0.7338±0.0095 | 0.5499±0.0046 | 0.5093±0.0055 | 0.5388±0.0661 | 0.5547±0.0801 | 0.4990±0.0041 | **0.5643** |
| Inflect-Nano-v1 (4.6M) | 0.1536±0.0594 | 0.1195±0.0204 | 0.1143±0.0033 | 0.1310±0.0137 | 0.1342±0.0015 | 0.1462±0.0032 | **0.1331** |

### Mean Wall-Clock Latency (seconds) by Config and Text Length

| Config | Tiny | Short | Medium | Long | Paragraph | Extended |
|--------|-------|-------|-------|-------|-------|-------|
| Supertonic-3 (2-step) | 0.258s | 0.485s | 1.329s | 3.250s | 5.703s | 10.647s |
| Supertonic-3 (5-step) | 0.447s | 0.811s | 2.242s | 5.669s | 10.231s | 19.854s |
| Kokoro-82M (PyTorch) | 0.705s | 1.697s | 6.262s | 19.061s | 33.113s | 63.661s |
| Kokoro-82M (ONNX) | 0.689s | 1.924s | 6.084s | 16.553s | 29.053s | 51.896s |
| Inflect-Nano-v1 (4.6M) | 0.125s | 0.371s | 1.073s | 1.956s | 2.005s | 2.183s |

### Mean Throughput (chars/sec) by Config and Text Length

| Config | Tiny | Short | Medium | Long | Paragraph | Extended |
|--------|-------|-------|-------|-------|-------|-------|
| Supertonic-3 (2-step) | 46.5 | 121.7 | 147.5 | 149.3 | 149.3 | 160.8 |
| Supertonic-3 (5-step) | 26.9 | 72.7 | 87.4 | 85.2 | 83.2 | 86.2 |
| Kokoro-82M (PyTorch) | 17.1 | 34.8 | 31.3 | 25.3 | 25.7 | 26.9 |
| Kokoro-82M (ONNX) | 17.4 | 30.7 | 32.2 | 29.5 | 29.8 | 33.0 |
| Inflect-Nano-v1 (4.6M) | 104.7 | 162.2 | 182.9 | 248.8 | 424.5 | 784.6 |

### Reference: Mean Audio Duration (seconds) per Config × Text Length

| Config | Tiny | Short | Medium | Long | Paragraph | Extended |
|--------|-------|-------|-------|-------|-------|-------|
| Supertonic-3 (2-step) | 1.39s | 4.25s | 13.65s | 34.09s | 62.55s | 120.43s |
| Supertonic-3 (5-step) | 1.39s | 4.25s | 13.65s | 34.09s | 62.55s | 120.43s |
| Kokoro-82M (PyTorch) | 1.52s | 4.05s | 12.53s | 30.35s | 52.95s | 110.58s |
| Kokoro-82M (ONNX) | 0.94s | 3.50s | 11.95s | 30.72s | 52.37s | 104.00s |
| Inflect-Nano-v1 (4.6M) | 0.81s | 3.10s | 9.39s | 14.93s | 14.93s | 14.93s |

### Audio Quality — UTMOS Predicted MOS by Config and Text Length

*(Higher = more natural; UTMOS predicts mean opinion score on a ~1–5 scale. Scores are objective neural estimates, not human ratings — and on Inflect-Nano-v1 the metric is optimistic: human listening rates it buzzy/robotic, below what its 3.48 suggests.)*

| Config | Tiny | Short | Medium | Long | Paragraph | Extended | **Mean** |
|--------|-------|-------|-------|-------|-------|-------|---------|
| Supertonic-3 (2-step) | 1.32 | 2.35 | 1.58 | 1.45 | 1.36 | 1.35 | **1.57** |
| Supertonic-3 (5-step) | 4.18 | 4.47 | 4.51 | 4.52 | 4.48 | 4.12 | **4.38** |
| Kokoro-82M (PyTorch) | 4.05 | 4.51 | 4.55 | 4.54 | 4.53 | 4.53 | **4.45** |
| Kokoro-82M (ONNX) | 4.04 | 4.51 | 4.54 | 4.55 | 4.52 | 4.49 | **4.44** |
| Inflect-Nano-v1 (4.6M) | 3.02 | 4.15 | 3.90 | 3.45 | 3.01 | 3.37 | **3.48** |

---

## Analysis & Findings

### 1. Overall Speed Ranking

1. **Supertonic-3 (2-step)** — Mean RTF: 0.1120 (8.9× real-time)
2. **Inflect-Nano-v1 (4.6M)** — Mean RTF: 0.1331 (7.5× real-time)
3. **Supertonic-3 (5-step)** — Mean RTF: 0.1951 (5.1× real-time)
4. **Kokoro-82M (PyTorch)** — Mean RTF: 0.5351 (1.9× real-time)
5. **Kokoro-82M (ONNX)** — Mean RTF: 0.5643 (1.8× real-time)

### 2. Speed vs Quality — the core trade-off

Supertonic 3 at 2-step mode is the fastest config (mean RTF **0.1120**, 8.9× real-time), **4.8× faster** than Kokoro 82M (PyTorch) at RTF 0.5351. But speed alone is misleading: its UTMOS quality is only **1.57**, by far the lowest in the field — the 2-step output is audibly robotic. The objective MOS confirms what listening reveals.

At 5-step mode, Supertonic's RTF rises to **0.1951** (a 1.74× slowdown vs 2-step from the extra flow-matching denoising steps), but quality jumps to **4.38** — competitive with Kokoro. This is the configuration that actually balances speed and quality.

Kokoro 82M scores highest on quality (PyTorch **4.45**, ONNX **4.44**) but is the slowest (RTF ~0.54–0.56).

### 3. Inflect-Nano-v1: tiny and fast, but robotic to the ear

At just 4.63M parameters — roughly 18× smaller than Kokoro and 21× smaller than Supertonic — Inflect-Nano-v1 is the second-fastest config (mean RTF **0.1331**, 7.5× real-time). Its UTMOS score is **3.48**, which places it mid-field on the metric — but **human listening does not agree with that score**: the output is audibly buzzy and robotic, with a metallic vocoder texture and flat prosody. It is more intelligible than Supertonic-2step (which is worse), but it is not in the same league as Kokoro or Supertonic-5step. This is a known UTMOS failure mode: it tends to over-rate small HiFi-GAN vocoders that are *clean* but not *natural*. Treat Inflect-Nano's 3.48 as an optimistic upper bound, not a usability verdict.

> **Important caveat — output length cap.** Inflect-Nano-v1's acoustic model has `max_frames = 1400`, which caps synthesis at **~14.93 seconds of audio** regardless of input length. Inputs longer than that (here: `long`, `paragraph`, `extended`) are **silently truncated** — only the first ~15s is rendered. Its RTF and throughput on those rows are therefore inflated (it is doing less work than the other models, which synthesize the full text). Treat Inflect-Nano's `tiny`/`short`/`medium` numbers as the honest comparison; for long-form use you must split text into <15s chunks yourself. Its audio-duration row below (flat 14.93s for the three longest inputs) makes the cap visible.

### 4. Kokoro PyTorch vs ONNX

On this hardware Kokoro ONNX (RTF **0.5643**) and PyTorch (**0.5351**) are within ~5% of each other, and their quality is identical to two decimal places (**4.44** vs **4.45**). The two are perceptually interchangeable; the choice is a deployment/packaging decision, not a quality one.

### 5. Practical Implications

| Use Case | Recommended Config | Reason |
|----------|-------------------|--------|
| Highest quality (human-like) | Kokoro-82M (PyTorch or ONNX) | Top UTMOS (~4.45), Apache-2.0 weights |
| Balanced speed + quality | Supertonic-3 (5-step) | MOS ~4.4 at ~5× real-time |
| Tiny footprint / edge, quality secondary | Inflect-Nano-v1 | 4.6M params, ~7.5× real-time, but buzzy/robotic (MOS 3.5 over-rates it) |
| Latency at any cost (prototyping) | Supertonic-3 (2-step) | Fastest, but MOS ~1.6 (robotic) |
| PyTorch ecosystem / fine-tuning | Kokoro-82M (PyTorch) | Native PyTorch, easy to extend |

### 6. Reproducibility Notes

- All runs performed on a single CPU process with default thread counts
- No process pinning or CPU affinity was set
- Results may vary ±5–10% across runs due to OS scheduling jitter
- The benchmark harness (`benchmark.py`) is fully reproducible: same text, same warmup protocol, same timing method

---

## Charts

### RTF Comparison
![RTF Comparison](charts/rtf_comparison.png)

### Latency vs Text Length
![Latency vs Text Length](charts/latency_vs_length.png)

### Quality vs Speed
![Quality vs Speed](charts/quality_vs_speed.png)

---

## Raw Data

Full raw results (150 rows): [`raw_results.csv`](raw_results.csv)

Per-sample MOS: [`mos_results.csv`](mos_results.csv)

Audio samples: [`audio_samples/`](audio_samples/) — 30 WAV files (1 per config × text_length)

---

*Report generated by `report.py` on 2026-06-22 11:14:33 UTC*