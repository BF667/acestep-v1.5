---
title: ACE-Step v1.5
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
python_version: 3.11
pinned: false
models: 
 - ACE-Step/Ace-Step1.5
 - ACE-Step/acestep-v15-xl-turbo
license: mit
app_file: app.py
short_description: Music Generation Foundation Model v1.5
---

<div align="center">

# 🎵 ACE-Step 1.5

### Pushing the Boundaries of Open-Source Music Generation

[![GitHub Stars](https://img.shields.io/github/stars/ACE-Step/ACE-Step-1.5?style=social)](https://github.com/ACE-Step/ACE-Step-1.5)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Gradio-6.2-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)

[![Project Page](https://img.shields.io/badge/🌐-Project_Page-0A0A0A?style=flat-square)](https://ace-step.github.io/ace-step-v1.5.github.io/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging_Face-FFD21E?style=flat-square)](https://huggingface.co/collections/ACE-Step/ace-step-15)
[![ModelScope](https://img.shields.io/badge/🧩-ModelScope-6236FF?style=flat-square)](https://modelscope.cn/models/ACE-Step/ACE-Step-v1-5)
[![Space Demo](https://img.shields.io/badge/🚀-Live_Demo-FF6F00?style=flat-square)](https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5)
[![Discord](https://img.shields.io/badge/💬-Discord-5865F2?style=flat-square)](https://discord.gg/PeWDxrkdj7)
[![Paper](https://img.shields.io/badge/📄-Technical_Report-B31B1B?style=flat-square)](https://arxiv.org/abs/2602.00744)

<img src="./assets/orgnization_logos.png" width="100%" alt="ACE Studio & StepFun">

</div>

---

## 📑 Table of Contents

- [🆕 What's New in v1.5](#-whats-new-in-v15)
- [📝 Abstract](#-abstract)
- [✨ Features](#-features)
- [🔄 Remix Song — New!](#-remix-song--new)
- [⚡ Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🚀 Usage](#-usage)
- [💻 Usage Examples](#-usage-examples)
- [🔨 Train](#-train)
- [🏗️ Architecture](#️-architecture)
- [🦁 Model Zoo](#-model-zoo)
- [📊 Performance Benchmarks](#-performance-benchmarks)
- [📜 License & Disclaimer](#-license--disclaimer)
- [🙏 Acknowledgements](#-acknowledgements)
- [📖 Citation](#-citation)

---

## 🆕 What's New in v1.5

> **ACE-Step v1.5** is a major leap forward for open-source music generation. Here's what's changed:

| Feature | Description |
|:--------|:------------|
| 🔄 **Remix Song** | Transform any song into a new style with adjustable remix strength — from subtle reimagining to radical reinvention |
| ⚡ **Turbo Model** | 8-step generation (vs. 50 steps) for 100× faster inference with comparable quality |
| 🧠 **Intrinsic RL** | Alignment through the model's own mechanisms — no external reward models or human preference biases |
| 🎯 **Precise Prompt Adherence** | Strict adherence to prompts across 50+ languages |
| 🎛️ **Unified Editing Suite** | Cover, repaint, vocal-to-BGM, track separation & multi-track generation in one model |
| 💾 **Consumer-Grade VRAM** | Runs on <4GB VRAM with CPU offloading — no datacenter required |
| 🎵 **10-Minute Compositions** | Scale from 10-second loops to full 10-minute tracks |
| 🔧 **LoRA Fine-Tuning** | One-click LoRA training in Gradio — 8 songs, 1 hour on an RTX 3090 |

<details>
<summary>📋 Changelog</summary>

**v1.5.2** (Current)
- Added `shift` parameter for timestep shifting
- Added `infer_method` parameter for ODE/SDE selection
- Added `timesteps` parameter for custom timestep schedules
- Added `understand_music()` function for audio analysis
- Added `create_sample()` function for simple mode generation
- Added `format_sample()` function for input enhancement
- Added `UnderstandResult`, `CreateSampleResult`, `FormatSampleResult` dataclasses
- **New Remix Song feature** with adjustable strength

**v1.5.1**
- Split `GenerationConfig` into `GenerationParams` and `GenerationConfig`
- Renamed parameters for consistency (`key_scale` → `keyscale`, `time_signature` → `timesignature`, etc.)
- Added `instrumental` and `use_constrained_decoding` parameters
- Changed default `audio_format` to "flac" and `batch_size` to 2

**v1.5** (Initial)
- Introduced `GenerationConfig` and `GenerationResult` dataclasses
- Simplified parameter passing
- Full inference pipeline with LM + DiT

</details>

---

## 📝 Abstract

We present ACE-Step v1.5, a highly efficient foundation model that democratizes commercial-grade music production on consumer hardware. Optimized for local deployment (<4GB VRAM), the model accelerates generation by over 100× compared to traditional pure LM architectures, producing superior high-fidelity audio in seconds characterized by coherent semantics and exceptional melodies.

At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints — scaling from short loops to 10-minute compositions — while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences.

Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities — such as **remix**, cover generation, repainting, and vocal-to-BGM conversion — while maintaining strict adherence to prompts across 50+ languages.

---

## ✨ Features

<p align="center">
    <img src="./assets/application_map.png" width="100%" alt="ACE-Step Application Map">
</p>

### ⚡ Performance

| | Feature | Details |
|:-:|:--------|:--------|
| 🚀 | **Ultra-Fast Generation** | 0.5s – 10s on A100 (depending on think mode & diffusion steps) |
| ⏱️ | **Flexible Duration** | 10 seconds to 10 minutes (600s) of audio generation |
| 📦 | **Batch Generation** | Generate up to 8 songs simultaneously |

### 🎵 Generation Quality

| | Feature | Details |
|:-:|:--------|:--------|
| 🏆 | **Commercial-Grade Output** | Quality between Suno v4.5 and Suno v5 |
| 🎸 | **Rich Style Support** | 1000+ instruments and styles with fine-grained timbre description |
| 🌍 | **Multi-Language Lyrics** | Supports 50+ languages with lyrics prompt for structure & style control |

### 🎛️ Versatility & Control

| Feature | Description |
|:---------|:------------|
| 🎧 **Reference Audio Input** | Use reference audio to guide generation style |
| 🎤 **Cover Generation** | Create covers from existing audio |
| 🔄 **Remix Song** | Remix existing songs with new styles and adjustable strength |
| 🖌️ **Repaint & Edit** | Selective local audio editing and regeneration |
| 🔀 **Track Separation** | Separate audio into individual stems |
| 🎚️ **Multi-Track Generation** | Add layers like Suno Studio's "Add Layer" feature |
| 🗣️ **Vocal2BGM** | Auto-generate accompaniment for vocal tracks |
| 🎼 **Metadata Control** | Control duration, BPM, key/scale, time signature |
| 💡 **Simple Mode** | Generate full songs from simple descriptions |
| ✍️ **Query Rewriting** | Auto LM expansion of tags and lyrics |
| 🔍 **Audio Understanding** | Extract BPM, key/scale, time signature & caption from audio |
| 📝 **LRC Generation** | Auto-generate lyric timestamps for generated music |
| 🔧 **LoRA Training** | One-click annotation & training in Gradio. 8 songs, 1 hour on 3090 (12GB VRAM) |
| ⭐ **Quality Scoring** | Automatic quality assessment for generated audio |

---

## 🔄 Remix Song — New!

<div align="center">

### 🎛️ Transform Any Song Into a New Style

</div>

The **Remix Song** feature is one of the flagship capabilities of ACE-Step v1.5. It lets you take any existing song and reimagine it in an entirely different style — from a pop ballad to a lo-fi chill track, from rock to smooth jazz — with fine-grained control over how much the original is preserved vs. transformed.

#### How It Works

| Parameter | Description | Range |
|:----------|:------------|:------|
| `task_type` | Set to `"remix"` | — |
| `src_audio` | Path to the source song | Any audio file |
| `remix_strength` | How much to deviate from the original | `0.0` (preserve) → `1.0` (reinvent) |
| `remix_style_prompt` | Target style description | Free text |
| `caption` | Detailed caption for the remixed output | Free text |

#### Remix Strength Guide

| Strength | Effect | Best For |
|:--------:|:-------|:---------|
| `0.1 – 0.3` | Subtle texture & mood shift | Gentle reimagining, same genre |
| `0.3 – 0.5` | Noticeable style transformation | Cross-genre remixes |
| `0.5 – 0.7` | Major reinvention | Radical style changes |
| `0.7 – 1.0` | Near-complete rework | Creative reinterpretations |

#### Remix Examples

```python
# Lo-fi chill remix (gentle transformation)
params = GenerationParams(
    task_type="remix",
    src_audio="original_song.mp3",
    remix_strength=0.5,
    caption="lo-fi hip hop, vinyl crackle, mellow piano, chill beats, warm atmosphere",
    instrumental=True,
)

# EDM festival remix (radical reinvention)
params = GenerationParams(
    task_type="remix",
    src_audio="pop_ballad.mp3",
    remix_strength=0.3,
    caption="EDM, dance, heavy bass drops, synth leads, energetic, festival anthem",
    instrumental=True,
)

# Smooth jazz remix (with vocals)
params = GenerationParams(
    task_type="remix",
    src_audio="rock_song.mp3",
    remix_strength=0.4,
    caption="smooth jazz, saxophone, brushed drums, warm bass, sophisticated",
    instrumental=False,
    vocal_language="en",
)
```

> 💡 **Pro Tip:** Start with `remix_strength=0.5` and adjust from there. Lower values keep more of the original's identity; higher values let the new style dominate.

---

## ⚡ Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5

# 3. Install dependencies
uv sync

# 4. Launch the Gradio Web UI
uv run acestep
```

Then open **http://localhost:7860** in your browser. Models download automatically on first run.

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./ACE_Step_v1_5_Colab.ipynb)

**Try ACE-Step v1.5 instantly in Google Colab — no GPU required!**

</div>

---

## 📦 Installation

> **Requirements:** Python 3.11 · CUDA GPU recommended (works on CPU/MPS but slower)

### Step 1: Install uv (Package Manager)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone & Install

```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
```

### Step 3: Launch

#### 🖥️ Gradio Web UI (Recommended)

```bash
uv run acestep
```

Open http://localhost:7860 in your browser. Models will be downloaded automatically on first run.

#### 🌐 REST API Server

```bash
uv run acestep-api
```

API runs at http://localhost:8001. See [API Documentation](./docs/en/API.md) for endpoints.

#### 📒 Google Colab (Cloud, Free GPU)

<div align="left">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./ACE_Step_v1_5_Colab.ipynb)

</div>

No local GPU? Run ACE-Step v1.5 directly in Google Colab with a free cloud GPU. The notebook handles all setup automatically.

### Command Line Options

**Gradio UI (`acestep`):**

| Option | Default | Description |
|:-------|:--------|:------------|
| `--port` | `7860` | Server port |
| `--server-name` | `127.0.0.1` | Server address (use `0.0.0.0` for network access) |
| `--share` | `false` | Create public Gradio link |
| `--language` | `en` | UI language: `en`, `zh`, `ja` |
| `--init_service` | `false` | Auto-initialize models on startup |
| `--config_path` | `auto` | DiT model (e.g., `acestep-v15-turbo`, `acestep-v15-turbo-shift3`) |
| `--lm_model_path` | `auto` | LM model (e.g., `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-1.7B`) |
| `--offload_to_cpu` | `auto` | CPU offload (auto-enabled if VRAM < 16GB) |

**Examples:**

```bash
# Public access with Chinese UI
uv run acestep --server-name 0.0.0.0 --share --language zh

# Pre-initialize models on startup
uv run acestep --init_service true --config_path acestep-v15-turbo

# Use larger LM model for better composition
uv run acestep --lm_model_path acestep-5Hz-lm-1.7B
```

### 🔧 Troubleshooting

| Issue | Solution |
|:------|:---------|
| **Out of Memory (CUDA OOM)** | Reduce `batch_size`, enable `--offload_to_cpu true`, or use the turbo model |
| **Model download fails** | Check internet connection; manually download from [Hugging Face](https://huggingface.co/collections/ACE-Step/ace-step-15) |
| **Slow generation on CPU** | Use `--config_path acestep-v15-turbo` for fewer steps; CUDA is strongly recommended |
| **`uv` not found** | Restart your shell or run `source $HOME/.local/bin/env` |
| **Port 7860 already in use** | Use `--port 7861` or another available port |
| **Poor quality results** | Increase `inference_steps`, use `guidance_scale=7.0-9.0`, or switch to base model |
| **Results don't match prompt** | Make caption more specific, increase `guidance_scale`, enable LM (`thinking=True`) |

<details>
<summary>🛠️ Development Setup</summary>

```bash
# Add dependencies
uv add package-name
uv add --dev package-name

# Update all dependencies
uv sync --upgrade
```

</details>

---

## 🚀 Usage

We provide multiple ways to use ACE-Step:

| Method | Description | Documentation |
|:-------|:------------|:--------------|
| 🖥️ **Gradio Web UI** | Interactive web interface for music generation | [Gradio Guide](./docs/en/GRADIO_GUIDE.md) |
| 🐍 **Python API** | Programmatic access for integration | [Inference API](./docs/en/INFERENCE.md) |
| 🌐 **REST API** | HTTP-based async API for services | [REST API](./docs/en/API.md) |
| 📒 **Google Colab** | Run in the cloud with free GPU | [Open in Colab](./ACE_Step_v1_5_Colab.ipynb) |

**📚 Documentation available in:** [English](./docs/en/) | [中文](./docs/zh/) | [日本語](./docs/ja/)

---

## 💻 Usage Examples

### 🎵 Text-to-Music Generation

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Initialize
dit_handler = AceStepHandler()
llm_handler = LLMHandler()
dit_handler.initialize_service(config_path="acestep-v15-turbo", device="cuda")
llm_handler.initialize(lm_model_path="acestep-5Hz-lm-0.6B", device="cuda")

# Generate from text
params = GenerationParams(
    caption="upbeat electronic dance music with heavy bass",
    bpm=128,
    duration=30,
)
config = GenerationConfig(batch_size=2, audio_format="flac")
result = generate_music(dit_handler, llm_handler, params, config, save_dir="./output")
```

### 🎤 Song with Lyrics

```python
params = GenerationParams(
    caption="pop ballad with emotional vocals",
    lyrics="""
    [Verse 1]
    Walking down the street today
    Thinking of the words you used to say

    [Chorus]
    I'm moving on, I'm staying strong
    This is where I belong
    """,
    vocal_language="en",
    bpm=72,
    duration=45,
)
result = generate_music(dit_handler, llm_handler, params, config, save_dir="./output")
```

### 🔄 Remix a Song

```python
params = GenerationParams(
    task_type="remix",
    src_audio="original_song.mp3",
    remix_strength=0.5,
    caption="lo-fi hip hop, vinyl crackle, mellow piano, warm atmosphere",
    instrumental=True,
)
result = generate_music(dit_handler, llm_handler, params, config, save_dir="./output")
```

### 🎤 Cover Generation

```python
params = GenerationParams(
    task_type="cover",
    src_audio="original_pop_song.mp3",
    caption="orchestral symphonic arrangement",
    audio_cover_strength=0.7,
    thinking=True,
)
result = generate_music(dit_handler, llm_handler, params, config, save_dir="./output")
```

### 🖌️ Repaint (Edit a Section)

```python
params = GenerationParams(
    task_type="repaint",
    src_audio="generated_song.mp3",
    repainting_start=10.0,  # seconds
    repainting_end=20.0,    # seconds
    caption="smooth piano solo transition",
)
result = generate_music(dit_handler, llm_handler, params, config, save_dir="./output")
```

### 💡 Simple Mode (From a Description)

```python
from acestep.inference import create_sample

sample = create_sample(
    llm_handler=llm_handler,
    query="a soft Bengali love song for a quiet evening",
    vocal_language="bn",
)

if sample.success:
    params = GenerationParams(
        caption=sample.caption,
        lyrics=sample.lyrics,
        bpm=sample.bpm,
        duration=sample.duration,
        keyscale=sample.keyscale,
    )
    result = generate_music(dit_handler, llm_handler, params, config, save_dir="./output")
```

> 📖 **More examples** — including batch generation, custom timesteps, and audio understanding — can be found in the [Inference Documentation](./docs/en/INFERENCE.md).

---

## 🔨 Train

See the **LoRA Training** tab in Gradio UI for one-click training, or check [Gradio Guide - LoRA Training](./docs/en/GRADIO_GUIDE.md#lora-training) for details.

**Quick summary:**
- 8 songs minimum for fine-tuning
- ~1 hour on RTX 3090 (12GB VRAM)
- One-click annotation & training directly in the Gradio UI

---

## 🏗️ Architecture

<p align="center">
    <img src="./assets/ACE-Step_framework.png" width="100%" alt="ACE-Step Framework">
</p>

ACE-Step v1.5 uses a novel hybrid architecture:
1. **Language Model (LM)** — An omni-capable planner that transforms simple queries into comprehensive song blueprints, synthesizing metadata, lyrics, and captions via Chain-of-Thought
2. **Diffusion Transformer (DiT)** — Generates high-fidelity audio conditioned on the LM's structured output
3. **Intrinsic RL** — Alignment through the model's own mechanisms, eliminating biases from external reward models

---

## 🦁 Model Zoo

<p align="center">
    <img src="./assets/model_zoo.png" width="100%" alt="Model Zoo">
</p>

### DiT Models

| DiT Model | Pre-Train | SFT | RL | CFG | Steps | Refer | Text2Music | Cover | Repaint | Extract | Lego | Complete | Quality | Diversity | Fine-Tune | Hugging Face |
|:----------|:---------:|:---:|:--:|:---:|:-----:|:-----:|:----------:|:-----:|:-------:|:-------:|:----:|:--------:|:-------:|:---------:|:---------:|:-------------|
| `acestep-v15-base` | ✅ | ❌ | ❌ | ✅ | 50 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Medium | High | Easy | [Link](https://huggingface.co/ACE-Step/acestep-v15-base) |
| `acestep-v15-sft` | ✅ | ✅ | ❌ | ✅ | 50 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | High | Medium | Easy | [Link](https://huggingface.co/ACE-Step/acestep-v15-sft) |
| `acestep-v15-turbo` | ✅ | ✅ | ❌ | ❌ | 8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Very High | Medium | Medium | [Link](https://huggingface.co/ACE-Step/Ace-Step1.5) |
| `acestep-v15-turbo-rl` | ✅ | ✅ | ✅ | ❌ | 8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Very High | Medium | Medium | *To be released* |

### LM Models

| LM Model | Pretrain From | Pre-Train | SFT | RL | CoT Metas | Query Rewrite | Audio Understand | Composition | Copy Melody | Hugging Face |
|:---------|:--------------|:---------:|:---:|:--:|:---------:|:-------------:|:----------------:|:-----------:|:-----------:|:-------------|
| `acestep-5Hz-lm-0.6B` | Qwen3-0.6B | ✅ | ✅ | ✅ | ✅ | ✅ | Medium | Medium | Weak | [Link](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B) |
| `acestep-5Hz-lm-1.7B` | Qwen3-1.7B | ✅ | ✅ | ✅ | ✅ | ✅ | Medium | Medium | Medium | [Link](https://huggingface.co/ACE-Step/acestep-5Hz-lm-1.7B) |
| `acestep-5Hz-lm-4B` | Qwen3-4B | ✅ | ✅ | ✅ | ✅ | ✅ | Strong | Strong | Strong | *To be released* |

---

## 📊 Performance Benchmarks

> 🚧 **This section is a placeholder.** Official benchmarks will be added soon.

| Metric | acestep-v15-base | acestep-v15-turbo | Reference (Suno v4.5) |
|:-------|:----------------:|:------------------:|:---------------------:|
| Generation Time (30s audio, A100) | *TBD* | *TBD* | — |
| Quality Score | *TBD* | *TBD* | — |
| Prompt Adherence | *TBD* | *TBD* | — |
| VRAM Usage (batch=1) | *TBD* | *TBD* | — |
| VRAM Usage (batch=8) | *TBD* | *TBD* | — |

> See the [Technical Report](https://arxiv.org/abs/2602.00744) for detailed methodology and results.

---

## 📜 License & Disclaimer

This project is licensed under [MIT](./LICENSE).

ACE-Step enables original music generation across diverse genres, with applications in creative production, education, and entertainment. While designed to support positive and artistic use cases, we acknowledge potential risks such as unintentional copyright infringement due to stylistic similarity, inappropriate blending of cultural elements, and misuse for generating harmful content. To ensure responsible use, we encourage users to verify the originality of generated works, clearly disclose AI involvement, and obtain appropriate permissions when adapting protected styles or materials. By using ACE-Step, you agree to uphold these principles and respect artistic integrity, cultural diversity, and legal compliance. The authors are not responsible for any misuse of the model, including but not limited to copyright violations, cultural insensitivity, or the generation of harmful content.

> 🔔 **Important Notice**
>
> The only official website for the ACE-Step project is our GitHub Pages site. We do not operate any other websites.
>
> 🚫 Fake domains include but are not limited to: `ac**p.com`, `a**p.org`, `a***c.org`
>
> ⚠️ Please be cautious. Do not visit, trust, or make payments on any of those sites.

---

## 🙏 Acknowledgements

This project is co-led by **ACE Studio** and **StepFun**.

---

## 📖 Citation

If you find this project useful for your research, please consider citing:

```bibtex
@misc{gong2026acestep,
        title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
        author={Junmin Gong, Song Yulin, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
        howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
        year={2026},
        note={GitHub repository}
}
```
