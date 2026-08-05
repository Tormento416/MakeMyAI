# 🤖 MakeMyAI — SLM / LLM Architect & Framework Studio

A standalone GUI application for designing, instantiating, configuring, and packaging generic, untrained Small Language Models (SLMs) and Large Language Models (LLMs). The app exports models in standard Hugging Face format (`.safetensors`, `config.json`, `tokenizer.json`) optimized for direct fine-tuning in **[QLoRA Trainer Command Center](https://github.com/Tormento416/QLoRA_Trainer_Command_Center.git)**.


---

## 🌟 Key Features

- **✨ Easy Mode (Guided Recommendations)**: Select presets from Micro 50M to Base 3B+ based on target hardware (CPU, 4GB VRAM, 8GB VRAM, 16GB+ VRAM).
- **🛠️ AI-Engineer Mode (Full Architectural Tuning)**: Granular control over hidden dimensions, intermediate (MLP) size, layer count, Q attention heads, KV heads (GQA/MQA), vocab size, context length, activation functions (SwiGLU, SiLU, GeLU), and positional RoPE parameters.
- **🔒 Quantify My Life (Personal Dataset Builder)**: Ingests ChatGPT export files (`conversations.json`), WhatsApp chat exports (`.txt`), and local text/markdown notes. Automatically redacts PII (emails, phone numbers, SSNs, API keys) and packages custom fine-tuning datasets (ShareGPT, Alpaca, JSONL).

- **⚡ Real-Time Math & Compatibility Validation**: Immediate warnings for head divisibility errors, GQA ratios, or invalid context length settings.
- **🖥️ Hardware Awareness & Non-Blocking Overrides**: Detects local GPU/VRAM capacity and displays advisories, while allowing expert users targeting high-spec remote server rigs to dismiss local limits.
- **📦 QLoRA Trainer Command Center Integration**: Generates complete model repositories with random initialized weights (`model.safetensors`), `config.json`, tokenizer config, `qlora_trainer_spec.json`, and personal dataset files ready for **QLoRA Trainer Command Center**.


---

## 📦 Precompiled Binaries (Standalone Executables)

Due to GitHub's file size limitations (100MB per file), the precompiled executables (`MakeMyAI.exe` and `SLM_LLM_Architect_Studio.exe`) are stored in the `dist/` directory as split multi-volume 7z archives (`.7z.001`, `.7z.002`, etc.).

### How to Extract & Run:
1. Make sure you have **[7-Zip](https://www.7-zip.org/)** (or another archive utility that supports split volumes) installed.
2. Download all parts for the executable you want to run (e.g., `MakeMyAI.7z.001`, `MakeMyAI.7z.002`, etc.) into the same folder.
3. Right-click on the first volume (the one ending in `.7z.001`) and select **7-Zip -> Extract Here** (or extract to a folder).
4. Run the reconstructed `.exe` file directly. No Python installation or dependency setup is required for the standalone versions.

---

## 🚀 How to Run (From Source)

### Requirements
- Python 3.9+
- `PySide6`
- `torch`
- `transformers`
- `psutil`

### Quick Start
Double-click `Launch_SLM_Architect.bat` or run in terminal:

```bash
python app.py
```

