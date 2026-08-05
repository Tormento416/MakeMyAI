# 📘 Beginner's Guide & Manual: MakeMyAI Studio

Welcome to **MakeMyAI Studio**! This guide will walk you step-by-step through designing your own Small Language Model (SLM) or Large Language Model (LLM), converting your personal files into a training dataset, and getting everything ready for fine-tuning in your **QLoRA Trainer Command Center**.


---

## 💡 Key Concepts Explained Simply

### What is an SLM vs. LLM?
- **SLM (Small Language Model)**: Models under 2 billion parameters (e.g. 50M, 150M, 350M, 1B). They run extremely fast on regular PCs, laptops, and mobile devices, requiring very little GPU memory.
- **LLM (Large Language Model)**: Models over 3 billion parameters. They have vast knowledge but require powerful GPUs with 12GB–24GB+ VRAM.

### What does "Untrained Architecture" mean?
This tool builds the **structure** and **blank neural network weights** for a model. It creates everything needed (the `config.json`, tokenizer, `.safetensors` weight files) so that you or your users can train the model from scratch on any custom dataset you choose.

---

## 🚀 Quick Start Tutorial: Build Your First Model (5 Minutes)

### Step 1: Open the Application
Launch the standalone app by double-clicking `dist\SLM_LLM_Architect_Studio.exe` or `Launch_SLM_Architect.bat`.

### Step 2: Choose Easy Mode
1. Keep **✨ Easy Mode** selected at the top.
2. Select your desired model size from the dropdown:
   - **SLM Micro (~50M Params)**: Perfect for ultra-fast testing and mobile devices.
   - **SLM Small (~150M Params)**: Great standard balance for laptops and CPUs.
   - **SLM Standard (~1.0B Params)**: High capability model for gaming GPUs (8GB+ VRAM).
3. The app will automatically calculate the optimal layer counts and head sizes for your system.

### Step 3: Instantiate & Save Model
1. Choose an output folder (e.g. `./built_models/my_first_slm`).
2. Click **🚀 Instantiate & Package Model for QLoRA Trainer**.
3. In a few seconds, your new model files (`model.safetensors`, `config.json`, `tokenizer.json`, `qlora_trainer_spec.json`) will be generated!

---

## 🔒 How to "Quantify Your Life" (Personal Dataset Builder)

You can turn your personal chat logs, journal entries, or notes into a dataset to train your model to act like a personal AI digital twin.

1. Click on the **🔒 Quantify My Life (Dataset Builder)** tab at the top.
2. Choose your **Source Type**:
   - **ChatGPT Export**: Download your data from ChatGPT settings (`conversations.json`).
   - **WhatsApp Chat Export**: Export chat history from WhatsApp as a `.txt` file.
   - **Text / Markdown Folder**: Select a folder containing `.txt` or `.md` notes.
3. Keep **"Automatically Redact Sensitive PII"** checked to ensure private emails, phone numbers, and keys are automatically removed.
4. Click **✨ Process & Quantify Personal Life Dataset**.
5. Your cleaned dataset (`personal_life_dataset.jsonl`) will be saved directly alongside your model output folder!

---

## 📱 Device Sync & Terms of Service Notice

The **📱 Device Sync & Account** tab allows you to sign in with your email to connect secondary devices (smartphones, tablets, smart watches).

> **⚠️ Terms & Responsibility Disclaimer**:  
> The model and datasets built with this app are your private personal property. By signing in and connecting external devices or cloud services, you acknowledge that you accept full responsibility for connected devices and third-party account services.

---

## ❓ Frequently Asked Questions (FAQ)

**Q: Can I train the model directly inside this app?**  
*A: This app builds the model structure and prepares the datasets. To train the model, open your generated model folder inside the **QLoRA Trainer Command Center**.*

**Q: Can I build a model larger than my PC's GPU memory?**  
*A: Yes! Switch to **🛠️ AI-Engineer Mode** and check the box **"Ignore Hardware Memory Limits (Targeting Remote Server Rig)"**. This allows you to design large models on your PC that you plan to train on cloud servers or dedicated GPU rigs.*

