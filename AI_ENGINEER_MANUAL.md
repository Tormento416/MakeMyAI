# 🛠️ AI-Engineer's Architecture & Settings Manual: MakeMyAI Studio

This manual provides an in-depth reference for all architecture parameters, controls, and mathematical settings in **MakeMyAI Studio**.


---

## 🎛️ AI-Engineer Mode Parameter Reference

### 1. Architecture Type
- **`llama`**: Standard decoder-only architecture utilizing RMSNorm, Rotary Positional Embeddings (RoPE), and SwiGLU activations. Highly compatible with Hugging Face and vLLM.
- **`mistral`**: High-efficiency variant with sliding window attention and Grouped-Query Attention.
- **`gpt2`**: Classic decoder-only transformer utilizing GELU activations and standard LayerNorm.

---

### 2. Core Model Dimensions

#### **Hidden Size (`d_model`)**
- **What it is**: The internal vector embedding size for every token passing through the neural network.
- **Standard Values**: `512` (Micro), `768` (Small), `1024` (Compact), `2048` (1B), `3072` (3B), `4096` (7B), `8192` (70B).
- **Rule**: Must be cleanly divisible by the **Attention Heads** count.

#### **Intermediate / MLP Size (`d_ff`)**
- **What it is**: The expansion dimension inside the Feed-Forward / MLP layers where model reasoning occurs.
- **Standard Ratio**: Usually $\sim 8/3 \times \text{Hidden Size}$ (e.g. 5632 for 2048 hidden size) or $4 \times \text{Hidden Size}$ for standard MLPs.

#### **Hidden Layers (`n_layers`)**
- **What it is**: The vertical depth of the neural network. Higher layer counts allow for deeper reasoning and abstractions.
- **Standard Values**: `12` (Small), `16` (Compact), `28` (3B), `32` (7B), `80` (70B).

---

### 3. Attention Mechanisms

#### **Attention Heads (`num_attention_heads`)**
- **What it is**: The number of parallel Query attention channels splitting the hidden dimension.
- **Head Dimension Formula**: $\text{Head Dim} = \frac{\text{Hidden Size}}{\text{Attention Heads}}$.
- **Optimal Hardware Dim**: `64` or `128` (maximizes Tensor Core and FlashAttention CUDA speed).

#### **KV Heads (`num_key_value_heads`) — GQA / MQA**
- **Multi-Head Attention (MHA)**: `KV Heads == Attention Heads`. Highest quality, but uses more VRAM during inference KV caching.
- **Grouped-Query Attention (GQA)**: `KV Heads < Attention Heads` (e.g., 8 KV heads for 32 Q heads). Reduces VRAM memory usage by 75% during inference with minimal quality loss.
- **Multi-Query Attention (MQA)**: `KV Heads == 1`. Maximum memory savings for extreme edge devices.

---

### 4. Vocabulary & Context Length

#### **Vocab Size (`vocab_size`)**
- **What it is**: The total number of unique token sub-words in the model's dictionary.
- **Standard Values**: `32000` (LLaMA/Mistral), `50257` (GPT-2), `128256` (LLaMA 3).

#### **Max Context Length (`max_position_embeddings`)**
- **What it is**: The maximum sequence length (in tokens) the model can read or generate in a single prompt.
- **Standard Values**: `2048`, `4096`, `8192`, `16384`, `32768`.

---

### 5. Activations & Weights

#### **Activation Functions (`hidden_act`)**
- **`silu` / `swiglu`**: Modern gated activations (used by LLaMA and Mistral). Provides superior convergence during training.
- **`gelu`**: Standard activation function used by GPT-2, GPT-3, and BERT.

#### **Tie Word Embeddings**
- **Enabled (`True`)**: Shares weight matrices between token input embeddings and output LM head. Reduces parameter count by $\text{Vocab Size} \times \text{Hidden Size}$ (useful for micro SLMs).
- **Disabled (`False`)**: Keeps separate input/output embedding weights (standard for models $\ge 1\text{B}$).

---

## 🖥️ Real-Time Estimations & Diagnostic Messages

| Diagnostic Code | Type | Meaning & Action Required |
| :--- | :--- | :--- |
| `HIDDEN_HEAD_DIVISIBILITY` | **ERROR** | Hidden size must be divisible by attention heads. Adjust hidden size or head count. |
| `INVALID_GQA_RATIO` | **ERROR** | Attention heads must be a multiple of KV heads. |
| `NON_STANDARD_HEAD_DIM` | **WARNING** | Head dimension is not 64 or 128. FlashAttention speed may drop. |
| `EXCEEDS_LOCAL_VRAM` | **HARDWARE** | Estimated QLoRA training memory exceeds local GPU VRAM. Check *"Ignore Hardware Memory Limits"* if training on a remote server rig. |
