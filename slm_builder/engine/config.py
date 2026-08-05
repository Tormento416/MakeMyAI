"""
Model architecture configuration schema and hyperparameter structures.
Supports LLaMA, Mistral, GPT-2, and custom decoder/encoder architectures.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ArchitectureConfig:
    # Basic info
    model_name: str = "custom_slm_model"
    architecture_type: str = "llama"  # llama, mistral, gpt2
    
    # Model size & dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 16  # For GQA/MQA. If equal to num_attention_heads -> MHA
    
    # Positional & Context Length
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling_type: str | None = None  # None, "linear", "dynamic", "yarn"
    rope_scaling_factor: float = 1.0
    
    # Activations & Norms
    hidden_act: str = "silu"  # silu, gelu, gelu_new, relu, swiglu
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Options & Toggles
    tie_word_embeddings: bool = False
    use_bias: bool = False
    attention_dropout: float = 0.0
    
    def calculate_parameter_count(self) -> dict[str, Any]:
        """Calculates exact & estimated parameter counts for decoder-only architectures."""
        v = self.vocab_size
        h = self.hidden_size
        i = self.intermediate_size
        n_layers = self.num_hidden_layers
        a_h = self.num_attention_heads
        kv_h = self.num_key_value_heads or a_h
        head_dim = h // a_h if a_h > 0 else 0
        
        # Token Embeddings
        embedding_params = v * h
        
        # Self Attention Per Layer (Q, K, V, Output Projections)
        q_params = h * (a_h * head_dim)
        k_params = h * (kv_h * head_dim)
        v_params = h * (kv_h * head_dim)
        o_params = (a_h * head_dim) * h
        attn_layer_params = q_params + k_params + v_params + o_params
        
        # MLP / Intermediate Per Layer
        if self.hidden_act in ["silu", "swiglu"]:
            # Gated MLP (Gate, Up, Down projections)
            mlp_layer_params = (h * i) + (h * i) + (i * h)
        else:
            # Standard MLP (Up, Down projections)
            mlp_layer_params = (h * i) + (i * h)
            
        # Layer Norms per layer (2 per layer: input norm + post attn norm)
        norm_layer_params = 2 * h
        
        # Single Layer Total
        single_layer_params = attn_layer_params + mlp_layer_params + norm_layer_params
        
        # Final Norm + Output LM Head
        final_norm_params = h
        lm_head_params = 0 if self.tie_word_embeddings else (v * h)
        
        total_params = embedding_params + (n_layers * single_layer_params) + final_norm_params + lm_head_params
        
        # VRAM Requirements Estimation
        # Weights in FP16 / BF16 (2 bytes per param)
        weight_vram_gb = (total_params * 2) / (1024 ** 3)
        # Optimizer state (AdamW FP32: 8 bytes per param + 4 bytes FP32 grads) -> 12 bytes per param
        optimizer_vram_gb = (total_params * 12) / (1024 ** 3)
        # Baseline KV Cache VRAM per batch_size=1, seq_len=max_position_embeddings (FP16)
        kv_cache_bytes = 2 * n_layers * 2 * kv_h * head_dim * self.max_position_embeddings * 2  # 2 for K and V
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)
        
        return {
            "total_params": total_params,
            "total_params_formatted": f"{total_params / 1e6:.2f}M" if total_params < 1e9 else f"{total_params / 1e9:.2f}B",
            "embedding_params": embedding_params,
            "layer_params": single_layer_params,
            "weight_vram_fp16_gb": round(weight_vram_gb, 2),
            "full_training_vram_gb": round(weight_vram_gb + optimizer_vram_gb + 2.0, 2),  # +2GB headroom
            "qlora_training_vram_gb": round((weight_vram_gb * 0.25) + 3.0 + kv_cache_gb, 2), # 4-bit base + LoRA states
            "kv_cache_gb_per_seq": round(kv_cache_gb, 2),
            "head_dim": head_dim
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
