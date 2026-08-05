"""
Model & Tokenizer Builder Engine.
Instantiates generic PyTorch / Hugging Face models from scratch with random weight initialization.
Exports weights, tokenizer configs, and metadata completely compliant with QLoRA Trainer Command Center.
"""

import json
import os
from collections.abc import Callable
from typing import Any

from .config import ArchitectureConfig

try:
    import torch  # noqa: F401  # imported to probe availability
    from transformers import (
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        PreTrainedTokenizerFast,  # noqa: F401  # imported to probe availability
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModelBuilderEngine:
    def __init__(self, config: ArchitectureConfig):
        self.config = config

    def build_hf_config(self):
        """Constructs Hugging Face PretrainedConfig based on architecture settings."""
        cfg = self.config
        
        if cfg.architecture_type in ["llama", "custom_slm"]:
            return LlamaConfig(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads or cfg.num_attention_heads,
                hidden_act=cfg.hidden_act,
                max_position_embeddings=cfg.max_position_embeddings,
                initializer_range=cfg.initializer_range,
                rms_norm_eps=cfg.rms_norm_eps,
                use_cache=True,
                rope_theta=cfg.rope_theta,
                tie_word_embeddings=cfg.tie_word_embeddings,
                attention_bias=cfg.use_bias,
                attention_dropout=cfg.attention_dropout
            )
        elif cfg.architecture_type == "mistral":
            return MistralConfig(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads or cfg.num_attention_heads,
                hidden_act=cfg.hidden_act,
                max_position_embeddings=cfg.max_position_embeddings,
                initializer_range=cfg.initializer_range,
                rms_norm_eps=cfg.rms_norm_eps,
                use_cache=True,
                rope_theta=cfg.rope_theta,
                tie_word_embeddings=cfg.tie_word_embeddings
            )
        elif cfg.architecture_type == "gpt2":
            return GPT2Config(
                vocab_size=cfg.vocab_size,
                n_embd=cfg.hidden_size,
                n_layer=cfg.num_hidden_layers,
                n_head=cfg.num_attention_heads,
                n_positions=cfg.max_position_embeddings,
                activation_function=cfg.hidden_act,
                resid_pdrop=cfg.attention_dropout
            )
        else:
            # Fallback default to LLaMA structure
            return LlamaConfig(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads or cfg.num_attention_heads
            )

    def generate_qlora_trainer_metadata(self) -> dict[str, Any]:
        """Generates target modules and metadata matching QLoRA Trainer Command Center specs."""
        arch = self.config.architecture_type.lower()
        if arch in ["llama", "mistral", "custom_slm"]:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif arch == "gpt2":
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            target_modules = ["q_proj", "v_proj"]

        return {
            "model_name": self.config.model_name,
            "architecture_type": self.config.architecture_type,
            "qlora_compatible": True,
            "suggested_target_modules": target_modules,
            "suggested_target_modules_str": ", ".join(target_modules),
            "torch_dtype": "bfloat16",
            "vocab_size": self.config.vocab_size,
            "context_length": self.config.max_position_embeddings,
            "parameters": self.config.calculate_parameter_count()["total_params_formatted"]
        }

    def build_and_save_model(
        self,
        output_dir: str,
        base_tokenizer_name: str = "gpt2",
        progress_callback: Callable[[str, int], None] | None = None
    ) -> dict[str, Any]:
        """
        Instantiates untrained model weights from scratch and saves full HF repository bundle.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("PyTorch and Hugging Face Transformers packages are required to build model weights.")

        os.makedirs(output_dir, exist_ok=True)
        
        if progress_callback:
            progress_callback("Initializing Model Configuration...", 10)
            
        hf_config = self.build_hf_config()
        
        if progress_callback:
            progress_callback("Instantiating Untrained Architecture with Random Weights...", 30)
            
        # Instantiate model architecture with 0 pretrained weights
        if isinstance(hf_config, LlamaConfig):
            model = LlamaForCausalLM(hf_config)
        elif isinstance(hf_config, MistralConfig):
            model = MistralForCausalLM(hf_config)
        elif isinstance(hf_config, GPT2Config):
            model = GPT2LMHeadModel(hf_config)
        else:
            model = LlamaForCausalLM(hf_config)

        if progress_callback:
            progress_callback("Saving Model Weights (.safetensors / PyTorch)...", 60)
            
        # Save model and config
        model.save_pretrained(output_dir, safe_serialization=True)
        
        if progress_callback:
            progress_callback(f"Configuring Tokenizer (Base: {base_tokenizer_name})...", 80)
            
        # Obtain/Configure tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(output_dir)
        except Exception:
            # Create standard fallback config if offline or tokenizer fetch fails
            tokenizer_meta = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "model_max_length": self.config.max_position_embeddings
            }
            with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_meta, f, indent=2)

        if progress_callback:
            progress_callback("Exporting QLoRA Trainer Command Center Metadata...", 95)
            
        # Save QLoRA Command Center Integration Spec File
        qlora_meta = self.generate_qlora_trainer_metadata()
        with open(os.path.join(output_dir, "qlora_trainer_spec.json"), "w") as f:
            json.dump(qlora_meta, f, indent=2)

        if progress_callback:
            progress_callback("Build Complete!", 100)

        return {
            "output_dir": output_dir,
            "qlora_spec": qlora_meta,
            "status": "SUCCESS"
        }
