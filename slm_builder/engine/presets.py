"""
Preset engine for Easy Mode.
Provides smart architectural configurations based on target model scale & hardware constraints.
"""

from .config import ArchitectureConfig


class ModelPresets:
    @staticmethod
    def get_presets() -> dict[str, ArchitectureConfig]:
        return {
            "slm_micro_50m": ArchitectureConfig(
                model_name="SLM-Micro-50M",
                architecture_type="llama",
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1536,
                num_hidden_layers=12,
                num_attention_heads=8,
                num_key_value_heads=8,
                max_position_embeddings=2048,
                hidden_act="silu",
                tie_word_embeddings=True
            ),
            "slm_small_150m": ArchitectureConfig(
                model_name="SLM-Small-150M",
                architecture_type="llama",
                vocab_size=32000,
                hidden_size=768,
                intermediate_size=2048,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_key_value_heads=12,
                max_position_embeddings=4096,
                hidden_act="silu",
                tie_word_embeddings=True
            ),
            "slm_compact_350m": ArchitectureConfig(
                model_name="SLM-Compact-350M",
                architecture_type="llama",
                vocab_size=32000,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=16,
                num_attention_heads=16,
                num_key_value_heads=16,
                max_position_embeddings=4096,
                hidden_act="silu",
                tie_word_embeddings=False
            ),
            "slm_standard_1b": ArchitectureConfig(
                model_name="SLM-Standard-1B",
                architecture_type="llama",
                vocab_size=32000,
                hidden_size=2048,
                intermediate_size=5632,
                num_hidden_layers=18,
                num_attention_heads=16,
                num_key_value_heads=4,  # GQA
                max_position_embeddings=8192,
                hidden_act="silu",
                rope_theta=10000.0,
                tie_word_embeddings=False
            ),
            "llm_base_3b": ArchitectureConfig(
                model_name="LLM-Base-3B",
                architecture_type="llama",
                vocab_size=32000,
                hidden_size=3072,
                intermediate_size=8192,
                num_hidden_layers=28,
                num_attention_heads=24,
                num_key_value_heads=8,  # GQA
                max_position_embeddings=8192,
                hidden_act="silu",
                rope_theta=500000.0,
                tie_word_embeddings=False
            )
        }

    @staticmethod
    def recommend_for_hardware(vram_gb: float) -> str:
        """Recommends an Easy Mode preset based on local VRAM."""
        if vram_gb <= 4.0:
            return "slm_micro_50m"
        elif vram_gb <= 8.0:
            return "slm_small_150m"
        elif vram_gb <= 12.0:
            return "slm_compact_350m"
        elif vram_gb <= 16.0:
            return "slm_standard_1b"
        else:
            return "llm_base_3b"
