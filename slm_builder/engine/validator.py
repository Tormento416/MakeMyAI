"""
Validation Engine for hyperparameter sanity checking & hardware advisories.
Distinguishes between hard errors (invalid architecture math) and soft warnings (hardware constraints).
"""

from typing import Any

import psutil

from .config import ArchitectureConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ValidationSeverity:
    ERROR = "ERROR"       # Hard error - Model cannot be constructed (e.g. invalid math)
    WARNING = "WARNING"   # Software / Math warning (suboptimal config)
    HARDWARE = "HARDWARE" # Exceeds local hardware limits (can be ignored for remote training)

class ValidationIssue:
    def __init__(self, severity: str, message: str, code: str, can_ignore: bool = False):
        self.severity = severity
        self.message = message
        self.code = code
        self.can_ignore = can_ignore

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "code": self.code,
            "can_ignore": self.can_ignore
        }

def get_system_hardware_info() -> dict[str, Any]:
    """Detects local CPU RAM and GPU VRAM capacity."""
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    vram_gb = 0.0
    gpu_name = "N/A (CPU only)"
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 2)
        
    return {
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "gpu_name": gpu_name,
        "has_cuda": TORCH_AVAILABLE and torch.cuda.is_available()
    }

def validate_architecture(cfg: ArchitectureConfig, target_remote: bool = False) -> tuple[bool, list[ValidationIssue]]:
    """
    Validates model configuration against math rules and hardware resources.
    Returns (is_valid, list_of_issues).
    """
    issues: list[ValidationIssue] = []
    
    # 1. Math & Divisibility checks (HARD ERRORS)
    if cfg.hidden_size <= 0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Hidden size must be > 0.", "INVALID_HIDDEN_SIZE"))
        
    if cfg.num_attention_heads <= 0:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Number of attention heads must be > 0.", "INVALID_HEADS"))
        
    if (
        cfg.hidden_size > 0
        and cfg.num_attention_heads > 0
        and cfg.hidden_size % cfg.num_attention_heads != 0
    ):
        issues.append(ValidationIssue(
            ValidationSeverity.ERROR,
            f"Hidden size ({cfg.hidden_size}) must be divisible by attention heads ({cfg.num_attention_heads}). "
            f"Current head dimension would be fractional ({cfg.hidden_size / cfg.num_attention_heads:.2f}).",
            "HIDDEN_HEAD_DIVISIBILITY"
        ))
            
    # GQA / MQA checks
    if cfg.num_key_value_heads is not None and cfg.num_key_value_heads > 0:
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Attention heads ({cfg.num_attention_heads}) must be a multiple of KV heads ({cfg.num_key_value_heads}) for Grouped Query Attention (GQA).",
                "INVALID_GQA_RATIO"
            ))
        if cfg.num_key_value_heads > cfg.num_attention_heads:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"KV heads ({cfg.num_key_value_heads}) cannot be greater than attention heads ({cfg.num_attention_heads}).",
                "KV_EXCEEDS_ATTN_HEADS"
            ))
            
    if cfg.vocab_size < 100:
        issues.append(ValidationIssue(ValidationSeverity.ERROR, "Vocabulary size must be at least 100.", "VOCAB_TOO_SMALL"))

    # 2. Performance & Recommendation Warnings (SOFT WARNINGS)
    head_dim = cfg.hidden_size // cfg.num_attention_heads if cfg.num_attention_heads > 0 else 0
    if head_dim not in [32, 64, 128, 256] and head_dim > 0:
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            f"Head dimension is {head_dim}. Standard head dimensions for optimal CUDA / FlashAttention speed are 64 or 128.",
            "NON_STANDARD_HEAD_DIM",
            can_ignore=True
        ))
        
    if cfg.max_position_embeddings > 16384 and cfg.rope_scaling_type is None:
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            f"Context length is very large ({cfg.max_position_embeddings}) without RoPE scaling. Consider enabling RoPE scaling (e.g. Dynamic or YaRN).",
            "LARGE_CONTEXT_NO_ROPE",
            can_ignore=True
        ))

    # 3. Hardware Resource Checks (HARDWARE ADVISORIES)
    hw = get_system_hardware_info()
    estimates = cfg.calculate_parameter_count()
    
    if not target_remote:
        if hw["vram_gb"] > 0:
            # Checking local VRAM
            if estimates["qlora_training_vram_gb"] > hw["vram_gb"]:
                issues.append(ValidationIssue(
                    ValidationSeverity.HARDWARE,
                    f"Estimated QLoRA fine-tuning memory (~{estimates['qlora_training_vram_gb']} GB VRAM) exceeds detected local GPU memory ({hw['vram_gb']} GB VRAM on {hw['gpu_name']}).",
                    "EXCEEDS_LOCAL_VRAM",
                    can_ignore=True
                ))
            elif estimates["weight_vram_fp16_gb"] > hw["vram_gb"]:
                issues.append(ValidationIssue(
                    ValidationSeverity.HARDWARE,
                    f"Raw model weight size (~{estimates['weight_vram_fp16_gb']} GB FP16) exceeds local GPU VRAM ({hw['vram_gb']} GB).",
                    "EXCEEDS_LOCAL_INFERENCE_VRAM",
                    can_ignore=True
                ))
        else:
            # CPU fallback memory check
            if estimates["weight_vram_fp16_gb"] > hw["ram_gb"] * 0.8:
                issues.append(ValidationIssue(
                    ValidationSeverity.HARDWARE,
                    f"Model memory requirement (~{estimates['weight_vram_fp16_gb']} GB) exceeds 80% of local system RAM ({hw['ram_gb']} GB).",
                    "EXCEEDS_LOCAL_RAM",
                    can_ignore=True
                ))

    is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    return is_valid, issues
