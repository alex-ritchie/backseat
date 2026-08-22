"""Registry of local VLM presets shown in the model dropdown.

Each preset knows the Hugging Face GGUF repo (llama-server's `-hf` argument) plus
any model-specific flags. To add a model, drop a new ModelPreset in PRESETS —
that's the whole extension point.

Two of these are verified against current GGUF repos (the two Qwen3-VL sizes we
picked); the ones marked TEMPLATE are the right shape but you should confirm the
exact repo:tag on Hugging Face before relying on them, since tags move around.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Flags shared by every preset. Per-preset extras are appended after these, so a
# preset can add (or effectively override) anything here.
SHARED_ARGS: list[str] = [
    "--n-gpu-layers", "99",   # put all layers on the GPU
    "--flash-attn", "on",
    "--jinja",                # use the model's chat template
    "--ctx-size", "65536",
]


@dataclass(frozen=True)
class ModelPreset:
    key: str                              # stable id used internally
    label: str                            # shown in the dropdown
    hf_repo: str                          # value for llama-server -hf
    extra_args: list[str] = field(default_factory=list)


PRESETS: list[ModelPreset] = [
    ModelPreset(
        key="qwen38-27b-q4",
        label="Qwen3.8-27B · Q4_K_M (latest, quality)",
        # ~17GB at Q4; DeltaNet linear attention keeps the KV cache tiny.
        # NOTE: needs a recent llama.cpp build (older CUDA builds mis-run the
        # DeltaNet layers). Vision needs the mmproj — if the repo doesn't bundle
        # one that -hf auto-loads, add: "--mmproj", "/path/to/mmproj-...f16.gguf".
        # Thinking mode is ON by default; Config.enable_thinking controls it
        # (the backend passes it via chat_template_kwargs and sizes the token
        # budget accordingly).
        hf_repo="unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
        # Qwen3.8 ships an MTP head; 2 is the conservative starting point
        # recommended by Unsloth, with 1-6 worth benchmarking per machine.
        extra_args=["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    ),
    ModelPreset(
        key="qwen3vl-8b-q4",
        label="Qwen3-VL-8B-Instruct · UD-Q4_K_XL (fast, stable)",
        hf_repo="unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL",
    ),
    ModelPreset(
        key="qwen3vl-32b-q4",
        label="Qwen3-VL-32B-Instruct · Q4_K_M (prev-gen quality)",
        hf_repo="Qwen/Qwen3-VL-32B-Instruct-GGUF:Q4_K_M",
        extra_args=["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    ),
    # --- TEMPLATE preset: confirm the repo:tag on Hugging Face before use ---
    ModelPreset(
        key="gemma3-12b-q4",
        label="Gemma 3 12B-it · Q4_K_M (alt)",
        hf_repo="unsloth/gemma-3-12b-it-GGUF:Q4_K_M",
    ),
]

# The model launched on startup.
DEFAULT_KEY = "qwen38-27b-q4"


def preset_by_key(key: str) -> ModelPreset:
    for p in PRESETS:
        if p.key == key:
            return p
    raise KeyError(f"No model preset with key {key!r}")


def build_command(binary: str, port: int, preset: ModelPreset) -> list[str]:
    """Full argv for launching llama-server with this preset."""
    return [
        binary,
        "-hf", preset.hf_repo,
        "--port", str(port),
        *SHARED_ARGS,
        *preset.extra_args,
    ]