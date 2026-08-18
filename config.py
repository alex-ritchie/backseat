"""Central configuration for the co-work watcher.

Everything the app's behavior depends on lives here so it can be tuned in one
place (and surfaced in the GUI). Instances are cheap value objects; the GUI
builds a fresh one each time you press Start.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- LLM backend -------------------------------------------------------
    backend: str = "claude"                # "claude" or "llamacpp"
    claude_model: str = "claude-sonnet-5"  # change to whatever you have access to
    llamacpp_model: str = "local-vlm"      # label only; llama-server serves one model
    max_tokens: int = 300                  # cap on the *visible* reply (output
                                           # tokens; applies to both backends)
    enable_thinking: bool = True           # let the local model think first
    thinking_budget: int = 2048            # extra output tokens for the <think>
                                           # trace (llamacpp only)
    request_timeout: float = 300.0         # HTTP timeout (s) for llamacpp calls;
                                           # sized for thinking_budget at ~20 tok/s

    # --- Local llama-server management ------------------------------------
    llama_server_bin: str = "llama-server"  # must be on PATH (or give a full path)
    server_port: int = 8080                 # llama-server port; backend host derives
    autostart_server: bool = True           # launch the default model on app start

    @property
    def llamacpp_host(self) -> str:
        return f"http://localhost:{self.server_port}"

    # --- Watch loop timing -------------------------------------------------
    poll_interval: float = 2.0             # seconds between screen samples
    min_seconds_between_calls: float = 8.0  # hard cooldown between LLM calls
    stabilize_frames: int = 1              # stable samples required after a change
                                           # before we call (avoids mid-scroll shots)

    # --- Change detection --------------------------------------------------
    change_threshold: float = 6.0          # mean abs pixel diff (0-255) = "changed"

    # --- Image / context ---------------------------------------------------
    capture_scale: float = 1.0             # downscale sent images (0.5 = half size)
    history_turns: int = 4                 # recent comments fed back to avoid repeats
