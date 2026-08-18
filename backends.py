"""LLM backends.

Both backends take a screenshot plus a short list of recent things the model
already said, and return either a comment or the literal string "PASS" meaning
"nothing worth interrupting for right now".

The system prompt is where the "act like a co-worker, not a chatbot" behavior
lives. Tune it freely — it's the highest-leverage knob in the whole app.
"""
from __future__ import annotations

import base64
import io
import re
from abc import ABC, abstractmethod

import requests
from PIL import Image

from config import Config

SYSTEM_PROMPT = """You are an ambient co-working partner. A teammate has shared \
a live view of their screen with you. Every so often you receive a screenshot of \
what they're currently doing.

Behave like a thoughtful colleague sitting next to them: mostly quiet, but \
occasionally offering something genuinely useful. If the screen clearly displays\
a new question or problem, provide the answer or solution with proper reasoning. \
Otherwise, if the screen is routine, substantively unchanged, or you have nothing \
useful to add, reply with EXACTLY: PASS

Speak up ONLY when the \
interruption is clearly worth it, e.g.:
  - a sharp question that would improve their thinking
  - a concrete, specific suggestion or next step
  - a connection to something else on screen or a relevant idea
  - a likely mistake, bug, or risk you notice

Stay silent when the screen is routine, substantively unchanged, or when you'd \
just be narrating what they can already see. Saying nothing is usually the \
correct choice.

Hard rules:
- Answer all new questions or problems with proper reasoning and a clear solution.
- If you have nothing worth saying, reply with EXACTLY: PASS
- Otherwise reply in at most two short sentences. No preamble, no "I see that",
  no restating the screen back to them.
- Never repeat a point you've recently made.
- Be specific to what is actually on the screen."""


def _instruction(recent: list[str]) -> str:
    if recent:
        joined = "\n".join(f"- {r}" for r in recent)
        recent_block = f"Recent things you already said (do NOT repeat):\n{joined}"
    else:
        recent_block = "You haven't said anything yet."
    return (
        "Here is the current screen.\n\n"
        f"{recent_block}\n\n"
        "If the current screen clearly displays a new question or problem, provide the answer or solution with proper reasoning. Otherwise, if the screen is routine, unchanged, or you have nothing useful to add, reply with EXACTLY: PASS\n\n"
    )


def _png_b64(img: Image.Image, scale: float) -> str:
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Backend(ABC):
    PASS = "PASS"

    @abstractmethod
    def comment(self, image: Image.Image, recent: list[str]) -> str:
        """Return a comment, or self.PASS if there's nothing to say."""


class ClaudeBackend(Backend):
    def __init__(self, cfg: Config) -> None:
        try:
            import anthropic
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run: uv add anthropic"
            ) from e
        # Reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic()
        self.cfg = cfg

    def comment(self, image: Image.Image, recent: list[str]) -> str:
        b64 = _png_b64(image, self.cfg.capture_scale)
        resp = self.client.messages.create(
            model=self.cfg.claude_model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png",
                                "data": b64}},
                    {"type": "text", "text": _instruction(recent)},
                ],
            }],
        )
        # Concatenate any text blocks in the response.
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


class LlamaCppBackend(Backend):
    """Talks to a running `llama-server` via its OpenAI-compatible endpoint.

    Start the server with a vision model + its multimodal projector, e.g.:
        llama-server -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF --port 8080
    or with local files:
        llama-server -m model.gguf --mmproj mmproj.gguf --port 8080
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def comment(self, image: Image.Image, recent: list[str]) -> str:
        b64 = _png_b64(image, self.cfg.capture_scale)
        data_uri = f"data:image/png;base64,{b64}"
        r = requests.post(
            f"{self.cfg.llamacpp_host}/v1/chat/completions",
            json={
                "model": self.cfg.llamacpp_model,  # ignored by llama-server
                "stream": False,
                # The thinking trace and the reply share one output budget, so
                # thinking needs its own headroom on top of the reply cap —
                # otherwise the model burns every token inside <think> and
                # content comes back empty.
                "max_tokens": self.cfg.max_tokens
                + (self.cfg.thinking_budget if self.cfg.enable_thinking else 0),
                # Fed into the --jinja chat template per request; Qwen-style
                # templates use enable_thinking.
                "chat_template_kwargs": {
                    "enable_thinking": self.cfg.enable_thinking,
                },
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": _instruction(recent)},
                        {"type": "image_url",
                         "image_url": {"url": data_uri}},
                    ]},
                ],
            },
            timeout=self.cfg.request_timeout,
        )
        r.raise_for_status()
        # content is null when every generated token was reasoning (i.e. the
        # thinking budget was still too small).
        text = r.json()["choices"][0]["message"]["content"] or ""
        # llama-server usually routes reasoning to reasoning_content, but if a
        # template leaks the <think> block into content, drop it (including a
        # block truncated mid-thought by the token limit).
        text = re.sub(r"<think>.*?(?:</think>|\Z)", "", text, flags=re.DOTALL)
        return text.strip()


def make_backend(cfg: Config) -> Backend:
    if cfg.backend == "claude":
        return ClaudeBackend(cfg)
    if cfg.backend == "llamacpp":
        return LlamaCppBackend(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend!r}")


def is_pass(text: str) -> bool:
    """True if the model declined to comment."""
    t = text.strip().strip(".").upper()
    return t == "" or t == "PASS"
