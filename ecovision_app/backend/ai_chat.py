# backend/ai_chat.py
from __future__ import annotations

import os
from typing import AsyncGenerator, Optional
import httpx

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Point to your local Ollama server; override with OLLAMA_URL env var
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# Default LLM model; override with LLM_MODEL env var
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")


# ----------------------------------------------------------------------
# Non-streaming: return a full answer
# ----------------------------------------------------------------------

async def ask(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 768,
    temperature: float = 0.2,
) -> str:
    """
    Send a single prompt to Ollama and return the full response text.
    """
    payload = {
        "model": MODEL,
        "prompt": _compose_prompt(prompt, system),
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()


# ----------------------------------------------------------------------
# Streaming: yield chunks as they arrive
# ----------------------------------------------------------------------

async def stream_ask(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 768,
    temperature: float = 0.2,
) -> AsyncGenerator[str, None]:
    """
    Stream an answer from Ollama chunk by chunk.
    Each yielded string is part of the response text.
    """
    payload = {
        "model": MODEL,
        "prompt": _compose_prompt(prompt, system),
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = httpx.Response(200, content=line).json()
                except Exception:
                    continue
                piece = chunk.get("response")
                if piece:
                    yield piece


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _compose_prompt(user_prompt: str, system: Optional[str]) -> str:
    """
    Combine system and user messages into a single string.
    Ollama does not natively support role separation like OpenAI,
    so we prepend system text explicitly.
    """
    if system:
        return f"System:\n{system}\n\nUser:\n{user_prompt}\n\nAssistant:"
    return user_prompt
