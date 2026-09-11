"""An in-process model, so soup does not need a server.

The seat used to HTTP-chat with ollama. That made a running server a hard
dependency of reading a sentence, which is daft when the weights are already
on the machine. This module loads them.

On Apple Silicon the loader is MLX, and `qwen3.5:4b` means
`mlx-community/Qwen3.5-4B-MLX-4bit`. A Hugging Face id is used as-is. A path
to a GGUF still goes through llama.cpp, for the days when that loader can
actually read the file ollama wrote.

Failed loads are the seat's problem. It falls back to HTTP if a server is
there, and goes deaf if it is not.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

__all__ = ["find_weights", "load", "chat", "OLLAMA_ROOT", "MLX_ALIASES"]

OLLAMA_ROOT = os.path.expanduser("~/.ollama/models")

# ollama-style names -> MLX repos. The GGUF ollama stores for qwen3.5 is a
# slightly different export than stock llama.cpp will load (rope sections of
# length 3, not 4), so the same name has to mean a different file when we
# run it ourselves.
MLX_ALIASES = {
    "qwen3.5:4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen3.5:0.8b": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "qwen3.5:2b": "mlx-community/Qwen3.5-2B-MLX-4bit",
}


def find_weights(
    name: str = "",
    weights: Optional[str] = None,
    root: str = OLLAMA_ROOT,
) -> Optional[str]:
    """Turn a model name into a GGUF path, or None if we do not have one."""
    if weights and _is_gguf(weights):
        return os.path.abspath(weights)
    if name and _is_gguf(name):
        return os.path.abspath(name)
    env = os.environ.get("SOUP_LLM_WEIGHTS")
    if env and _is_gguf(env):
        return os.path.abspath(env)
    if name:
        found = _ollama_blob(name, root)
        if found:
            return found
    return None


def mlx_id(name: str = "", weights: Optional[str] = None) -> Optional[str]:
    """A Hugging Face / MLX repo to load, if this name is one of those."""
    if weights and not _is_gguf(weights) and not os.path.isabs(weights) and "/" in weights:
        return weights
    if name in MLX_ALIASES:
        return MLX_ALIASES[name]
    if not name or name.startswith("/") or name.startswith(".") or name.endswith(".gguf"):
        return None
    if name.count("/") == 1:
        return name
    return None


def load(name: str = "", weights: Optional[str] = None) -> Optional[dict]:
    """Load whatever we can for this name. Returns None if nothing fits."""
    repo = mlx_id(name, weights)
    if repo:
        try:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            from mlx_lm import load as mlx_load
        except ImportError:
            repo = None
        else:
            model, tokenizer = mlx_load(repo)
            return {"kind": "mlx", "model": model, "tokenizer": tokenizer, "id": repo}

    path = find_weights(name, weights)
    if path:
        from llama_cpp import Llama

        llm = Llama(
            model_path=path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False,
            logits_all=False,
        )
        return {"kind": "gguf", "model": llm, "path": path}
    return None


def chat(engine: dict, system: str, utterance: str) -> str:
    """One completion. Thinking stays off; this is a line of syntax."""
    if engine["kind"] == "mlx":
        return _chat_mlx(engine, system, utterance)
    return _chat_gguf(engine["model"], system, utterance)


def _chat_mlx(engine: dict, system: str, utterance: str) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    tokenizer = engine["tokenizer"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": utterance},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = generate(
        engine["model"],
        tokenizer,
        prompt=prompt,
        max_tokens=300,
        sampler=make_sampler(temp=0),
        verbose=False,
    )
    return _strip_think(text)


def _chat_gguf(llm: Any, system: str, utterance: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": utterance},
    ]
    kwargs = dict(messages=messages, temperature=0, max_tokens=300)
    try:
        reply = llm.create_chat_completion(
            chat_template_kwargs={"enable_thinking": False},
            **kwargs,
        )
    except TypeError:
        reply = llm.create_chat_completion(**kwargs)
    content = reply["choices"][0]["message"]["content"] or ""
    return _strip_think(content)


def _strip_think(text: str) -> str:
    """Qwen3 will soliloquise in <think> tags if thinking was not actually off."""
    if "<think>" in text and "</think>" in text:
        head, _, rest = text.partition("<think>")
        text = (head + rest.partition("</think>")[2]).strip()
    return text.strip()


def _is_gguf(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def _ollama_blob(name: str, root: str) -> Optional[str]:
    """The GGUF ollama already downloaded for this name, if any."""
    manifest = _ollama_manifest(name, root)
    if manifest is None or not os.path.isfile(manifest):
        return None
    try:
        with open(manifest) as fh:
            data = json.loads(fh.read())
    except (OSError, ValueError):
        return None
    blobs = os.path.join(root, "blobs")
    for layer in data.get("layers") or ():
        if layer.get("mediaType") != "application/vnd.ollama.image.model":
            continue
        digest = (layer.get("digest") or "").replace(":", "-")
        path = os.path.join(blobs, digest)
        if _is_gguf(path):
            return path
    return None


def _ollama_manifest(name: str, root: str) -> Optional[str]:
    """qwen3.5:4b -> ~/.ollama/models/manifests/registry.ollama.ai/library/qwen3.5/4b"""
    if not name:
        return None
    host = "registry.ollama.ai"
    rest = name
    if "/" in name and "." in name.split("/", 1)[0]:
        host, rest = name.split("/", 1)
    if "/" not in rest:
        library, rest = "library", rest
    else:
        library, rest = rest.split("/", 1)
    if ":" in rest:
        model, tag = rest.split(":", 1)
    else:
        model, tag = rest, "latest"
    return os.path.join(root, "manifests", host, library, model, tag)
