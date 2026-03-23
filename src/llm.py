"""
src/llm.py
==========
Unified LLM interface that supports two backends:
  - **llamacpp** — runs GGUF-format models via ``llama-cpp-python``.
    Works on CPU and GPU (CUDA / Metal).
  - **transformers** — runs HuggingFace models via ``transformers`` + ``torch``.
    Supports 4-bit / 8-bit quantisation via ``bitsandbytes``.

Usage
-----
    from src.llm import LLMBackend
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    llm = LLMBackend(cfg.llm)

    reply = llm.generate(
        system_prompt="You are a witty bot.",
        user_message="What's 2 + 2?",
        temperature=0.7,
    )
    print(reply)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.config_manager import LLMConfig

logger = logging.getLogger(__name__)


class LLMBackend:
    """Wraps either llama-cpp-python or HuggingFace transformers.

    The model is loaded lazily on the first call to :meth:`generate` (or you
    can force it by calling :meth:`load` explicitly).

    Parameters
    ----------
    config:
        The ``llm`` section of the bot configuration.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._cfg = config
        self._model = None   # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model into memory.  Call once at startup to avoid cold starts."""
        if self._model is not None:
            return
        if self._cfg.backend == "llamacpp":
            self._load_llamacpp()
        elif self._cfg.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"Unknown backend: {self._cfg.backend!r}")

    def generate(
        self,
        system_prompt: str,
        messages: Optional[List[dict]] = None,
        user_message: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response from the LLM.

        Parameters
        ----------
        system_prompt:
            The system/persona instructions prepended to every request.
        messages:
            Optional list of ``{"role": ..., "content": ...}`` dicts representing
            the conversation history (without the system message).
        user_message:
            Convenience shorthand — if supplied and *messages* is ``None``,
            a single user message is constructed.
        temperature:
            Sampling temperature for this call.  Falls back to config value.

        Returns
        -------
        str
            The model's generated reply (stripped of leading/trailing whitespace).
        """
        if self._model is None:
            self.load()

        temp = temperature if temperature is not None else self._cfg.temperature

        if messages is None:
            messages = []
        if user_message is not None:
            messages = messages + [{"role": "user", "content": user_message}]

        if self._cfg.backend == "llamacpp":
            return self._generate_llamacpp(system_prompt, messages, temp)
        return self._generate_transformers(system_prompt, messages, temp)

    def unload(self) -> None:
        """Free the model from memory."""
        self._model = None

    # ------------------------------------------------------------------
    # llama-cpp-python backend
    # ------------------------------------------------------------------

    def _load_llamacpp(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed.  "
                "Run:  pip install llama-cpp-python"
            ) from exc

        gpu_layers = self._cfg.n_gpu_layers if self._cfg.hardware_mode == "gpu" else 0
        logger.info(
            "Loading llama.cpp model: %s  (n_gpu_layers=%d)",
            self._cfg.model_path,
            gpu_layers,
        )
        self._model = Llama(
            model_path=self._cfg.model_path,
            n_ctx=self._cfg.context_length,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        logger.info("Model loaded successfully (llamacpp).")

    def _generate_llamacpp(
        self, system_prompt: str, messages: List[dict], temperature: float
    ) -> str:
        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        result = self._model.create_chat_completion(  # type: ignore[union-attr]
            messages=chat_messages,
            max_tokens=self._cfg.max_new_tokens,
            temperature=temperature,
            top_p=self._cfg.top_p,
            repeat_penalty=self._cfg.repeat_penalty,
        )
        return result["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # HuggingFace Transformers backend
    # ------------------------------------------------------------------

    def _load_transformers(self) -> None:
        try:
            import torch  # type: ignore[import-untyped]
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are not installed.  "
                "Run:  pip install transformers torch accelerate bitsandbytes"
            ) from exc

        model_id = self._cfg.hf_model_id
        use_gpu = self._cfg.hardware_mode == "gpu" and torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        logger.info(
            "Loading HuggingFace model: %s  (device=%s)", model_id, device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Use 4-bit quantisation on GPU when available to save VRAM
        if use_gpu:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)

        self._model = {"model": model, "tokenizer": tokenizer, "device": device}
        logger.info("Model loaded successfully (transformers).")

    def _generate_transformers(
        self, system_prompt: str, messages: List[dict], temperature: float
    ) -> str:
        import torch  # type: ignore[import-untyped]

        model = self._model["model"]  # type: ignore[index]
        tokenizer = self._model["tokenizer"]  # type: ignore[index]
        device = self._model["device"]  # type: ignore[index]

        # Build prompt using the tokenizer's chat template if available
        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        try:
            prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = system_prompt + "\n"
            for m in messages:
                role = m["role"].capitalize()
                prompt += f"{role}: {m['content']}\n"
            prompt += "Assistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._cfg.max_new_tokens,
                temperature=temperature,
                top_p=self._cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
