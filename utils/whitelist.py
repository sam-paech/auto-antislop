# utils/whitelist.py
"""
Centralised construction / persistence of the project-wide whitelist.
"""
from __future__ import annotations
import re, threading, json
from pathlib import Path
from typing import Iterable, Set, List
from transformers import AutoTokenizer
from utils.chat_template_helper import ChatTemplateFormatter   # already in repository


_WORD_RE = re.compile(r"[A-Za-z']+")


class WhitelistBuilder:
    """
    Collects strings that must *never* be banned.  
    Sources
    -------
    1. Tokeniser special-token texts (bos/eos/etc. + `additional_special_tokens`)
    2. Every word occurring in the model’s chat-template scaffolding
       (including an injected system prompt, if provided)
    3. Arbitrary user-supplied strings (from YAML or CLI)
    """

    _tok_cache: dict[str, "AutoTokenizer"] = {}
    _lock = threading.Lock()

    # --------------------------------------------------------------- #
    # public helpers                                                  #
    # --------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        model_id: str,
        *,
        system_prompt: str = "",
        extra_user_items: Iterable[str] | None = None,
    ) -> Set[str]:
        tok = cls._tokenizer(model_id)

        # (1) special tokens -----------------------------------------
        wl: set[str] = {
            t for t in [
                tok.bos_token, tok.eos_token, tok.pad_token, tok.sep_token,
                tok.cls_token, tok.unk_token, * (tok.additional_special_tokens or [])
            ] if t
        }

        # (2) chat-template boilerplate ------------------------------
        fmt = ChatTemplateFormatter(model_id, system_prompt)
        scaffold = fmt.build_prompt("PLACEHOLDER_USER", "PLACEHOLDER_ASSISTANT")
        wl.update(w.lower() for w in _WORD_RE.findall(scaffold))

        # (3) user extras --------------------------------------------
        if extra_user_items:
            wl.update(str(x).lower() for x in extra_user_items)

        # final clean-up – only real words / tokens
        return {w.strip() for w in wl if w.strip()}

    @classmethod
    def write(cls, path: Path, whitelist: Iterable[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(whitelist), indent=2, ensure_ascii=False), "utf-8")

    # --------------------------------------------------------------- #
    # internals                                                       #
    # --------------------------------------------------------------- #
    @classmethod
    def _tokenizer(cls, model_id: str):
        with cls._lock:
            tok = cls._tok_cache.get(model_id)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                cls._tok_cache[model_id] = tok
            return tok
