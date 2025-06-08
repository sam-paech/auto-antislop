# utils/whitelist.py
"""
Build and persist the global whitelist of strings that must never be
added to any ban-list.

Sources
-------
1. Every special-token string exposed by the tokenizer.
2. Every word that occurs *after* the assistant placeholder in a
   one-turn chat-template instance (user → assistant).
3. Arbitrary user-supplied extras from YAML / CLI.
"""
from __future__ import annotations
import json, re, threading
from pathlib import Path
from typing import Iterable, Set

from transformers import AutoTokenizer

_WORD_RE = re.compile(r"[A-Za-z']+")


class WhitelistBuilder:
    _tok_cache: dict[str, "AutoTokenizer"] = {}
    _lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        model_id: str,
        *,
        extra_user_items: Iterable[str] | None = None,
    ) -> Set[str]:
        """
        Parameters
        ----------
        model_id
            HuggingFace model id or local checkpoint directory.
        extra_user_items
            Free-form strings that the user always wants whitelisted.

        Returns
        -------
        set[str]  – lower-cased, deduplicated whitelist entries.
        """
        tok = cls._tokenizer(model_id)

        # 1) special tokens ----------------------------------------------
        wl: set[str] = {
            t for t in [
                tok.bos_token, tok.eos_token, tok.unk_token,
                tok.pad_token, tok.cls_token, tok.sep_token,
                *(tok.additional_special_tokens or []),
            ] if t
        }

        # 2) chat-template tail (everything after assistant placeholder) --
        tail = cls._template_tail(tok)           # one string
        wl.update(w.lower() for w in _WORD_RE.findall(tail))

        # 3) user extras --------------------------------------------------
        if extra_user_items:
            wl.update(str(x).lower() for x in extra_user_items)

        # final clean-up
        return {w.strip() for w in wl if w.strip()}

    @classmethod
    def write(cls, path: Path, whitelist: Iterable[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sorted(whitelist), indent=2, ensure_ascii=False),
            "utf-8",
        )

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def _tokenizer(cls, model_id: str):
        with cls._lock:
            tok = cls._tok_cache.get(model_id)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                cls._tok_cache[model_id] = tok
            return tok

    @staticmethod
    def _template_tail(tok) -> str:
        """Return the textual scaffold *after* the assistant placeholder."""
        PH_USER = "__USER_MSG__"
        PH_AST  = "__ASSISTANT_MSG__"

        msgs = [
            {"role": "user",      "content": PH_USER},
            {"role": "assistant", "content": PH_AST},
        ]
        full = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        i_ast = full.find(PH_AST)
        if i_ast == -1:
            raise ValueError("assistant placeholder not found in chat template")

        # slice strictly *after* the placeholder and strip whitespace
        return full[i_ast + len(PH_AST):].strip()
