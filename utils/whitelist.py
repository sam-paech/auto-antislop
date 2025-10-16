# utils/whitelist.py
"""
Constructs a global whitelist of strings that must never be placed in
any ban list.

Sources
-------
1. All special-token texts exposed by the model’s tokenizer.
2. Every phrase (entire line) – and every word inside those phrases –
   that appears *after* the assistant-message placeholder in a single
   user→assistant chat-template example.
3. Optional user-supplied strings from the YAML / CLI configuration.
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Iterable, Set

from transformers import AutoTokenizer
from slop_forensics.utils import normalize_text as normalise_keep_marks
from slop_forensics.utils import extract_words

# ------------------------------------------------------------------------------
# Helper class
# ------------------------------------------------------------------------------

class WhitelistBuilder:
    """
    Static helper for creating and persisting the whitelist.

    All strings are:

    * converted to lowercase
    * normalised via `normalise_keep_marks`
    * deduplicated
    """

    _tokenizer_cache: dict[str, "AutoTokenizer"] = {}
    _cache_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
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
            Hugging Face model ID or local checkpoint directory.
        extra_user_items
            Arbitrary strings provided by the user that must also be whitelisted.

        Returns
        -------
        Set[str]
            Normalised whitelist entries (lower-cased, no duplicates, no blanks).
        """
        tokenizer = cls._get_tokenizer(model_id)
        whitelist: set[str] = set()

        # 1. Special token texts -----------------------------------------
        special_token_texts = [
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.unk_token,
            tokenizer.pad_token,
            tokenizer.cls_token,
            tokenizer.sep_token,
            *(tokenizer.additional_special_tokens or []),
        ]
        for raw_text in special_token_texts:
            if not raw_text:
                continue
            cls._add_phrase_and_words(whitelist, raw_text)

        # 2. Tail of the chat template -----------------------------------
        template_tail_text = cls._get_chat_template_tail(tokenizer)
        for line in template_tail_text.splitlines():
            cls._add_phrase_and_words(whitelist, line)

        # 3. User-supplied extras ----------------------------------------
        if extra_user_items:
            for item in extra_user_items:
                cls._add_phrase_and_words(whitelist, str(item))

        # Final clean-up: remove any empty strings that might have slipped in
        whitelist.discard("")
        return whitelist

    @classmethod
    def write(cls, file_path: Path, whitelist: Iterable[str]) -> None:
        """Write the whitelist to *file_path* as pretty-printed JSON."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(sorted(whitelist), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _get_tokenizer(cls, model_id: str):
        """Thread-safe one-time load of the tokenizer."""
        with cls._cache_lock:
            tokenizer = cls._tokenizer_cache.get(model_id)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )
                cls._tokenizer_cache[model_id] = tokenizer
            return tokenizer

    @staticmethod
    def _add_phrase_and_words(target_set: set[str], raw_text: str) -> None:
        """
        Normalise *raw_text*, add the whole phrase, then add each individual
        word extracted from the phrase.
        """
        normalised = normalise_keep_marks(raw_text)
        if not normalised:
            return
        target_set.add(normalised)
        target_set.update(extract_words(normalised))

    @staticmethod
    def _get_chat_template_tail(tokenizer) -> str:
        """
        Build one user→assistant chat-template instance and return only the
        text *after* the assistant placeholder.  That is the scaffold the
        model tends to emit, so its words must be whitelisted.
        """
        placeholder_user = "__USER__"
        placeholder_assistant = "__ASSISTANT__"

        messages = [
            {"role": "user",      "content": placeholder_user},
            {"role": "assistant", "content": placeholder_assistant},
        ]

        full_template: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        assistant_pos = full_template.find(placeholder_assistant)
        if assistant_pos == -1:
            # Fallback: return the whole template if the placeholder wasn't found
            return full_template.strip()

        tail = full_template[assistant_pos + len(placeholder_assistant):].strip()
        return tail
