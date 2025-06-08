# utils/whitelist.py
"""
Centralised logic for constructing and persisting the project-wide
whitelist of strings that must NEVER be banned.

Sources
-------
1. Text of every special token exposed by the tokenizer.
2. All words that appear in the model’s chat-template scaffolding
   (including an injected system prompt, if supplied).
3. Arbitrary user-defined entries from YAML / CLI.
"""
from __future__ import annotations
import json, re, threading
from pathlib import Path
from typing import Iterable, Set, List

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
        system_prompt: str = "",
        extra_user_items: Iterable[str] | None = None,
    ) -> Set[str]:
        """
        Parameters
        ----------
        model_id
            HF model id or local path whose tokenizer defines the chat template.
        system_prompt
            Optional system prompt text injected once as a “system” role.
        extra_user_items
            Free-form strings provided by the user that should also be whitelisted.

        Returns
        -------
        Set[str]
            Lower-cased, deduplicated whitelist.
        """
        tok = cls._get_tokenizer(model_id)

        # 1) special tokens ----------------------------------------------
        wl: set[str] = {
            t for t in [
                tok.bos_token, tok.eos_token, tok.unk_token,
                tok.pad_token, tok.cls_token, tok.sep_token,
                * (tok.additional_special_tokens or []),
            ] if t
        }

        # 2) full chat-template scaffold ---------------------------------
        scaffold_text = cls._generate_template_scaffold(tok, system_prompt)
        wl.update(w.lower() for w in _WORD_RE.findall(scaffold_text))

        # 3) arbitrary user extras ---------------------------------------
        if extra_user_items:
            wl.update(str(x).lower() for x in extra_user_items)

        # final clean-up
        return {w.strip() for w in wl if w.strip()}

    @classmethod
    def write(cls, path: Path, whitelist: Iterable[str]) -> None:
        """Persist whitelist to *path* as a JSON list (one string per entry)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(whitelist), indent=2, ensure_ascii=False), "utf-8")

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @classmethod
    def _get_tokenizer(cls, model_id: str):
        with cls._lock:
            tok = cls._tok_cache.get(model_id)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                cls._tok_cache[model_id] = tok
            return tok

    @staticmethod
    def _generate_template_scaffold(tok, system_prompt: str) -> str:
        """
        Build one *complete* chat-template instance containing:
           [ optional system ]  +  user  +  assistant
        with placeholder texts so the template is fully realised.
        The placeholders are removed before word extraction.
        """
        PH_SYS  = "__SYS_PROMPT__"
        PH_USER = "__USER_MSG__"
        PH_AST  = "__ASSISTANT_MSG__"

        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": PH_SYS})
        messages.extend([
            {"role": "user", "content": PH_USER},
            {"role": "assistant", "content": PH_AST},
        ])

        # full template text (no truncation, no generation prompt)
        tmpl: str = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # drop placeholders but keep everything else
        tmpl = (
            tmpl.replace(PH_SYS, system_prompt)
                .replace(PH_USER, "")
                .replace(PH_AST, "")
        )
        return tmpl
