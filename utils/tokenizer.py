# Source for "Build a Large Language Model From Scratch"
import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Llama3Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set()):
        ids: list[int] = []

        if bos:
            ids.append(self.special["<|begin_of_text|>"])

        # delegate to underlying tiktoken.Encoding.encode
        ids.extend(
            self.model.encode(
                text,
                allowed_special=allowed_special,
            )
        )
        if eos:
            ids.append(self.special["<|end_of_text|>"])

        return ids

    def decode(self, ids):
        return self.model.decode(ids)


class ChatFormat:

    def __init__(self, tokenizer: Llama3Tokenizer, *,
                 default_system="You are a helpful assistant."):
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None, allowed_special=set()):
        sys_msg = system_message if system_message is not None else self.default_system

        ids = [self.tok.special["<|begin_of_text|>"]]

        # system
        ids += self._header("system")
        ids += self.tok.encode(sys_msg, allowed_special=allowed_special)
        ids += [self.tok.special["<|eot_id|>"]]

        # user
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # assistant header (no content yet)
        ids += self._header("assistant")

        return ids

    def decode(self, ids):
        return self.tok.decode(ids)


def clean_text(text):
    # Trim everything before the assistant header
    start_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    end_marker = "<|eot_id|>"

    start = text.find(start_marker)
    if start != -1:
        text = text[start + len(start_marker):]

    # Trim anything after the assistant's end-of-turn token
    end = text.find(end_marker)
    if end != -1:
        text = text[:end]

    return text.strip()

