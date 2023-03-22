from dataclasses import fields
from typing import Iterable, cast

import tiktoken

from jupyter_aichat.api_types import Message


def num_tokens_from_messages(
    messages: Iterable[Message], model: str = "gpt-3.5-turbo-0301"
) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for field in fields(message):
                value = getattr(message, field.name)
                if value is None:
                    continue
                num_tokens += len(encoding.encode(cast(str, value)))
                if field.name == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            "num_tokens_from_messages() is not presently implemented for model"
            f" {model}.\n"
            "See https://github.com/openai/openai-python/blob/main/chatml.md for"
            " information on how messages are converted to tokens."
        )
