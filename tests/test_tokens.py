import re

import pytest

from jupyter_aichat.api_types import Message
from jupyter_aichat.tokens import num_tokens_from_messages


@pytest.mark.kwparametrize(
    dict(messages=[], expect=2),
    dict(messages=[{"role": "user", "content": "foo"}], expect=8),
    dict(messages=[{"role": "assistant", "content": "bar"}], expect=8),
    dict(
        messages=[
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ],
        expect=14,
    ),
    dict(
        messages=[
            {"role": "system", "content": "foo"},
            {"role": "user", "content": "bar"},
        ],
        expect=14,
    ),
    dict(messages=[{"name": "Samuel"}], expect=7,),
    dict(
        messages=[
            {"role": "system", "content": "foo"},
            {"role": "user", "content": "bar"},
            {
                "role": "assistant",
                "content": "I think you meant to say baz. Didn't you really mean baz?",
            },
        ],
        expect=34,
    ),
)
def test_num_tokens_from_messages(messages: list[str], expect: int) -> None:
    messages_ = [Message(**m) for m in messages]  # type: ignore[misc]
    result = num_tokens_from_messages(messages_)

    assert result == expect


def test_num_tokens_from_messages_unknown_model() -> None:
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "num_tokens_from_messages() is not presently implemented for model foo.\n"
            "See https://github.com/openai/openai-python/blob/main/chatml.md for"
            " information on how messages are converted to tokens."
        ),
    ):
        # end of test setup

        num_tokens_from_messages([], "foo")
