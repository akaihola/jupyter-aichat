from typing import Union, Type

import pytest

from jupyter_aichat.api_types import (
    Response,
    Request,
    Choice,
    Message,
    PromptUsage,
    CompletionUsage,
)
from jupyter_aichat.client import is_system_prompt, prompt_role_is
from tests.assertion import raises_or_matches


@pytest.mark.kwparametrize(
    dict(
        prompt=Request(choices=[], usage=PromptUsage(total_tokens=0)), expect=IndexError
    ),
    dict(
        prompt=Response(
            choices=[],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        ),
        expect=IndexError,
    ),
    dict(
        prompt=Request(
            choices=[Choice(message=Message(role="different_role", content=""))],
            usage=PromptUsage(total_tokens=0),
        ),
        expect=False,
    ),
    dict(
        prompt=Response(
            choices=[Choice(message=Message(role="different_role", content=""))],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        ),
        expect=False,
    ),
    dict(
        prompt=Request(
            choices=[Choice(message=Message(role="expected_role", content=""))],
            usage=PromptUsage(total_tokens=0),
        ),
        expect=True,
    ),
    dict(
        prompt=Response(
            choices=[Choice(message=Message(role="expected_role", content=""))],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        ),
        expect=True,
    ),
)
def test_prompt_role_is(
    prompt: Union[Request, Response],
    expect: Union[bool, Type[BaseException]],
) -> None:
    with raises_or_matches(expect):
        # end of test setup

        result = prompt_role_is(prompt, "expected_role")

        assert result == expect


@pytest.mark.parametrize(
    "prompt, expect",
    [
        (Request(choices=[], usage=PromptUsage(total_tokens=0)), IndexError),
        (
            Response(
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ),
            IndexError,
        ),
        (
            Request(
                choices=[Choice(message=Message(role="system", content=""))],
                usage=PromptUsage(total_tokens=0),
            ),
            True,
        ),
        (
            Response(
                choices=[Choice(message=Message(role="system", content=""))],
                usage=CompletionUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ),
            True,
        ),
        (
            Request(
                choices=[Choice(message=Message(role="user", content=""))],
                usage=PromptUsage(total_tokens=0),
            ),
            False,
        ),
        (
            Response(
                choices=[Choice(message=Message(role="user", content=""))],
                usage=CompletionUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ),
            False,
        ),
        (
            Request(
                choices=[Choice(message=Message(role="assistant", content=""))],
                usage=PromptUsage(total_tokens=0),
            ),
            False,
        ),
        (
            Response(
                choices=[Choice(message=Message(role="assistant", content=""))],
                usage=CompletionUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            ),
            False,
        ),
    ],
)
def test_is_system_prompt(
    prompt: Union[Request, Response], expect: Union[bool, Type[BaseException]]
) -> None:
    with raises_or_matches(expect):
        # end of test setup

        result = is_system_prompt(prompt)

        assert result == expect
