import os
from typing import Iterable, Type, Union
from unittest.mock import Mock, call, patch

import pytest

from jupyter_aichat.api_types import (
    Choice,
    CompletionUsage,
    Message,
    PromptUsage,
    Request,
    Response,
)
from jupyter_aichat.client import (
    Conversation,
    ScheduledMessage,
    is_system_prompt,
    prompt_role_is,
)
from jupyter_aichat.schedule import Schedule, SchedulePattern
from tests.assertion import raises_or_matches


@pytest.fixture
def authenticate() -> Iterable[Mock]:
    with patch("jupyter_aichat.client.authenticate") as authenticate_:
        yield authenticate_


@pytest.fixture
def chat_completion() -> Iterable[Mock]:
    with patch("openai.ChatCompletion.create") as chat_completion_create:
        chat_completion_create.return_value = [
            {"choices": [{"delta": {"role": "system"}}]},
            {"choices": [{"delta": {"content": "This one is ignored"}}]},
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " World"}}]},
        ]
        yield chat_completion_create


@pytest.fixture
def chat_completion_empty() -> Iterable[Mock]:
    with patch("openai.ChatCompletion.create") as chat_completion_create:
        chat_completion_create.return_value = [
            {"choices": [{"delta": {"role": "system"}}]},
            {"choices": [{"delta": {"content": "This one is ignored"}}]},
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": ""}}]},
        ]
        yield chat_completion_create


@pytest.fixture
def update_output() -> Iterable[Mock]:
    with patch("jupyter_aichat.client.update_output") as update_output_:
        yield update_output_


def test_conversation_init_empty_transmissions() -> None:
    conversation = Conversation()

    assert not conversation.transmissions


def test_conversation_init_empty_system_message_schedules() -> None:
    conversation = Conversation()

    assert not conversation.system_message_schedules


def test_say_and_listen_authenticates(
    authenticate: Mock, chat_completion: Mock, update_output: Mock
) -> None:
    conversation = Conversation()

    conversation.say_and_listen("")

    assert authenticate.called


def test_say_and_listen_calls_api(
    authenticate: Mock, chat_completion: Mock, update_output: Mock
) -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="Shout!"))],
            usage=PromptUsage(total_tokens=50),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=70),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Here!"))],
            usage=CompletionUsage(completion_tokens=40, total_tokens=110),
        ),
    ]
    with patch("jupyter_aichat.client.openai") as openai:

        conversation.say_and_listen("Hello World")

    openai.ChatCompletion.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Shout!"},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Here!"},
            {"role": "user", "content": "Hello World"},
        ],
        stream=True,
        timeout=5,
    )


def test_say_and_listen_outputs_completion(
    authenticate: Mock, chat_completion: Mock, update_output: Mock
) -> None:
    conversation = Conversation()

    conversation.say_and_listen("")

    assert update_output.call_args_list == [
        call(None, "Hello"),
        call(None, "Hello World"),
    ]


def test_say_and_listen_doesnt_output_empty(
    authenticate: Mock, chat_completion_empty: Mock, update_output: Mock
) -> None:
    conversation = Conversation()

    conversation.say_and_listen("")

    assert not update_output.called


@pytest.mark.vcr
def test_say_and_listen_records_transmissions(update_output: Mock) -> None:
    r"""ChatGPT response is stored in `Conversation.transmissions`.

    This test uses ``pytest-recording`` to record the actual API response.

    To re-record the cassette:

    - set the ``OPENAI_API_KEY`` environment variable to a valid OpenAI API key
    - delete the file
      ``tests/cassettes/test_client/test_say_and_listen_records_transmissions.yaml``
    - run::

          pytest --record-mode=once \
            tests/test_client.py \
            -k test_say_and_listen_records_transmissions

    """
    conversation = Conversation()
    with patch("jupyter_aichat.authentication.getpass") as getpass:
        getpass.return_value = os.environ.get("OPENAI_API_KEY", "sk-dummy API key")

    conversation.say_and_listen("Hi!")

    assert conversation.transmissions == [
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=9),
        ),
        Response(
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content="\n\nHello! How can I assist you today?",
                    )
                )
            ],
            usage=CompletionUsage(completion_tokens=17, total_tokens=26),
        ),
    ]


def test_say_and_listen_doesnt_record_empty(
    authenticate: Mock, chat_completion_empty: Mock, update_output: Mock
) -> None:
    conversation = Conversation()

    conversation.say_and_listen("Hi!")

    assert conversation.transmissions == [
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=9),
        ),
    ]


@pytest.mark.parametrize(
    "max_tokens, expect",
    [
        (
            0,
            RuntimeError("The last message has 20 tokens, more than the maximum of 0."),
        ),
        (
            1,
            RuntimeError("The last message has 20 tokens, more than the maximum of 1."),
        ),
        (
            19,
            RuntimeError(
                "The last message has 20 tokens, more than the maximum of 19."
            ),
        ),
        (20, [2]),
        (21, [2]),
        (59, [2]),
        (60, [0, 2]),
        (61, [0, 2]),
        (69, [0, 2]),
        (70, [0, 1, 2]),
        (4096, [0, 1, 2]),
    ],
)
def test_get_transmissions(max_tokens: int, expect: list[int]) -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=50),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Hello World"))],
            usage=CompletionUsage(completion_tokens=20, total_tokens=70),
        ),
    ]
    with raises_or_matches(expect):
        # end of test setup

        result = conversation.get_transmissions(max_tokens)

        assert result == [conversation.transmissions[index] for index in expect]


def test_get_initial_system_messages() -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
        Request(
            choices=[Choice(message=Message(role="system", content="Shout!"))],
            usage=PromptUsage(total_tokens=50),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=60),
        ),
        Request(
            choices=[Choice(message=Message(role="system", content="Argue!"))],
            usage=PromptUsage(total_tokens=70),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="No!"))],
            usage=CompletionUsage(completion_tokens=10, total_tokens=90),
        ),
    ]

    result = conversation._get_initial_system_messages()

    assert result == (2, 50)


def test_get_initial_system_messages_none_exist() -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=60),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="No!"))],
            usage=CompletionUsage(completion_tokens=10, total_tokens=90),
        ),
    ]

    result = conversation._get_initial_system_messages()

    assert result == (0, 0)


@pytest.mark.parametrize(
    "start, stop, expect",
    [
        (0, 0, 0),
        (0, 1, 40),
        (0, 2, 50),
        (0, 3, 60),
        (0, 4, 70),
        (0, 5, 90),
        (0, 6, IndexError),
        (1, 1, 0),
        (1, 2, 10),
        (1, 3, 20),
        (1, 4, 30),
        (1, 5, 50),
        (1, 6, IndexError),
        (2, 2, 0),
        (2, 3, 10),
        (2, 4, 20),
        (2, 5, 40),
        (2, 6, IndexError),
        (3, 3, 0),
        (3, 4, 10),
        (3, 5, 30),
        (3, 6, IndexError),
        (4, 4, 0),
        (4, 5, 20),
        (4, 6, IndexError),
        (5, 5, 0),
        (5, 6, IndexError),
        (6, 6, 0),
        (42, 6, ValueError("stop (6) must be greater than start (42)")),
    ],
)
def test_get_tokens_for_slice(start: int, stop: int, expect: int) -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
        Request(
            choices=[Choice(message=Message(role="system", content="Shout!"))],
            usage=PromptUsage(total_tokens=50),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=60),
        ),
        Request(
            choices=[Choice(message=Message(role="system", content="Argue!"))],
            usage=PromptUsage(total_tokens=70),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="No way!"))],
            usage=CompletionUsage(completion_tokens=20, total_tokens=90),
        ),
    ]
    with raises_or_matches(expect):
        # end of test setup

        result = conversation.get_tokens_for_slice(start, stop)

        assert result == expect


@pytest.mark.parametrize(
    "max_tokens, expect",
    [
        (
            0,
            RuntimeError("The last message has 20 tokens, more than the maximum of 0."),
        ),
        (
            1,
            RuntimeError("The last message has 20 tokens, more than the maximum of 1."),
        ),
        (
            19,
            RuntimeError(
                "The last message has 20 tokens, more than the maximum of 19."
            ),
        ),
        (20, [("assistant", "Hello World")]),
        (21, [("assistant", "Hello World")]),
        (59, [("assistant", "Hello World")]),
        (60, [("system", "You are a dog."), ("assistant", "Hello World")]),
        (61, [("system", "You are a dog."), ("assistant", "Hello World")]),
        (69, [("system", "You are a dog."), ("assistant", "Hello World")]),
        (
            70,
            [
                ("system", "You are a dog."),
                ("user", "Hi!"),
                ("assistant", "Hello World"),
            ],
        ),
        (
            4096,
            [
                ("system", "You are a dog."),
                ("user", "Hi!"),
                ("assistant", "Hello World"),
            ],
        ),
    ],
)
def test_get_messages(max_tokens: int, expect: list[tuple[str, str]]) -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Hi!"))],
            usage=PromptUsage(total_tokens=50),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Hello World"))],
            usage=CompletionUsage(completion_tokens=20, total_tokens=70),
        ),
    ]
    with raises_or_matches(expect):
        # end of test setup

        result = conversation.get_messages(max_tokens)

        assert result == [
            Message(role=role, content=content) for role, content in expect
        ]


def test_current_step() -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="How to help?"))],
            usage=CompletionUsage(completion_tokens=30, total_tokens=70),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Please!"))],
            usage=PromptUsage(total_tokens=80),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Here you go."))],
            usage=CompletionUsage(completion_tokens=30, total_tokens=110),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Thanks."))],
            usage=PromptUsage(total_tokens=120),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Not at all."))],
            usage=CompletionUsage(completion_tokens=30, total_tokens=150),
        ),
    ]

    assert conversation.current_step == 3


def test_total_tokens_empty() -> None:
    conversation = Conversation()

    assert conversation.total_tokens == 0


def test_total_tokens() -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="How to help?"))],
            usage=CompletionUsage(completion_tokens=30, total_tokens=70),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Please!"))],
            usage=PromptUsage(total_tokens=80),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Here you go."))],
            usage=CompletionUsage(completion_tokens=30, total_tokens=110),
        ),
        Request(
            choices=[Choice(message=Message(role="user", content="Thanks."))],
            usage=PromptUsage(total_tokens=120),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="Not at all."))],
            usage=CompletionUsage(completion_tokens=30, total_tokens=150),
        ),
    ]

    assert conversation.total_tokens == 150


def test_register_system_message() -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Request(
            choices=[Choice(message=Message(role="system", content="You are a dog."))],
            usage=PromptUsage(total_tokens=40),
        ),
    ]
    conversation.register_system_message(
        "You are a dog.", Schedule(SchedulePattern([1, 2, ..., 4]))
    )
    conversation.register_system_message(
        "You are a dog.", Schedule(SchedulePattern([1, 2, ..., 4]))
    )
    conversation.register_system_message(
        "You are a dog.", Schedule(SchedulePattern([6]))
    )
    conversation.register_system_message(
        "You are a dog.", Schedule(SchedulePattern([1, 2, ..., 4])), skip_if_exists=True
    )
    conversation.register_system_message(
        "You are a dog.", Schedule(SchedulePattern([7])), skip_if_exists=True
    )
    conversation.register_system_message(
        "You rock!", Schedule(SchedulePattern([1, 2, ..., 4])), skip_if_exists=True
    )
    conversation.register_system_message(
        "You rock!", Schedule(SchedulePattern([7])), skip_if_exists=True
    )

    assert conversation.system_message_schedules == [
        ScheduledMessage(
            message=Message(role="system", content="You are a dog."),
            schedule=Schedule(pattern=[1, 2, Ellipsis, 4], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="You are a dog."),
            schedule=Schedule(pattern=[1, 2, Ellipsis, 4], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="You are a dog."),
            schedule=Schedule(pattern=[6], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="You rock!"),
            schedule=Schedule(pattern=[1, 2, Ellipsis, 4], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="You rock!"),
            schedule=Schedule(pattern=[7], start=0),
        ),
    ]


def test_get_scheduled_system_messages() -> None:
    conversation = Conversation()
    conversation.system_message_schedules = [
        ScheduledMessage(
            message=Message(role="system", content="Always"),
            schedule=Schedule(pattern=[0, 1, Ellipsis, 9], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Odd"),
            schedule=Schedule(pattern=[1, 3, Ellipsis, 9], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Even"),
            schedule=Schedule(pattern=[0, 2, Ellipsis, 8], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Five"),
            schedule=Schedule(pattern=[5], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Three, start from two"),
            schedule=Schedule(pattern=[3], start=2),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Zero, start from five"),
            schedule=Schedule(pattern=[3], start=2),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Five, start from one"),
            schedule=Schedule(pattern=[5], start=1),
        ),
        ScheduledMessage(
            message=Message(role="system", content="One, start from five"),
            schedule=Schedule(pattern=[1], start=5),
        ),
    ]

    result = conversation.get_scheduled_system_messages(5)

    assert [request.content for request in result] == [
        "Always",
        "Odd",
        "Five",
        "Three, start from two",
        "Zero, start from five",
    ]


def test_add_scheduled_system_messages() -> None:
    conversation = Conversation()
    conversation.transmissions = [
        Response(
            choices=[Choice(message=Message(role="assistant", content="1"))],
            usage=CompletionUsage(completion_tokens=1, total_tokens=1),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="2"))],
            usage=CompletionUsage(completion_tokens=1, total_tokens=2),
        ),
        Response(
            choices=[Choice(message=Message(role="assistant", content="3"))],
            usage=CompletionUsage(completion_tokens=1, total_tokens=3),
        ),
    ]
    conversation.system_message_schedules = [
        ScheduledMessage(
            message=Message(role="system", content="Always"),
            schedule=Schedule(pattern=[0, 1, Ellipsis, 9], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Odd"),
            schedule=Schedule(pattern=[1, 3, Ellipsis, 9], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Even"),
            schedule=Schedule(pattern=[0, 2, Ellipsis, 8], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Three"),
            schedule=Schedule(pattern=[3], start=0),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Two, start from one"),
            schedule=Schedule(pattern=[2], start=1),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Zero, start from three"),
            schedule=Schedule(pattern=[0], start=3),
        ),
        ScheduledMessage(
            message=Message(role="system", content="Three, start from one"),
            schedule=Schedule(pattern=[3], start=1),
        ),
        ScheduledMessage(
            message=Message(role="system", content="One, start from three"),
            schedule=Schedule(pattern=[1], start=3),
        ),
    ]

    conversation.add_scheduled_system_messages()

    assert [request.content for request in conversation.transmissions] == [
        "1",
        "2",
        "3",
        "Always",
        "Odd",
        "Three",
        "Two, start from one",
        "Zero, start from three",
    ]


@pytest.mark.kwparametrize(
    dict(prompt=Request(choices=[]), expect=IndexError),
    dict(
        prompt=Response(choices=[]),
        expect=IndexError,
    ),
    dict(
        prompt=Request(
            choices=[Choice(message=Message(role="different_role", content=""))],
        ),
        expect=False,
    ),
    dict(
        prompt=Response(
            choices=[Choice(message=Message(role="different_role", content=""))],
        ),
        expect=False,
    ),
    dict(
        prompt=Request(
            choices=[Choice(message=Message(role="expected_role", content=""))],
        ),
        expect=True,
    ),
    dict(
        prompt=Response(
            choices=[Choice(message=Message(role="expected_role", content=""))],
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
        (Request(choices=[]), IndexError),
        (Response(choices=[]), IndexError),
        (
            Request(choices=[Choice(message=Message(role="system", content=""))]),
            True,
        ),
        (
            Response(choices=[Choice(message=Message(role="system", content=""))]),
            True,
        ),
        (
            Request(choices=[Choice(message=Message(role="user", content=""))]),
            False,
        ),
        (
            Response(choices=[Choice(message=Message(role="user", content=""))]),
            False,
        ),
        (
            Request(choices=[Choice(message=Message(role="assistant", content=""))]),
            False,
        ),
        (
            Response(choices=[Choice(message=Message(role="assistant", content=""))]),
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
