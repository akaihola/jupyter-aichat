from typing import Union
from unittest.mock import DEFAULT, Mock, patch

import pytest

from jupyter_aichat import authentication
from jupyter_aichat.aichat_magic import ConversationMagic
from jupyter_aichat.api_types import (
    Message,
    Request,
    Response,
    Usage,
)
from jupyter_aichat.client import ScheduledMessage
from jupyter_aichat.output import TemplateLoader, output
from jupyter_aichat.schedule import Schedule, SchedulePattern
from tests.assertion import raises_or_matches


@pytest.fixture
def magic() -> ConversationMagic:
    return ConversationMagic()


def test_init_adds_empty_conversation(magic: ConversationMagic) -> None:
    assert not magic.conversation.transmissions


def test_init_adds_conversation_with_no_system_message_schedules(
    magic: ConversationMagic,
) -> None:
    assert not magic.conversation.system_message_schedules


def test_init_adds_template_loader(magic: ConversationMagic) -> None:
    assert magic.templates


@pytest.fixture(
    params=[("", None), ("   ", None), ("", ""), ("", " \n "), ("   ", " \n ")]
)
def ai_magic_with_empty_input(
    magic: ConversationMagic, request: pytest.FixtureRequest
) -> None:
    line, cell = request.param
    magic.ai(line, cell)


def test_ai_outputs_help(
    magic: ConversationMagic,
    capsys: pytest.CaptureFixture[str],
    ai_magic_with_empty_input: None,
) -> None:
    assert capsys.readouterr().out[:21] == "{'text/markdown': '# "


def test_ai_registers_system_message(
    magic: ConversationMagic, ai_magic_with_empty_input: None
) -> None:
    expect_content = TemplateLoader()["help_assistant_system_message"]
    assert expect_content[:51] == "The user is chatting with you in a Jupyter notebook"
    expect_msg = Message(role="system", content=expect_content)
    expect_schedule = Schedule(pattern=[0], start=0)
    expect_schedules = [ScheduledMessage(message=expect_msg, schedule=expect_schedule)]
    assert magic.conversation.system_message_schedules == expect_schedules


@pytest.mark.kwparametrize(
    dict(line="hello", expect="hello"),
    dict(line="  hello", expect="  hello"),
    dict(line="hello  ", expect="hello  "),
    dict(cell="hello\n", expect="hello\n"),
    dict(cell="  hello\n", expect="  hello\n"),
    dict(cell="hello  \n", expect="hello  \n"),
    dict(line="hello", cell="world\n", expect="hello world\n"),
    dict(line="  hello", cell="\n\nworld\n", expect="  hello \n\nworld\n"),
    line="",
    cell=None,
)
def test_ai_with_prompt_calls_say_and_listen(
    magic: ConversationMagic, line: str, cell: str, expect: str
) -> None:
    with patch.object(magic.conversation, "say_and_listen") as say_and_listen:
        # end of test setup

        magic.ai(line, cell)

    say_and_listen.assert_called_once_with(expect)


def test_ai_with_prompt_outputs_only_response(
    magic: ConversationMagic, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch.object(magic.conversation, "say_and_listen") as say_and_listen:
        say_and_listen.side_effect = lambda text: output("response")

        magic.ai("hello", None)

    assert capsys.readouterr().out == "{'text/markdown': 'response'}\n"


def test_ai_with_prompt_registers_no_system_message(magic: ConversationMagic) -> None:
    with patch.object(magic.conversation, "say_and_listen"):

        magic.ai("hello", None)

    assert not magic.conversation.system_message_schedules


def test_ai_with_prompt_calls_no_command_handler(magic: ConversationMagic) -> None:
    with patch.object(magic.conversation, "say_and_listen"), patch.object(
        magic, "handle_command"
    ) as handle_command:

        magic.ai("hello", None)

    assert not handle_command.called


def test_ai_doesnt_duplicate_system_message(magic: ConversationMagic) -> None:
    help_text = TemplateLoader()["help_assistant_system_message"]
    help_msg = Message(role="system", content=help_text)
    initial_request = Request(message=help_msg, usage=Usage(total_tokens=40))
    magic.conversation.transmissions = [initial_request]

    magic.ai("", None)

    assert magic.conversation.transmissions == [initial_request]


@pytest.mark.kwparametrize(
    dict(line="/restart", expect=("/restart",)),
    dict(line="  /save-key", expect=("/save-key",)),
    dict(line="/get_object  ", expect=("/get_object",)),
    dict(line='/system  "You are a dog"  ', expect=("/system", '"You are a dog"  ')),
    dict(line="/history", cell="\n \n limit\n", expect=("/history", "limit\n")),
    dict(line="", cell="\n /template-name \n ", expect=("/template-name",)),
    dict(line="", cell="\n /system \n 1 \n 2 \n ", expect=("/system", "1 \n 2 \n ")),
    cell=None,
)
def test_ai_handles_slash_commands(
    magic: ConversationMagic, line: str, cell: str, expect: list[str]
) -> None:
    with patch.multiple(magic, conversation=DEFAULT, handle_command=DEFAULT):
        # end of test setup

        magic.ai(line, cell)

        magic.handle_command.assert_called_once_with(  # type: ignore[attr-defined]
            *expect
        )


def test_ai_slash_command_doesnt_call_say_and_listen(magic: ConversationMagic) -> None:
    with patch.multiple(magic, conversation=DEFAULT):
        # end of test setup

        magic.ai("/get_object", None)

        assert (
            not magic.conversation.say_and_listen.called  # type: ignore[attr-defined]
        )


def test_handle_command_restart(magic: ConversationMagic) -> None:
    old_conversation = magic.conversation

    magic.handle_command("/restart")

    assert magic.conversation is not old_conversation


def test_handle_command_save_key(magic: ConversationMagic) -> None:
    keyring = Mock()
    openai = Mock(api_key="sk-123")
    with patch.multiple(authentication, keyring=keyring, openai=openai):
        # end of test setup

        magic.handle_command("/save-key")

    keyring.set_password.assert_called_once_with(
        "api.openai.com", "jupyter-aichat", "sk-123"
    )


@pytest.mark.kwparametrize(
    dict(params='"You are Plato."', expect_content='"You are Plato."'),
    dict(params=" \n You are Plato.\n", expect_content=" \n You are Plato.\n"),
    dict(
        params="schedule=[1, 2, ..., 5] \n You are Plato.\n",
        expect_content=" \n You are Plato.\n",
        expect_pattern=[1, 2, Ellipsis, 5],
    ),
    dict(params="Ho!", completions=["1", "2"], expect_content="Ho!", expect_start=2),
    completions=[],
    expect_pattern=[0],
    expect_start=0,
)
def test_handle_command_system(
    magic: ConversationMagic,
    params: str,
    expect_content: str,
    completions: list[str],
    expect_pattern: SchedulePattern,
    expect_start: int,
) -> None:
    messages = [Message(role="assistant", content=content) for content in completions]
    magic.conversation.transmissions = [
        Response(message=message) for message in messages
    ]

    magic.handle_command("/system", params)

    assert magic.conversation.system_message_schedules == [
        ScheduledMessage(
            message=Message(role="system", content=expect_content),
            schedule=Schedule(pattern=expect_pattern, start=expect_start),
        )
    ]


def test_handle_command_get_object(magic: ConversationMagic) -> None:
    result = magic.handle_command("/get_object")

    assert result == magic.conversation


@pytest.mark.kwparametrize(
    dict(params="", expect=r"**system:** 1\n\n**user:** 2\n\n**assistant:** 3"),
    dict(params="4097", expect=r"**system:** 1\n\n**user:** 2\n\n**assistant:** 3"),
    dict(params="limit", expect=r"**system:** 1\n\n**assistant:** 3"),
    dict(params="4096", expect=r"**system:** 1\n\n**assistant:** 3"),
    dict(params="97", expect=r"**system:** 1\n\n**assistant:** 3"),
    dict(params="96", expect=r"**assistant:** 3"),
    dict(params="49", expect=r"**assistant:** 3"),
    dict(
        params="48",
        expect=RuntimeError(
            "The last message has 49 tokens, more than the maximum of 48."
        ),
    ),
)
def test_handle_command_history(
    magic: ConversationMagic,
    capsys: pytest.CaptureFixture[str],
    params: str,
    expect: Union[list[str], RuntimeError],
) -> None:
    magic.conversation.transmissions = [
        Request(
            message=Message(role="system", content="1"),
            usage=Usage(total_tokens=48),
        ),
        Request(
            message=Message(role="user", content="2"),
            usage=Usage(total_tokens=4048),
        ),
        Response(
            message=Message(role="assistant", content="3"),
            usage=Usage(total_tokens=4097),
        ),
    ]
    with raises_or_matches(expect):
        # end of test setup

        magic.handle_command("/history", params)

        assert capsys.readouterr().out == f"{{'text/markdown': '{expect}'}}\n"


def test_handle_command_template(
    magic: ConversationMagic, capsys: pytest.CaptureFixture[str]
) -> None:
    magic.handle_command("/pricing")

    assert capsys.readouterr().out == (
        "{'text/markdown': '"
        rf"Also be aware of the [pricing](https://openai.com/pricing).\n"
        "'}\n"
    )


def test_handle_command_unknown(
    magic: ConversationMagic, capsys: pytest.CaptureFixture[str]
) -> None:
    magic.handle_command("/unknown-command")

    assert capsys.readouterr().out == (
        "{'text/markdown': '"
        "Unknown command `/unknown-command`. Try `%ai` for help."
        "'}\n"
    )
