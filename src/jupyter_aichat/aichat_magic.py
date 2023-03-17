from getpass import getpass
from itertools import takewhile
from textwrap import dedent
from typing import Optional, TypedDict, Union, Any

import openai
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.display import display_markdown
from jupyter_aichat.api_types import Message
from jupyter_aichat.tokens import num_tokens_from_messages


def output(markdown_text: str) -> None:
    display_markdown(markdown_text, raw=True)


HELP_ACCOUNT_AND_KEY = (
    "You need an [OpenAI account](https://platform.openai.com/) and an "
    "[API key](https://platform.openai.com/account/api-keys). Consider revoking the "
    "key after using it on a public server. "
)

HELP_PRICING = "Also be aware of the [pricing](https://openai.com/pricing)."


def _authenticate() -> None:
    if not openai.api_key:
        output(
            f"{HELP_ACCOUNT_AND_KEY} The key will be prompted when starting the chat. "
            f"{HELP_PRICING}"
        )
        openai.api_key = getpass("Enter your OpenAI API key:")


class PromptUsage(TypedDict):
    total_tokens: int


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(TypedDict):
    message: Message


class Transmission(TypedDict):
    choices: list[Choice]


class Request(Transmission):
    usage: PromptUsage


class ApiCompletionUsage:
    prompt_tokens: int
    completion_tokens: int


class Response(Transmission):
    usage: CompletionUsage


HELP = dedent(
    rf"""
    # 🗨 · · · **jupyter-aichat** · · · 🗨
    ## Talking to the chatbot
    Click on a cell and type either
    >```python
    >[1]: %ai "your message"
    >```

    or

    >```python
    >[2]: %%ai
    >      your message
    >      which may contain multiple lines
    >```

    and type `Shift-Enter`.

    ## Prerequisites
    {HELP_ACCOUNT_AND_KEY} {HELP_PRICING}

    ## Additional commands
    - `%ai` – show these usage instructions
    - `%ai /restart` – forget the conversation and start a new one
    - `%ai /history` – display the complete conversation so far
    - `conv = %ai /get_object` – assign the `Conversation` object to a variable

    In the `Conversation` class you can access the conversation history:
    >```python
    >[3]: conv.messages
    >
    >      [{{'role': 'user', 'content': 'Hello!'}},
    >       {{'role': 'assistant',
    >        'content': '\n\nHello there! How may I assist you today?'}}]
    >```

    You can also access the complete responses in `conv.requests_responses`.

    """
)


def is_system_message(message: Message) -> bool:
    return message["role"] == "system"


def is_system_prompt(prompt: Union[Request, Response]) -> bool:
    return is_system_message(prompt["choices"][0]["message"])


class Conversation:
    MAX_TOKENS = 4096
    MODEL = "gpt-3.5-turbo"

    def __init__(self) -> None:
        self.transmissions: list[Union[Request, Response]] = []

    def say_and_listen(self, text: str) -> None:
        _authenticate()
        request_message: Message = {"role": "user", "content": text}
        prompt_tokens = num_tokens_from_messages([request_message])
        prompt: Request = {
            "choices": [{"message": request_message}],
            "usage": {
                "total_tokens": self.total_tokens + prompt_tokens,
            },
        }
        self.transmissions.append(prompt)
        # https://platform.openai.com/docs/api-reference/chat/create
        response = openai.ChatCompletion.create(
            model=self.MODEL,
            messages=self.get_messages(max_tokens=self.MAX_TOKENS),
        )
        prompt["usage"]["total_tokens"] = response.usage.prompt_tokens
        self.transmissions.append(response.to_dict_recursive())
        response_message = response["choices"][0]["message"]
        output(response_message["content"].strip())

    def get_transmissions(self, max_tokens: int) -> list[Union[Request, Response]]:
        """Return the transmissions that fit within the token budget.

        :param max_tokens: The maximum number of tokens to use.
        :return: The transmissions that fit within the token budget.

        """
        if self.total_tokens <= max_tokens:
            return self.transmissions
        num_transmissions = len(self.transmissions)
        num_system_messages, system_message_tokens = self._get_initial_system_messages()
        for index in range(num_system_messages, num_transmissions):
            tail_tokens = self.get_tokens_for_slice(index, num_transmissions)
            if system_message_tokens + tail_tokens <= max_tokens:
                break
        else:
            index = num_transmissions - 1
            last_message_tokens = self.get_tokens_for_slice(index, num_transmissions)
            if last_message_tokens > max_tokens:
                raise RuntimeError(
                    f"The last message has {last_message_tokens} tokens,"
                    f" more than the maximum of {max_tokens}."
                )
            num_system_messages = 0
        return self.transmissions[:num_system_messages] + self.transmissions[index:]

    def _get_initial_system_messages(self) -> tuple[int, int]:
        """Return the number of initial system messages and their total tokens.

        :return: The number of initial system messages and their total tokens.

        """
        system_prompts = list(takewhile(is_system_prompt, self.transmissions))
        if not system_prompts:
            return 0, 0
        last_system_prompt = system_prompts[-1]
        return len(system_prompts), last_system_prompt["usage"]["total_tokens"]

    def get_tokens_for_slice(self, start: int, stop: int) -> int:
        """Return the total number of tokens in the slice.

        :param start: The start index of the slice.
        :param stop: The stop index of the slice.
        :return: The total number of tokens in the slice.

        """
        if not self.transmissions:
            return 0
        if start == 0:
            return self.transmissions[stop - 1]["usage"]["total_tokens"]
        return (
            self.transmissions[stop - 1]["usage"]["total_tokens"]
            - self.transmissions[start - 1]["usage"]["total_tokens"]
        )

    def get_messages(self, max_tokens: int = 2**63 - 1) -> list[Message]:
        """Return the messages that fit within the token budget.

        :param max_tokens: The maximum number of tokens to use.
        :return: The messages that fit within the token budget.

        """
        return [
            transmission["choices"][0]["message"]
            for transmission in self.get_transmissions(max_tokens)
        ]

    @property
    def total_tokens(self) -> int:
        if not self.transmissions:
            return 0
        return self.transmissions[-1]["usage"]["total_tokens"]


@magics_class
class ConversationMagic(Magics):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__(*args, **kwargs)
        self.conversation = Conversation()

    @line_cell_magic  # type: ignore[misc]
    def ai(self, line: str, cell: Optional[str] = None) -> Optional[Conversation]:
        text = cell if cell is not None else line
        if not text.strip():
            output(HELP)
            return None
        maybe_command, *params = text.split(None, 1)
        if maybe_command.startswith("/"):
            return self.handle_command(maybe_command, *params)
        self.conversation.say_and_listen(text)
        return None

    def handle_command(self, command: str, params: str = "") -> Optional[Conversation]:
        if command == "/restart":
            self.conversation = Conversation()
        elif command == "/get_object":
            return self.conversation
        elif command == "/history":
            output(
                "\n\n".join(
                    f"**{message['role']}:** {message['content'].strip()}"
                    for message in self.conversation.get_messages()
                )
            )
        else:
            output(f"Unknown command `{command}`. Try `%ai` for help.")
        return None
