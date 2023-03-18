from getpass import getpass
from itertools import takewhile
from string import Formatter
from textwrap import dedent
from typing import Optional, TypedDict, Union, Any

import openai
import pkg_resources
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.display import display_markdown
from jupyter_aichat.api_types import Message
from jupyter_aichat.tokens import num_tokens_from_messages


def output(markdown_text: str) -> None:
    display_markdown(markdown_text, raw=True)


def _authenticate() -> None:
    if not openai.api_key:
        output(TemplateLoader()["authentication_note"])
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

    def add_system_message(self, content: str, skip_if_exists: bool = False) -> None:
        """Add a system message to the conversation.

        :param content: The content of the system message.
        :param skip_if_exists: Whether to skip adding the message if it already exists.

        """
        message: Message = {"role": "system", "content": content}
        total_tokens = self.total_tokens + num_tokens_from_messages([message])
        usage: PromptUsage = {"total_tokens": total_tokens}
        request: Request = {"choices": [{"message": message}], "usage": usage}
        if skip_if_exists and any(
            is_system_prompt(prompt)
            and prompt["choices"][0]["message"]["content"] == content
            for prompt in self.transmissions
        ):
            return
        self.transmissions.append(request)


class TemplateLoader:
    def __contains__(self, name: str) -> bool:
        return pkg_resources.resource_exists(__name__, f"data/{name}.md")

    def __getitem__(self, name: str) -> str:
        """Load and return rendered contents of the data template with the given name.

        :param name: The name of the data template.
        :return: The rendered contents of the data template with the given name.

        """
        template = pkg_resources.resource_string(__name__, f"data/{name}.md").decode()
        return Formatter().vformat(template, (), self)


@magics_class
class ConversationMagic(Magics):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__(*args, **kwargs)
        self.conversation = Conversation()
        self.templates = TemplateLoader()

    @line_cell_magic  # type: ignore[misc]
    def ai(self, line: str, cell: Optional[str] = None) -> Optional[Conversation]:
        text = line if cell is None else f"{line} {cell}"
        if not text.strip():
            output(self.templates["help"])
            self.conversation.add_system_message(
                self.templates["help_assistant_system_message"],
                skip_if_exists=True,
            )
            return None
        maybe_command, *params = text.split(None, 1)
        if maybe_command.startswith("/"):
            return self.handle_command(maybe_command, *params)
        self.conversation.say_and_listen(text)
        return None

    def handle_command(self, command: str, params: str = "") -> Optional[Conversation]:
        if command == "/restart":
            self.conversation = Conversation()
        elif command == "/system":
            self.conversation.add_system_message(params)
        elif command == "/get_object":
            return self.conversation
        elif command == "/history":
            args = (
                [self.conversation.MAX_TOKENS]
                if params == "limit"
                else [int(params)]
                if params
                else []
            )
            output(
                "\n\n".join(
                    f"**{message['role']}:** {message['content'].strip()}"
                    for message in self.conversation.get_messages(*args)
                )
            )
        elif command.startswith("/") and command[1:] in self.templates:
            output(self.templates[command[1:]])
        else:
            output(f"Unknown command `{command}`. Try `%ai` for help.")
        return None
