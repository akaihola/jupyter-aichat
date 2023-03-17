from getpass import getpass
from textwrap import dedent
from typing import Optional, TypedDict

import openai
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.display import display_markdown


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


class Message(TypedDict):
    role: str
    content: str


class Request(TypedDict):
    choices: list[Message]


class Response(Request):
    pass


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


class Conversation:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.requests_responses: list[Request] = []

    def say_and_listen(self, text: str) -> None:
        _authenticate()
        request_message: Message = {"role": "user", "content": text}
        self.requests_responses.append({"choices": [request_message]})
        self.messages.append(request_message)
        # https://platform.openai.com/docs/api-reference/chat/create
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
        )
        self.requests_responses.append(response.to_dict_recursive())
        response_message = response.choices[0].message
        self.messages.append(response_message.to_dict())
        output(response_message.content.strip())


@magics_class
class ConversationMagic(Magics):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conversation = Conversation()

    @line_cell_magic
    def ai(self, line: str, cell: Optional[str] = None) -> None:
        text = cell if cell is not None else line
        if not text.strip():
            output(HELP)
            return
        maybe_command, *params = text.split(None, 1)
        if maybe_command.startswith("/"):
            return self.handle_command(maybe_command, *params)
        self.conversation.say_and_listen(text)

    def handle_command(self, command: str, params: str = "") -> None:
        if command == "/restart":
            self.conversation = Conversation()
        elif command == "/get_object":
            return self.conversation
        elif command == "/history":
            output(
                "\n\n".join(
                    f"**{message['role']}:** {message['content'].strip()}"
                    for message in self.conversation.messages
                )
            )
        else:
            output(f"Unknown command `{command}`. Try `%ai` for help.")
