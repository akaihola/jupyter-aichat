from getpass import getpass
from typing import Optional, TypedDict

import openai
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.display import Markdown, display


def _authenticate() -> None:
    if not openai.api_key:
        openai.api_key = getpass("Enter your OpenAI API key:")


class Message(TypedDict):
    role: str
    content: str


class Conversation:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def say_and_listen(self, text: str) -> None:
        _authenticate()
        self.messages.append({"role": "user", "content": text})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
        )
        response = completion.choices[0].message
        self.messages.append(response)
        display(Markdown(response.content.strip()))


@magics_class
class ConversationMagic(Magics):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conversation = Conversation()

    @line_cell_magic
    def ai(self, line: str, cell: Optional[str] = None):
        self.conversation.say_and_listen(cell or line)
