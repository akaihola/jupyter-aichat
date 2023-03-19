from typing import Optional, Any

from IPython.core.magic import Magics, line_cell_magic, magics_class

from jupyter_aichat.authentication import save_api_key
from jupyter_aichat.client import Conversation
from jupyter_aichat.output import output, TemplateLoader
from jupyter_aichat.schedule import parse_schedule, Schedule


@magics_class
class ConversationMagic(Magics):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        super().__init__(*args, **kwargs)
        self.conversation = Conversation()
        self.templates = TemplateLoader()

    @line_cell_magic  # type: ignore[misc]
    def ai(self, line: str, cell: Optional[str] = None) -> Optional[Conversation]:
        text = line if cell is None else f"{line} {cell}"
        if not text.strip():  # just `%ai` or `%%ai`
            output(self.templates["help"])
            self.conversation.register_system_message(
                self.templates["help_assistant_system_message"],
                Schedule([0], self.conversation.current_step),
                skip_if_exists=True,
            )
            return None
        # handle `%ai /<command> <params>` or `%%ai /command <params>`
        maybe_command, *params = text.split(None, 1)
        if maybe_command.startswith("/"):
            return self.handle_command(maybe_command, *params)
        # not a help request or command, so it's a user message
        self.conversation.say_and_listen(text)
        return None

    def handle_command(self, command: str, params: str = "") -> Optional[Conversation]:
        if command == "/restart":
            self.conversation = Conversation()
        elif command == "/save-key":
            save_api_key()
        elif command == "/system":
            schedule, system_message = (
                parse_schedule(params, self.conversation.current_step)
            )
            self.conversation.register_system_message(system_message, schedule)
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
