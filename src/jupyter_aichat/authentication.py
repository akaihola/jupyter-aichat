from getpass import getpass

import openai

from jupyter_aichat.output import output, TemplateLoader


def _authenticate() -> None:
    if not openai.api_key:
        output(TemplateLoader()["authentication_note"])
        openai.api_key = getpass("Enter your OpenAI API key:")
