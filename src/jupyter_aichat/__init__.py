"""Jupyter magic command for having AI chat conversations in a notebook"""

__version__ = "0.0.1"

from IPython.core.interactiveshell import InteractiveShell

from .aichat_magic import ConversationMagic
from .output import output, TemplateLoader


def load_ipython_extension(ipython: InteractiveShell) -> None:
    ipython.register_magics(ConversationMagic)  # type: ignore[no-untyped-call]
    output(TemplateLoader()["load_ipython_extension"])
