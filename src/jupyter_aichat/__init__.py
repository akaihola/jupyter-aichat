"""Jupyter magic command for having AI chat conversations in a notebook"""

__version__ = "0.0.1"

from .aichat_magic import ConversationMagic


def load_ipython_extension(ipython):
    ipython.register_magics(ConversationMagic)
