from string import Formatter
from typing import cast

import pkg_resources
from IPython.core.display import display_markdown


def output(markdown_text: str) -> None:
    """Display the given Markdown text.

    :param markdown_text: The Markdown text to display.

    """
    display_markdown(markdown_text, raw=True)  # type: ignore[no-untyped-call]


class TemplateLoader:
    def __contains__(self, name: str) -> bool:
        """Return whether a text template with the given name exists.

        :param name: The base file name of the text template.
        :return: Whether a data template with the given name exists.

        """
        return pkg_resources.resource_exists(__name__, f"data/{name}.md")

    def __getitem__(self, name: str) -> str:
        """Load and return rendered contents of the text template with the given name.

        :param name: The name of the text template.
        :return: The rendered contents of the text template with the given name.

        """
        template = pkg_resources.resource_string(__name__, f"data/{name}.md").decode()
        return cast(
            str, Formatter().vformat(template, (), self)  # type: ignore[call-overload]
        )
