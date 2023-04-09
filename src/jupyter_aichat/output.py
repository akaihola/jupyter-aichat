from string import Formatter
from typing import cast

import pkg_resources
from IPython.core.display import Markdown, display_markdown
from IPython.display import DisplayHandle, display


def output(markdown_text: str) -> None:
    """Display the given Markdown text.

    :param markdown_text: The Markdown text to display.

    """
    display_markdown(markdown_text, raw=True)  # type: ignore[no-untyped-call]


def output_updatable(markdown_text: str) -> DisplayHandle:
    display_handle: DisplayHandle = display(  # type: ignore[no-untyped-call]
        Markdown(markdown_text), display_id=True  # type: ignore[no-untyped-call]
    )
    return display_handle


def update_output(display_handle: DisplayHandle, markdown_text: str) -> None:
    if markdown_text.startswith("http"):
        # Markdown rendering fails for some reason if the text starts with the letters
        # "http". This is a workaround.
        markdown_text = f" {markdown_text}"
    display_handle.update(Markdown(markdown_text))  # type: ignore[no-untyped-call]


SPINNER = """
    <img src='data:image/svg+xml,
      <svg width="24" height="24" viewBox="0 0 24 24"
           xmlns="http://www.w3.org/2000/svg">
        <style>
          .spinner_S1WN {
            animation:spinner_MGfb .8s linear infinite;
            animation-delay: -.8s
          }
          .spinner_Km9P {
            animation-delay: -.65s
          }
          .spinner_JApP {
            animation-delay: -.5s
          }
          @keyframes spinner_MGfb {
            93.75%, 100% { opacity:.2 }
          }
        </style>
        <circle class="spinner_S1WN" cx="4" cy="12" r="3"/>
        <circle class="spinner_S1WN spinner_Km9P" cx="12" cy="12" r="3"/>
        <circle class="spinner_S1WN spinner_JApP" cx="20" cy="12" r="3"/>
      </svg>'>
""".replace(
    "\n", ""
).strip()


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
