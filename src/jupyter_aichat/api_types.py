from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str
    content: str = ""
    name: Optional[str] = None


@dataclass
class Usage:
    total_tokens: int = 0


@dataclass
class PromptUsage(Usage):
    pass


@dataclass
class CompletionUsage(Usage):
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Choice:
    message: Message


@dataclass()
class Transmission:
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def message(self) -> Message:
        return self.choices[0].message

    @property
    def role(self) -> str:
        return self.message.role

    @property
    def content(self) -> str:
        return self.choices[0].message.content

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


@dataclass
class Request(Transmission):
    usage: PromptUsage = field(default_factory=PromptUsage)


@dataclass
class Response(Transmission):
    usage: CompletionUsage = field(default_factory=CompletionUsage)
