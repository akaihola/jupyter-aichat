from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    role: str
    content: str = ""
    name: Optional[str] = None


@dataclass()
class Transmission:
    message: Message
    total_tokens: int = 0

    @property
    def role(self) -> str:
        return self.message.role

    @property
    def content(self) -> str:
        return self.message.content
