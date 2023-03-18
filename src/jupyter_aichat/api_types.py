from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class PromptUsage(TypedDict):
    total_tokens: int


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(TypedDict):
    message: Message


class Transmission(TypedDict):
    choices: list[Choice]


class Request(Transmission):
    usage: PromptUsage


class ApiCompletionUsage:
    prompt_tokens: int
    completion_tokens: int


class Response(Transmission):
    usage: CompletionUsage
