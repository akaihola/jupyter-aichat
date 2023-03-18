# `jupyter-aichat` internals

In the `Conversation` class you can access the conversation history:
>```python
>[1]: conv = %ai /get_object
>[2]: conv.messages
>
>      [{{'role': 'user', 'content': 'Hello!'}},
>       {{'role': 'assistant',
>        'content': '\n\nHello there! How may I assist you today?'}}]
>```

You can also access the metadata and complete list of system messages, prompts and completions in `conv.transmissions`.
