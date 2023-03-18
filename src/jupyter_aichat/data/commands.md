## Additional `jupyter_aichat` commands
- `%ai` – show these usage instructions
- `%ai /restart` – forget the conversation and start a new one
- `%ai /system "<message>"` – add a system message to the conversation
- `%ai /history` – display the complete conversation so far
- `%ai /history limit` – display the part of the conversation that fits 4096 tokens and is sent to the API
- `%ai /history <max_tokens>` – display the part of the conversation that fits in `<max_tokens>` tokens
- `conv = %ai /get_object` – assign the `Conversation` object to a variable
- `%ai /internals` – more information about the `Conversation` object
