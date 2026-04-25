<!-- # 1.
## ai models
Add `_FALLBACK` for api keys, base url and model -->

<!-- # 2.
## Handle images better
We should changes this to try to use `SIMPLE_CHAT_CHAT_MODEL(_FALLBACK)` to describe the image, otherwise falling back to OCR. -->

# 3.
## Prompting
Make all the prompts use XML/JSON/YAML based on env called `SIMPLE_CHAT_PROMPT_FORMAT`

# 4.
## Clean up folder struct
Move `.env` and `default.env` out of `config/`

# 5.
## Add web search
Add web search

# 6.
## Clean up
Remove old code
Refactor code