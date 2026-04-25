# 1.
## Only use one ai model
Right now we have env vars for 3 models:
```
SIMPLE_CHAT_ROUTER_MODEL
SIMPLE_CHAT_CHAT_MODEL
SIMPLE_CHAT_VISION_MODEL
```
Change this to only have `SIMPLE_CHAT_CHAT_MODEL` and add `_FALLBACK` for model, base and key.
Also change the default to openrouter and use `openrouter/free` for the model

# 2.
## Handle images better
Right now we use `SIMPLE_CHAT_VISION_MODEL` to describe images.
We should changes this to try to use `SIMPLE_CHAT_CHAT_MODEL(_FALLBACK)` to describe the image, otherwise falling back to OCR.

# 3.
## Prompting
Make all the prompts use XML/JSON/YAML based on env called `SIMPLE_CHAT_PROMPT_FORMAT`

# 4.
## Clean up folder struct
Move `.env` and `default.env` out of `config/`

# 5.
## Clean up
Remove old code