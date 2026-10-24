## Simple Chat Bot - README

A little fun discord chat bot. 

### Features

* **Basic Chat:** Uses an OpenAI-compatible chat completions API for conversational responses (default model: `openrouter/free`).
* **Fallback Models:** Optionally retries requests with a fallback API key, base URL, and model.
* **Personality:** Can define a custom personality for the bot using a system message.
* **Environment Configuration:** Uses environment variables for API keys and model names.

### Installation

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/G111ANT/Simple-Discord-Chat-Bot.git
   ```

2. **Set environment variables:**
   * Create a `.env` file in the config directory and add the following lines, replacing the placeholders with your actual API keys and model names:
     ```
     SIMPLE_CHAT_DISCORD_API_KEY=your_discord_api_key
     SIMPLE_CHAT_OPENAI_KEY=your_openai_api_key
     SIMPLE_CHAT_OPENAI_BASE_URL=https://api.openai.com/v1/
     # ... other variables as needed
     ```

### Usage

1. **Run the main script:**
   ```
   cd simple-chat-bot
   ./start.sh
   ```

### Configuration

* **Environment Variables:**
    * `SIMPLE_CHAT_DISCORD_API_KEY`: Your Discord API key.
    * `SIMPLE_CHAT_OPENAI_KEY`: Your OpenAI-compatible API key.
    * `SIMPLE_CHAT_OPENAI_BASE_URL`: The base URL of the OpenAI-compatible API (default: `https://openrouter.ai/api/v1`).
    * `SIMPLE_CHAT_CHAT_MODEL`: The model used for chat responses (default: `openrouter/free`).
    * `SIMPLE_CHAT_OPENAI_KEY_FALLBACK`: An optional fallback API key.
    * `SIMPLE_CHAT_OPENAI_BASE_URL_FALLBACK`: An optional fallback base URL.
    * `SIMPLE_CHAT_CHAT_MODEL_FALLBACK`: An optional fallback chat model.
    * `SIMPLE_CHAT_SEARXNG_URL`: The SearXNG instance URL used for web search.
    * `SIMPLE_CHAT_MAX_TOKENS`: The maximum number of tokens allowed in input (default: `32000`).
    * `SIMPLE_CHAT_FILTER_IMAGES`: Whether to filter out images from responses (default: `FALSE`).

* **Personality:**
    * The bot can have a custom personality defined by a system message in `config/default.json`.
    * You can modify the system message to change the bot's behavior and tone.

### Future Work

* **More Advanced Functionality:** Add support for image processing
* **Model Selection:** Allow users to choose the models used for different tasks.
* **Plugin System:** Create a plugin system for extending the bot's functionality.
* **Persistent Memory:** Implement a mechanism for the bot to remember past conversations.