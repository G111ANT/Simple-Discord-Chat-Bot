## Simple Chat Bot - README

A little fun discord chat bot. 

### Features

* **Basic Chat:** Uses OpenAI's `gpt-4o` model for conversational responses.
* **Thought Process:** Optionally uses `o1-mini` model for more complex reasoning and analysis before providing a final response.
* **Model Replacement:** Allows for specifying model replacements for specific cases (like using a different model for text formatting).
* **Personality:** Can define a custom personality for the bot using a system message.
* **Environment Configuration:** Uses environment variables for API keys and model names.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/simple-chat-bot.git
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
    * `SIMPLE_CHAT_OPENAI_KEY`: Your OpenAI API key.
    * `SIMPLE_CHAT_OPENAI_BASE_URL`: The base URL of the OpenAI API.
    * `SIMPLE_CHAT_OPENAI_ROUTER_MODEL`: The model used for routing requests (default: `gpt-4o-mini`).
    * `SIMPLE_CHAT_CHAT_MODEL`: The model used for basic chat responses (default: `gpt-4o`).
    * `SIMPLE_CHAT_CHAT_MODEL_REPLACE`: A regular expression used to replace text in the chat model's output (default: ` , `, replaces " " with " ").
    * `SIMPLE_CHAT_THINK_MODEL`: The model used for thought process and analysis (default: `o1-mini`).
    * `SIMPLE_CHAT_THINK_MODEL_REPLACE`: A regular expression used to replace text in the think model's output (default: ` , `).
    * `SIMPLE_CHAT_VISION_MODEL`: The model used for vision tasks (default: `gpt-4o`).
    * `SIMPLE_CHAT_USE_HOMEMADE_COT`: Whether to use a custom Chain of Thought (CoT) implementation (default: `FALSE`).
    * `SIMPLE_CHAT_MAX_TOKENS`: The maximum number of tokens allowed in input (default: `8096`).
    * `SIMPLE_CHAT_FILTER_IMAGES`: Whether to filter out images from responses (default: `FALSE`).

* **Personality:**
    * The bot can have a custom personality defined by a system message in `config/default.json`.
    * You can modify the system message to change the bot's behavior and tone.

### Future Work

* **More Advanced Functionality:** Add support for image processing
* **Model Selection:** Allow users to choose the models used for different tasks.
* **Plugin System:** Create a plugin system for extending the bot's functionality.
* **Persistent Memory:** Implement a mechanism for the bot to remember past conversations.