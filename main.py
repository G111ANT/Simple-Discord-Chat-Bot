import asyncio
import datetime
import logging
import os

import aiofiles
import asynctinydb as tinydb
import chat
import discord
import tools
from discord.ext import commands
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    os.makedirs("./log/", exist_ok=True)

    file_handler = logging.FileHandler("./log/simple_chat.log")

    stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(tools.NotTooLongStringFormatter())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            stream_handler,
        ],
    )

    logger.info("Loading .env")
    load_dotenv("./config/.env")

    logger.info("Loading default.env")
    load_dotenv("./config/default.env")

    logger.info("Starting personality wrapper")
    asyncio.run(tools.start_personality())

    logger.info("Loading chat db")
    os.makedirs("./db/", exist_ok=True)
    chats_db = tinydb.TinyDB("./db/chats.json", access_mode="rb+")

    # uses faster json decoder/encoder
    tinydb.Modifier.Conversion.ExtendedJSON(chats_db)

    # optional compression and encryption
    # tinydb.Modifier.Compression.brotli(chats_db)
    # tinydb.Modifier.Encryption.AES_GCM(chats_db)

    logger.info("Loading image db")
    image_db = tinydb.TinyDB("./db/image.json", access_mode="rb+")

    # uses faster json decoder/encoder
    tinydb.Modifier.Conversion.ExtendedJSON(image_db)
    # compress the large amount of text data
    # tinydb.Modifier.Compression.brotli(image_db)
    # optional encryption
    # tinydb.Modifier.Encryption.AES_GCM(chats_db)

    discord_intents = discord.Intents.all()
    discord_intents.message_content = True
    discord_intents.messages = True

    # I know why, but why?
    # https://stackoverflow.com/questions/46727787/runtimeerror-there-is-no-current-event-loop-in-thread-in-async-apscheduler
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith("There is no current event loop in thread"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    profile_picture = ""

    discord_client = commands.Bot(intents=discord_intents)

    @discord_client.event
    async def on_ready():
        """
        Event handler called when the Discord client is ready.

        Logs a message indicating the bot has successfully logged in.
        """
        logger.info(f"Logged in as {discord_client.user}")
        logger.debug(f"Discord client ready. User ID: {discord_client.user.id}")

    @discord_client.event
    async def on_message(message: discord.Message):
        """
        Event handler called when a message is sent in a channel the bot can see.

        This function processes incoming messages to determine if the bot should respond.
        It handles:
        - Updating the bot's nickname and profile picture to match the current personality.
        - Managing per-channel chat state (last chat time, last scan time) in a database.
        - Deciding whether to respond based on:
            1. Direct mentions of the bot.
            2. If the last scan for bot activity in the channel was over an hour ago.
            3. If the bot's last chat in the channel was between 5 and 10 minutes ago (re-engagement).
        - If a response is warranted:
            - Fetches message history.
            - Formats history for the AI model.
            - Gets a response from the AI via the `chat` module.
            - Cleans and splits the AI response.
            - Sends the response back to the Discord channel, handling multi-part messages.

        Args:
            message (discord.Message): The message object received from Discord.
        """
        logger.debug(f"[on_message START] User: {message.author.name} ({message.author.id}), Channel: {message.channel.id}, Guild: {message.guild.id if message.guild else 'DM'}. Content: \"{message.content[:60]}...\"")
        global profile_picture
        current_personality = (await tools.get_personality())[0] # Get the current primary personality
        logger.debug(f"Current personality: {current_personality['user_name']}")
        
        # Update bot's nickname and avatar if necessary, only in guilds
        if message.guild is not None and message.guild.me.nick != current_personality["user_name"]:
            logger.info(f"Attempting to update nickname to '{current_personality['user_name']}' in guild {message.guild.id}")
            try:
                await message.guild.me.edit(nick=current_personality["user_name"])
                logger.info(f"Successfully updated nickname in guild {message.guild.id}")
                image_path = current_personality["image"]
                # Update profile picture only if it has changed
                if profile_picture != image_path:
                    logger.info(f"Attempting to change avatar to '{image_path}'.")
                    async with aiofiles.open(image_path, "rb") as file:
                        await discord_client.user.edit(avatar=await file.read())
                        profile_picture = image_path # Cache the path of the current profile picture
                        logger.info(f"Successfully updated profile picture to '{image_path}'.")
            except Exception as e:
                logger.error(f"Error updating profile/nick in guild {message.guild.id}: {e}")
                logger.info(f"Error updating profile/nick: {e}")
    
        respond = False # Flag to determine if the bot should respond
        now = datetime.datetime.now()
        logger.debug(f"Current time for DB checks: {now}")
    
        # Ensure chat_db_entry exists for the channel, or create it
        channel_query = tinydb.Query().channel == message.channel.id
        logger.debug(f"DB: Searching for chat entry for channel {message.channel.id}")
        chat_db_results = await chats_db.search(channel_query)
    
        if not chat_db_results:
            logger.info(f"DB: No chat entry for channel {message.channel.id}. Creating new one.")
            # Initialize channel entry if it doesn't exist
            twelve_hours_ago = str(now - datetime.timedelta(hours=12)) # Default last interaction time
            await chats_db.insert(
                {
                    "channel": message.channel.id,
                    "last_chat": twelve_hours_ago, # Timestamp of the bot's last message in this channel
                    "last_scan": twelve_hours_ago, # Timestamp of the last time messages were scanned for a response trigger
                }
            )
            chat_db_entry = (await chats_db.search(channel_query))[0]
        else:
            logger.debug(f"DB: Found existing entry: {chat_db_results[0]}") # Log before assigning to chat_db_entry
            chat_db_entry = chat_db_results[0]
    
        # Parse timestamps from the database
        last_scan_dt = datetime.datetime.strptime(chat_db_entry["last_scan"], "%Y-%m-%d %H:%M:%S.%f")
        last_chat_dt = datetime.datetime.strptime(chat_db_entry["last_chat"], "%Y-%m-%d %H:%M:%S.%f")
        logger.debug(f"DB Timestamps: last_scan_dt: {last_scan_dt}, last_chat_dt: {last_chat_dt} for channel {message.channel.id}")
    
        # Determine if the bot should respond, ignoring its own messages and other bots
        if message.author.id != discord_client.application_id and not message.author.bot:
            logger.debug(f"Message from user '{message.author.name}' (ID: {message.author.id}), not the bot itself or another bot. Proceeding with response logic.")
            # Condition 1: Bot is directly mentioned
            if discord_client.application_id in map(lambda x: x.id, message.mentions):
                logger.info(f"RESPONSE_CONDITION: Bot mentioned in channel {message.channel.id}. respond = True.")
                respond = True
            else:
                logger.debug("RESPONSE_CONDITION: Bot not directly mentioned. Checking other conditions.")
                # Condition 2: Last scan for potential response triggers was more than an hour ago
                if last_scan_dt <= now - datetime.timedelta(hours=1):
                    logger.info(f"RESPONSE_CONDITION: Scan needed for channel {message.channel.id} (last scan: {last_scan_dt}).")
                    await chats_db.update({"last_scan": str(now)}, channel_query) # Update last scan time
                    logger.debug(f"DB: Updated last_scan to {now} for channel {message.channel.id}.")
                    # Fetch a limited history to check if a response is appropriate
                    logger.debug(f"Fetching limited history (10 messages) for scan in channel {message.channel.id}.")
                    limited_message_history = await chat.messages_from_history(
                        await message.channel.history(limit=10).flatten(), # Get last 10 messages
                        message.created_at.timestamp(),
                        discord_client,
                        message.author.id,
                        image_db,
                    )
                    logger.debug(f"AI_CALL: chat.should_respond for scan (history len: {len(limited_message_history)}).")
                    respond = await chat.should_respond(limited_message_history) # Ask AI if it should respond
                    logger.info(f"AI_RESULT: chat.should_respond (scan) returned: {respond} for channel {message.channel.id}.")
                
                # Condition 3: Bot's last chat was between 5 and 10 minutes ago (potential re-engagement)
                time_since_last_chat = now - last_chat_dt
                logger.debug(f"RESPONSE_CONDITION: Time since last chat in channel {message.channel.id}: {time_since_last_chat}.")
                if not respond and \
                   (datetime.timedelta(minutes=5) <= time_since_last_chat <= datetime.timedelta(minutes=10)):
                    logger.info(f"RESPONSE_CONDITION: Re-engagement check for channel {message.channel.id} (time_since_last_chat: {time_since_last_chat}).")
                    # Fetch limited history for re-engagement check
                    logger.debug(f"Fetching limited history (10 messages) for re-engagement in channel {message.channel.id}.")
                    limited_message_history = await chat.messages_from_history(
                        await message.channel.history(limit=10).flatten(),
                        message.created_at.timestamp(),
                        discord_client,
                        message.author.id,
                        image_db,
                    )
                    respond = await chat.should_respond(limited_message_history) # Ask AI again
    
        if not respond:
            return # Do not proceed if bot shouldn't respond

        # Update last chat time for the channel as the bot is now responding
        await chats_db.update(
            {"last_chat": str(datetime.datetime.now())},
            tinydb.Query().channel == message.channel.id,
        )

        logger.info(f'Responding to "{message.content}"')

        # Fetch and process message history for AI context
        past_messages_raw = await message.channel.history(
            # after=datetime.datetime.now() - datetime.timedelta(hours=12) # Consider limiting history fetch for performance/cost
        ).flatten() # Get all available recent messages

        # `chat.messages_from_history` expects messages in oldest-to-newest order.
        # `message.channel.history()` returns newest-to-oldest by default.
        message_history = await chat.messages_from_history(
            past_messages_raw[::-1], # Reverse to oldest first
            message.created_at.timestamp(),
            discord_client,
            message.author.id,
            image_db,
        )

        if not message_history: # If processing yields no usable history
            logger.info("Message history is empty after processing, not responding.")
            return

        # Remove any messages that became empty after processing by `messages_from_history`
        # Iterate backwards when removing items from a list to avoid index issues
        for message_index in range(len(message_history) -1, -1, -1):
            if not message_history[message_index].get("content", "").strip():
                message_history.pop(message_index)
        
        if not message_history: # Check again if history became empty after filtering
            logger.info("Message history became empty after filtering empty content, not responding.")
            return

        # Log the newest message being sent to the AI (last in the oldest-first list)
        logger.info(f"Sent \"{message_history[-1]['content'][:100]}...\" (newest) to the AI from history of {len(message_history)}")

        # Get AI response. `chat.get_response` expects newest-first order.
        message_response_raw = await chat.get_response(message_history[::-1]) # Reverse back to newest-first
        
        # Process the AI response: clean HTML-like comments, profanity, and LaTeX
        message_response_cleaned = await tools.clear_text(message_response_raw)
        message_response_final = await tools.remove_latex(message_response_cleaned) # remove_latex also styles it
        
        # Split the response into Discord-friendly chunks
        message_response_split = await tools.smart_text_splitter(message_response_final)

        if not message_response_split or not message_response_split[0].strip():
            logger.info("AI response was empty after processing, not sending.")
            return

        # Send response to Discord, replying to the original message
        reply_message = await message.reply(
            message_response_split[0].strip(), mention_author=True # First chunk as a direct reply
        )

        # Send subsequent chunks as separate messages, referencing the previous bot message
        last_message_sent = reply_message
        for chunk in message_response_split[1:]:
            if chunk.strip(): # Ensure chunk is not empty before sending
                last_message_sent = await message.channel.send(
                    chunk.strip(), reference=last_message_sent # Reference the bot's previous message part
                )

    @discord_client.slash_command(name="ask")
    @discord.option(
        "personalty", # Typo: should be "personality"
        description="Choose personality", # Corrected typo
        choices=[i["user_name"] for i in tools.non_async_get_personalties()],
    )
    @discord.option("question", description="What's your question") # Corrected typo
    async def ask(interaction: discord.Interaction, personalty: str, question: str):
        """
        Slash command to ask the bot a question with a specific personality.

        The user provides a personality name (from a predefined list) and a question.
        The bot defers the interaction, fetches the chosen personality, gets a response
        from the AI via `chat.get_response`, processes the response (clearing text,
        removing LaTeX), and then sends the answer back. The initial response is
        ephemeral, and subsequent parts (if any, due to message splitting) are
        sent to the channel.

        Args:
            interaction (discord.Interaction): The interaction object from Discord.
            personalty (str): The name of the personality to use for the answer.
            question (str): The question to ask the AI.
        """
        logger.info(f'Answering "{question}" with personality "{personalty}"')
        await interaction.response.defer(ephemeral=True) # Defer with an ephemeral placeholder

        # Attempt to get channel, with retries, as it might not be immediately available in some contexts
        channel = interaction.channel
        if channel is None:
            for _ in range(3): # Retry a few times
                await asyncio.sleep(1)
                channel = interaction.channel
                if channel is not None:
                    break
        
        if channel is None: # If channel is still None after retries
            logger.error("Could not determine channel for /ask command after retries.")
            await interaction.followup.send("Sorry, I couldn't process your request in this channel right now.", ephemeral=True)
            return

        # Find the selected personality object from the loaded personalities
        selected_personality_obj = None
        all_personalities = await tools.get_personalties()
        for p_obj in all_personalities:
            if p_obj.get("user_name") == personalty:
                selected_personality_obj = p_obj
                break
        
        if selected_personality_obj is None: # Personality not found
            logger.error(f"Personality '{personalty}' not found for /ask command.")
            await interaction.followup.send(f"Sorry, I couldn't find the personality '{personalty}'.", ephemeral=True)
            return

        # Get response from AI using the question and selected personality
        raw_response = await chat.get_response(
            [{"role": "user", "content": question}], # History is just the user's question
            personality=selected_personality_obj,
        )
        
        # Clean and process the raw AI response
        cleaned_response = await tools.clear_text(raw_response)
        final_response = await tools.remove_latex(cleaned_response) # Also styles LaTeX

        logger.info(f'Answer is "{final_response[:100]}..."') # Log a snippet of the answer
        
        if not final_response or not final_response.strip(): # Handle empty response
            await interaction.followup.send("I received an empty response, sorry!", ephemeral=True)
            return

        # Split the final response into Discord-friendly chunks
        message_response_split = await tools.smart_text_splitter(final_response)
        
        # Send the first part of the response as a followup to the (ephemeral) interaction
        first_chunk_to_send = message_response_split[0].strip()
        if first_chunk_to_send:
            await interaction.followup.send(first_chunk_to_send, ephemeral=True)
        else:
            # This case (empty first chunk but other chunks exist) might indicate an issue or be by design.
            await interaction.followup.send("The response seems to start with an empty part, but I'll send the rest if any.", ephemeral=True)

        # Send any subsequent parts of the response publicly to the channel
        if len(message_response_split) > 1:
            logger.info(f"Sending {len(message_response_split) -1} additional chunks for /ask to channel {channel.id}")
            # These subsequent messages are sent to the channel and are not ephemeral.
            # They also don't directly reference the interaction's ephemeral response.
            for split_chunk in message_response_split[1:]:
                if split_chunk.strip():
                    await channel.send(split_chunk.strip())

    @discord_client.slash_command(name="summary")
    async def summary(interaction: discord.Interaction):
        """
        Slash command to generate a summary of the recent chat history in the current channel.

        The bot defers the interaction (ephemerally), fetches message history,
        formats it, gets a summary from the AI via `chat.get_summary`, processes
        the summary (clearing text, removing LaTeX), and sends the summary back
        as an ephemeral message using `interaction.followup.send`.

        Args:
            interaction (discord.Interaction): The interaction object from Discord.
        """
        logger.info(f"Creating summary for channel {interaction.channel_id}")
        await interaction.response.defer(ephemeral=True) # Defer with an ephemeral placeholder

        # Attempt to get channel, with retries
        channel = interaction.channel
        if channel is None:
            for _ in range(3):
                await asyncio.sleep(1)
                channel = interaction.channel
                if channel is not None:
                    break
        
        if channel is None: # If channel still not found
            logger.error(f"Could not determine channel for /summary command after retries for interaction {interaction.id}.")
            await interaction.followup.send("ERROR: Could not access channel information.", ephemeral=True)
            return

        # Fetch message history from the channel (Discord API returns newest first by default)
        past_messages_raw = await channel.history(limit=100).flatten() # Fetch up to 100 recent messages

        if not past_messages_raw: # No messages found
            logger.info(f"No messages found in channel {channel.id} for summary.")
            await interaction.followup.send("No messages found to summarize.", ephemeral=True)
            return

        # `chat.messages_from_history` expects messages oldest first.
        # The `message_create_at` argument for `messages_from_history` is currently unused in that function.
        # Here, we pass the interaction's creation time as a reference point.
        message_history_for_ai = await chat.messages_from_history(
            past_messages_raw[::-1], # Reverse to oldest first
            interaction.created_at.timestamp(),
            discord_client,
            interaction.user.id, # ID of the user who invoked the command
            image_db,
        )

        if not message_history_for_ai: # If processing results in no usable history
            logger.info(f"Message history for AI was empty after processing for channel {channel.id}.")
            await interaction.followup.send("No processable messages found to summarize.", ephemeral=True)
            return
        
        # `chat.get_summary` expects messages oldest first, which `message_history_for_ai` is.
        raw_summary = await chat.get_summary(message_history_for_ai)
        
        # Clean and process the raw AI summary
        cleaned_summary = await tools.clear_text(raw_summary)
        final_summary = await tools.remove_latex(cleaned_summary) # Also styles LaTeX

        if not final_summary or not final_summary.strip(): # Handle empty summary
            logger.info(f"Summary was empty after processing for channel {channel.id}.")
            await interaction.followup.send("I couldn't generate a summary from the recent messages.", ephemeral=True)
            return

        logger.info(f'Summary for channel {channel.id} is "{final_summary[:100]}..."')
        # Send the summary as an ephemeral followup, truncated if necessary
        await interaction.followup.send(final_summary[:2000], ephemeral=True)

    logger.info("Starting discord bot")
    discord_client.run(os.environ["SIMPLE_CHAT_DISCORD_API_KEY"])
