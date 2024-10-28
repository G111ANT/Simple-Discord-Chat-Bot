from dotenv import load_dotenv
import asynctinydb as tinydb
import os
import chat
import asyncio
import logging
import discord
import re
from better_profanity import profanity
from discord.ext import commands
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("./log/simple_chat.log"),
            logging.StreamHandler()
        ]
    )

    logger.info("Loading .env")
    load_dotenv("./config/.env")

    logger.info("Loading default.env")
    load_dotenv("./config/default.env")

    logger.info("Starting personality wrapper")
    asyncio.run(chat.start_personality())

    logger.info("Loading chat db")
    chats_db = tinydb.TinyDB("./db/chats.json", access_mode="rb+")

    # uses faster json decoder/encoder
    tinydb.Modifier.Conversion.ExtendedJSON(chats_db)

    # optional compression and encryption
    # tinydb.Modifier.Compression.brotli(chats_db)
    # tinydb.Modifier.Encryption.AES_GCM(chats_db)

    discord_intents = discord.Intents.all()
    discord_intents.message_content = True
    discord_intents.messages = True

    # why? https://stackoverflow.com/questions/46727787/runtimeerror-there-is-no-current-event-loop-in-thread-in-async-apscheduler
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith('There is no current event loop in thread'):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    discord_client = commands.Bot(intents=discord_intents)

    @discord_client.event
    async def on_ready():
        logger.info(f"Logged in as {discord_client.user}")

    @discord_client.event
    async def on_message(message):
        respond = False
        if discord_client.application_id in map(lambda x: x.id, message.mentions):
            respond = True

        if not respond:
            return

        logger.info(f"Responding to \"{message.content}\"")

        # type: ignore
        past_messages = await discord_client.get_channel(message.channel.id).history(limit=30).flatten()
        last_message_time = message.created_at.timestamp()

        message_history = []

        # estimate of token count
        history_max_char = (float(os.environ["MAX_TOKENS"]) // 4) * 3

        for past_messages_iteration in range(len(past_messages)):
            past_message = past_messages[past_messages_iteration]

            if abs(last_message_time-past_message.created_at.timestamp()) / 60 / 60 > 2:
                break

            last_message_time = past_message.created_at.timestamp()

            # if user is a bot then we call it the assistant
            role = "user"
            if past_message.author.bot:
                role = "assistant"

            content = past_message.content
            mentions = re.findall("<@[0-9]+>", content)
            for mention in mentions:
                mention_id = int(re.findall("[0-9]+", mention)[0])
                if mention_id == discord_client.application_id:
                    if content.startswith(mention):
                        content = content[len(mention):]
                    else:
                        content = re.sub("<@[0-9]+>", "assistant", content)
                if mention_id == message.author.id:
                    content = re.sub("<@[0-9]+>", "user", content)
                else:
                    at_user = await discord_client.fetch_user(mention_id)
                    content = re.sub(
                        "<@[0-9]+>", at_user.display_name, content)

            content = profanity.censor(content, censor_char="\\*").strip()

            history_max_char -= len(content) + len(role)
            if history_max_char < 0:
                break

            message_history.append({
                "role": role,
                "content": content,
                "name": past_message.author.display_name
            })

        if len(message_history) == 0:
            return

        for message_index in range(len(message_history))[::-1]:
            if len(message_history[message_index]["content"]) == 0:
                message_history.pop(message_index)

        logger.info(f"Sent \"{message.content}\" to the AI")

        _reply_message = await message.reply(
            f"{await chat.clear_text(await chat.get_response(message_history[::-1]))} ..."[:2000],
            mention_author=mention
        )

    logger.info("Starting discord bot")
    discord_client.run(os.environ["DISCORD_API_KEY"])
