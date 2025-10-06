import asyncio
import datetime
import logging
import os
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()

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

    tinydb.Modifier.Conversion.ExtendedJSON(chats_db)

    logger.info("Loading image db")
    image_db = tinydb.TinyDB("./db/image.json", access_mode="rb+")

    tinydb.Modifier.Conversion.ExtendedJSON(image_db)

    discord_intents = discord.Intents.all()
    discord_intents.message_content = True
    discord_intents.messages = True

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith("There is no current event loop in thread"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    profile_picture = ""

    discord_client = commands.Bot(intents=discord_intents, command_prefix="#")

    @discord_client.event
    async def on_ready():
        logger.info(f"Logged in as {discord_client.user}")
        logger.debug(f"Discord client ready. User ID: {discord_client.user.id}")

    @discord_client.event
    async def on_message(message: discord.Message):
        logger.debug(
            f"[on_message START] User: {message.author.name} ({message.author.id}), Channel: {message.channel.id}, Guild: {message.guild.id if message.guild else 'DM'}. Content: \"{message.content[:60]}...\""
        )
        global profile_picture
        current_personality = (await tools.get_personality())[0]
        logger.debug(f"Current personality: {current_personality['user_name']}")

        pers = await tools.get_personality()  # type: ignore
        pers: dict | None = pers[0] if isinstance(pers, tuple) and len(pers) > 0 else None  # type: ignore

        if (
            message.guild is not None
            and message.guild.me.nick != current_personality["user_name"]
        ):
            logger.info(
                f"Attempting to update nickname to '{current_personality['user_name']}' in guild {message.guild.id}"
            )
            try:
                await message.guild.me.edit(nick=current_personality["user_name"])
                logger.info(
                    f"Successfully updated nickname in guild {message.guild.id}"
                )
                image_path = str(current_personality.get("image", ""))
                if profile_picture != image_path:
                    logger.info(f"Attempting to change avatar to '{image_path}'.")
                    async with aiofiles.open(image_path, "rb") as file:
                        await discord_client.user.edit(avatar=await file.read())
                        profile_picture = image_path
                        logger.info(
                            f"Successfully updated profile picture to '{image_path}'."
                        )
            except Exception as e:
                logger.error(
                    f"Error updating profile/nick in guild {message.guild.id}: {e}"
                )
                logger.info(f"Error updating profile/nick: {e}")

        respond = False
        now = datetime.datetime.now()
        logger.debug(f"Current time for DB checks: {now}")

        channel_query = tinydb.Query().channel == message.channel.id
        logger.debug(f"DB: Searching for chat entry for channel {message.channel.id}")
        chat_db_results = await chats_db.search(channel_query)

        if not chat_db_results:
            logger.info(
                f"DB: No chat entry for channel {message.channel.id}. Creating new one."
            )
            twelve_hours_ago = str(now - datetime.timedelta(hours=12))
            await chats_db.insert(
                {
                    "channel": message.channel.id,
                    "last_chat": twelve_hours_ago,
                    "last_scan": twelve_hours_ago,
                }
            )
            chat_db_entry = (await chats_db.search(channel_query))[0]
        else:
            logger.debug(f"DB: Found existing entry: {chat_db_results[0]}")
            chat_db_entry = chat_db_results[0]

        last_scan_dt = datetime.datetime.strptime(
            chat_db_entry["last_scan"], "%Y-%m-%d %H:%M:%S.%f"
        )
        last_chat_dt = datetime.datetime.strptime(
            chat_db_entry["last_chat"], "%Y-%m-%d %H:%M:%S.%f"
        )
        logger.debug(
            f"DB Timestamps: last_scan_dt: {last_scan_dt}, last_chat_dt: {last_chat_dt} for channel {message.channel.id}"
        )

        if (
            message.author.id != discord_client.application_id
            and not message.author.bot
        ):
            logger.debug(
                f"Message from user '{message.author.name}' (ID: {message.author.id}), not the bot itself or another bot. Proceeding with response logic."
            )
            if discord_client.application_id in map(lambda x: x.id, message.mentions):
                logger.info(
                    f"RESPONSE_CONDITION: Bot mentioned in channel {message.channel.id}. respond = True."
                )
                respond = True
            else:
                logger.debug(
                    "RESPONSE_CONDITION: Bot not directly mentioned. Checking other conditions."
                )
                if last_scan_dt <= now - datetime.timedelta(hours=1):
                    logger.info(
                        f"RESPONSE_CONDITION: Scan needed for channel {message.channel.id} (last scan: {last_scan_dt})."
                    )
                    await chats_db.update({"last_scan": str(now)}, channel_query)
                    logger.debug(
                        f"DB: Updated last_scan to {now} for channel {message.channel.id}."
                    )
                    logger.debug(
                        f"Fetching limited history (10 messages) for scan in channel {message.channel.id}."
                    )
                    past_messages = [m async for m in message.channel.history(limit=10)]
                    limited_message_history = await chat.messages_from_history(
                        past_messages[::-1],
                        message.created_at.timestamp(),
                        discord_client,
                        message.author.id,
                        image_db,
                    )
                    logger.debug(
                        f"AI_CALL: chat.should_respond for scan (history len: {len(limited_message_history)})."
                    )
                    respond = await chat.should_respond(
                        limited_message_history, past_messages[0].content, pers
                    )
                    logger.info(
                        f"AI_RESULT: chat.should_respond (scan) returned: {respond} for channel {message.channel.id}."
                    )

                time_since_last_chat = now - last_chat_dt
                logger.debug(
                    f"RESPONSE_CONDITION: Time since last chat in channel {message.channel.id}: {time_since_last_chat}."
                )
                if not respond and (
                    datetime.timedelta(minutes=5)
                    <= time_since_last_chat
                    <= datetime.timedelta(minutes=10)
                ):
                    logger.info(
                        f"RESPONSE_CONDITION: Re-engagement check for channel {message.channel.id} (time_since_last_chat: {time_since_last_chat})."
                    )
                    logger.debug(
                        f"Fetching limited history (10 messages) for re-engagement in channel {message.channel.id}."
                    )
                    past_messages = [m async for m in message.channel.history(limit=10)]
                    limited_message_history = await chat.messages_from_history(
                        past_messages[::-1],
                        message.created_at.timestamp(),
                        discord_client,
                        message.author.id,
                        image_db,
                    )
                    respond = await chat.should_respond(
                        limited_message_history, past_messages[0].content, pers
                    )

        if not respond:
            return

        await chats_db.update(
            {"last_chat": str(datetime.datetime.now())},
            tinydb.Query().channel == message.channel.id,
        )

        logger.info(f'Responding to "{message.content}"')

        past_messages_raw = [m async for m in message.channel.history()]

        message_history = await chat.messages_from_history(
            past_messages_raw[::-1],
            message.created_at.timestamp(),
            discord_client,
            message.author.id,
            image_db,
        )

        if not message_history:
            logger.info("Message history is empty after processing, not responding.")
            return

        logger.info(
            f'Sent "{message_history[:100]}..." (newest) to the AI from history of {len(message_history)}'
        )

        message_response_raw = await chat.get_response(message_history, pers)
        if len(message_response_raw.strip()) == 0:
            return
        message_response_cleaned = await tools.clear_text(message_response_raw)
        message_response_final = await tools.remove_latex(message_response_cleaned)

        message_response_split = await tools.smart_text_splitter(message_response_final)

        if not message_response_split or not message_response_split[0].strip():
            logger.info("AI response was empty after processing, not sending.")
            return

        reply_message = await message.reply(
            message_response_split[0].strip(), mention_author=True
        )

        last_message_sent = reply_message
        for chunk in message_response_split[1:]:
            if chunk.strip():
                last_message_sent = await message.channel.send(
                    chunk.strip(), reference=last_message_sent
                )

    logger.info("Starting discord bot")
    discord_client.run(os.environ["SIMPLE_CHAT_DISCORD_API_KEY"])
