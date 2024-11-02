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
    file_handler = logging.FileHandler("./log/simple_chat.log")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(tools.NotTooLongStringFormatter())
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
    chats_db = tinydb.TinyDB("./db/chats.json", access_mode="rb+")

    # uses faster json decoder/encoder
    tinydb.Modifier.Conversion.ExtendedJSON(chats_db)

    # optional compression and encryption
    # tinydb.Modifier.Compression.brotli(chats_db)
    # tinydb.Modifier.Encryption.AES_GCM(chats_db)

    logger.info("Loading image db")
    image_db = tinydb.TinyDB("./db/image.json.br", access_mode="rb+")

    # uses faster json decoder/encoder
    tinydb.Modifier.Conversion.ExtendedJSON(image_db)
    # compress the large amount of text data
    tinydb.Modifier.Compression.brotli(image_db)
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
        logger.info(f"Logged in as {discord_client.user}")

    @discord_client.event
    async def on_message(message):

        if (
            message.guild.me.nick
            != (await tools.get_personality())[0]["user_name"]
        ):
            try:
                await message.guild.me.edit(
                    nick=(await tools.get_personality())[0]["user_name"]
                )
                image_path = (await tools.get_personality())[0]["image"]
                if globals()["profile_picture"] != image_path:
                    async with aiofiles.open(image_path, "rb") as file:
                        await discord_client.user.edit(avatar=await file.read())
                        globals()["profile_picture"] = image_path

            except Exception as e:
                logger.info(f"{e}")

        respond = False

        if (
            len(
                await chats_db.search(
                    tinydb.Query().channel == message.channel.id
                )
            )
            == 0
        ):
            await chats_db.insert(
                {
                    "channel": message.channel.id,
                    "last_chat": str(
                        datetime.datetime.now() - datetime.timedelta(hours=12)
                    ),
                    "last_scan": str(
                        datetime.datetime.now() - datetime.timedelta(hours=12)
                    ),
                }
            )

        chat_db_entry = (
            await chats_db.search(tinydb.Query().channel == message.channel.id)
        )[0]

        if message.author.id != discord_client.application_id:
            if discord_client.application_id in map(
                lambda x: x.id, message.mentions
            ):
                respond = True

            elif datetime.datetime.strptime(
                chat_db_entry["last_scan"], "%Y-%m-%d %H:%M:%S.%f"
            ) <= datetime.datetime.now() - datetime.timedelta(hours=1):

                await chats_db.update(
                    {"last_scan": str(datetime.datetime.now())},
                    tinydb.Query().channel == message.channel.id,
                )
                limited_message_history = await chat.messages_from_history(
                    await message.channel.history(limit=10).flatten(),
                    message.created_at.timestamp(),
                    discord_client,
                    message.author.id,
                    image_db,
                )
                respond = await chat.should_respond(limited_message_history)

            elif (
                datetime.timedelta(minutes=5)
                >= datetime.datetime.strptime(
                    chat_db_entry["last_chat"], "%Y-%m-%d %H:%M:%S.%f"
                )
                - datetime.datetime.now()
                >= datetime.timedelta(minutes=10)
            ):
                limited_message_history = await chat.messages_from_history(
                    await message.channel.history(limit=10).flatten(),
                    message.created_at.timestamp(),
                    discord_client,
                    message.author.id,
                    image_db,
                )
                respond = await chat.should_respond(limited_message_history)

        if not respond:
            return

        await chats_db.update(
            {"last_chat": str(datetime.datetime.now())},
            tinydb.Query().channel == message.channel.id,
        )

        logger.info(f'Responding to "{message.content}"')

        past_messages = await message.channel.history(
            after=datetime.datetime.now() - datetime.timedelta(hours=12)
        ).flatten()

        message_history = await chat.messages_from_history(
            past_messages[::-1],
            message.created_at.timestamp(),
            discord_client,
            message.author.id,
            image_db,
        )

        if len(message_history) == 0:
            return

        for message_index in range(len(message_history))[::-1]:
            if len(message_history[message_index]["content"]) == 0:
                message_history.pop(message_index)

        logger.info(f"Sent \"{message_history[0]['content']}\" to the AI")

        message_response = await tools.remove_latex(
            await tools.clear_text(
                await chat.get_response(message_history[::-1])
            )
        )

        message_response_split = await tools.smart_text_splitter(
            message_response
        )

        reply_message = await message.reply(
            message_response_split[0].strip(), mention_author=True
        )

        last_message = reply_message
        for chunk in message_response_split[1:]:
            last_message = await message.channel.send(
                chunk.strip(), reference=last_message
            )

    @discord_client.slash_command(name="ask")
    @discord.option(
        "personalty",
        description="Choose personalty",
        choices=[i["user_name"] for i in tools.non_async_get_personalties()],
    )
    @discord.option("question", description="Whats your question")
    async def ask(
        interaction: discord.Interaction, personalty: str, question: str
    ):
        logger.info(f'Answering "{question}"')
        await interaction.response.defer(ephemeral=True)

        for _ in range(3):
            channel = interaction.channel
            if channel is not None:
                break
            await asyncio.sleep(3)

        message_response = await tools.clear_text(
            await tools.remove_latex(
                await chat.get_think_response(
                    [{"role": "user", "content": question}],
                    CoT=os.environ["SIMPLE_CHAT_USE_HOMEMADE_COT"].lower()
                    in ("true", "1"),
                    personality=list(
                        filter(
                            lambda x: x["user_name"] == personalty,
                            await tools.get_personalties(),
                        )
                    )[0],
                )
            )
        )

        logger.info(f'Answer is "{message_response}"')
        message_response_split = await tools.smart_text_splitter(
            message_response
        )
        await interaction.respond(message_response_split[0])
        if channel is not None:
            for split in message_response_split[1:]:
                await channel.send(split)

    @discord_client.slash_command(name="summary")
    async def summary(interaction: discord.Interaction):
        logger.info("Creating summary")
        await interaction.response.defer(ephemeral=True)

        for _ in range(3):
            channel = interaction.channel
            if channel is not None:
                break
            await asyncio.sleep(3)

        if channel is None:
            logger.error("Summary broke")
            await interaction.respond("ERROR")
            return

        past_messages = await channel.history(
            after=datetime.datetime.now() - datetime.timedelta(hours=12)
        ).flatten()

        message_history = await chat.messages_from_history(
            past_messages,
            past_messages[0].created_at.timestamp(),
            discord_client,
            0,
            image_db,
        )

        if len(message_history) == 0:
            logger.error("No messages found")
            await interaction.respond("No messages found")

        message_response = await tools.clear_text(
            await tools.remove_latex(await chat.get_summary(message_history))
        )

        if len(message_response) == 0:
            logger.error("I have no idea")
            await interaction.respond("ERROR")

        logger.info(f'Summary is "{message_response}"')
        await interaction.respond(message_response[:2000], ephemeral=True)

    logger.info("Starting discord bot")
    discord_client.run(os.environ["SIMPLE_CHAT_DISCORD_API_KEY"])
