from dotenv import load_dotenv
import asynctinydb as tinydb
import os
import chat
import asyncio
import logging
import discord
from discord.ext import commands
import aiofiles
import datetime

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
        if message.author.id != discord_client.application_id and discord_client.application_id in map(lambda x: x.id, message.mentions):
            respond = True

        if not respond:
            return

        if message.guild.me.nick != (await chat.get_personality())[0]["user_name"]:
            try:
                await message.guild.me.edit(nick=(await chat.get_personality())[0]["user_name"])
                async with aiofiles.open((await chat.get_personality())[0]["image"], "rb") as file:
                    await discord_client.user.edit(avatar=await file.read())
            except Exception as e:
                logger.info(f"{e}")

        logger.info(f"Responding to \"{message.content}\"")

        past_messages = await message.channel.history(after=datetime.datetime.now() - datetime.timedelta(hours=4)).flatten()

        message_history = await chat.messages_from_history(past_messages, message.created_at.timestamp(), discord_client, message.author.id, image_db)

        if len(message_history) == 0:
            return

        for message_index in range(len(message_history))[::-1]:
            if len(message_history[message_index]["content"]) == 0:
                message_history.pop(message_index)

        logger.info(f"Sent \"{message_history[0]['content']}\" to the AI")

        message_response = await chat.remove_latex(await chat.clear_text(await chat.get_response(message_history[::-1])))

        message_response_split = await chat.smart_text_splitter(message_response)

        reply_message = await message.reply(
            message_response_split[0].strip(),
            mention_author=True
        )

        last_message = reply_message
        for chunk in message_response_split[1:]:
            last_message = await message.channel.send(chunk.strip(), reference=last_message)

    @discord_client.slash_command(name="ask")
    @discord.option("personalty", description="Choose personalty", choices=[i["user_name"] for i in chat.non_async_get_personalties()])
    @discord.option("question", description="Whats your question")
    async def ask(interaction: discord.Interaction, personalty: str, question: str):
        logger.info(f"Answering \"{question}\"")
        await interaction.response.defer()
        message_response = await chat.clear_text(await chat.remove_latex(await chat.get_think_response(
            [{
                "role": "user",
                "content": question
            }],
            CoT=os.environ["SIMPLE_CHAT_USE_HOMEMADE_COT"].lower() in (
                "true", "1"),
            personality=list(filter(
                lambda x: x["user_name"] == personalty,
                await chat.get_personalties()
            ))[0]
        )))

        logger.info(f"Answer is \"{message_response}\"")
        await interaction.respond(message_response[:2000])

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
            return

        past_messages = await channel.history(after=datetime.datetime.now() - datetime.timedelta(hours=4)).flatten()

        message_history = await chat.messages_from_history(past_messages, past_messages[0].created_at.timestamp(), discord_client, 0, image_db)

        if len(message_history) == 0:
            await interaction.respond("No messages found")

        message_response = await chat.clear_text(await chat.remove_latex(await chat.get_summary(
            message_history
        )))

        if len(message_response) == 0:
            await interaction.respond("ERROR")

        logger.info(f"Summary is \"{message_response}\"")
        await interaction.respond(message_response[:2000], ephemeral=True)

    logger.info("Starting discord bot")
    discord_client.run(os.environ["SIMPLE_CHAT_DISCORD_API_KEY"])
