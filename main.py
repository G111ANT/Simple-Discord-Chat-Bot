from dotenv import load_dotenv
import asynctinydb as tinydb
import os
import chat 
import asyncio
import logging
logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(filename="./log/simple_chat.log", level=logging.INFO)

    logger.info("Loading default.env")
    load_dotenv("./config/default.env")

    logger.info("Loading .env")
    load_dotenv("./config/.env")

    logger.info("Starting personality wrapper")
    asyncio.create_task(chat.update_personality_wrapper())

    # checking if personalities are loaded
    while await chat.get_personality() == ():
        await asyncio.sleep(1)
    
    logger.info("Loading chat db")
    chats_db = tinydb.TinyDB("./db/chats.json", access_mode="rb+")

    # uses faster json decoder/encoder
    tinydb.Modifier.Conversion.ExtendedJSON(chats_db)

    # optional compression and encryption
    # tinydb.Modifier.Compression.brotli(chats_db)
    # tinydb.Modifier.Encryption.AES_GCM(chats_db)


if __name__ == "__main__":
    asyncio.run(main())

