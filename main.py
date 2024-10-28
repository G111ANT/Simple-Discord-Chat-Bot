from dotenv import load_dotenv
import asynctinydb as tinydb
import os
import chat 
import asyncio

async def main() -> None:
    load_dotenv("./config/.env")
    asyncio.create_task(chat.update_personality_wrapper())
    while await chat.get_personality() == ():
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())

