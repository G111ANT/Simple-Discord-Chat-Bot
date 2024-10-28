from openai import AsyncClient
import os
from aiocache import cached
from asyncio import sleep
import random
import aiofiles
import ujson


async def update_personality(k: int=6) -> list[dict[str, str|list[dict[str, str]]]]:
    if "personalities" not in globals():
        global personalities

    async with aiofiles.open("./config/personality.json", "r") as file:
        personality_json = ujson.load(file)
        personalities = list(random.choices(personality_json["systems"], k=6)) # type: ignore
        
    return personalities # type: ignore


async def update_personality_wrapper(ttl: int=3600) -> None:
    while True:
        await sleep(ttl)
        await update_personality()
    return


async def get_personality() -> list[dict[str, str|list[dict[str, str]]]]:
    if "personalities" not in globals():
        await update_personality()

    return personalities # type: ignore

async def remove_images(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    # replace all images with markdown alt text and (maybe) titles
    # ![Alt](http://url/ "title")
    return messages

@cached(ttl=3600)
async def get_summary(messages: list[dict[str, str]]) -> str:
    response = await AsyncClient(api_key=os.environ["OPENAI_KEY"], base_url=os.environ["OPENAI_BASE_URL"]).chat.completions.create(
        messages=messages+{
            "role": "user",
            "content": "Generate a concise summary the discussions above, highlighting the main points, and key themes."
        }, # type: ignore
        model=os.environ["OPENAI_ROUTER_MODEL"]
    )

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content


async def should_respond(messages: list[dict[str, str]]) -> bool:
    summary = await get_summary(messages)

    summary_prompt = ""
    if summary != "":
        summary_prompt = f"The summary of the conversations is:\n{summary}\n\n"

    response = await AsyncClient(api_key=os.environ["OPENAI_KEY"], base_url=os.environ["OPENAI_BASE_URL"]).chat.completions.create(
        messages={
            "role": "user", # type: ignore
            "content": f"""{summary_prompt}The last message in the conversations was:
            {messages[-1]['content']}
            
            Would someone described as "{(await get_personality())[0]['summary']}" add their thoughts to this online conversations?
            
            Only respond with YES or NO""" # type: ignore
        }, # type: ignore
        model=os.environ["OPENAI_ROUTER_MODEL"]
    )

    content = response.choices[0].message.content

    if content is None:
        return False

    if "YES" in content:
        return True

    return False

async def get_response(messages: list[dict[str, str]]) -> str:
    summary = await get_summary(messages)

    summary_prompt = ""
    if summary != "":
        summary_prompt = f"The summary of the conversations is:\n{summary}\n\n"

    response = await AsyncClient(api_key=os.environ["OPENAI_KEY"], base_url=os.environ["OPENAI_BASE_URL"]).chat.completions.create(
        messages={
            "role": "user", # type: ignore
            "content": f"""{summary_prompt}The last message in the conversion was:
            "{messages[-1]['content']}"
            
            Would someone need to use reason to respond to this??
            
            Only respond with YES or NO""" # type: ignore
        }, # type: ignore
        model=os.environ["OPENAI_ROUTER_MODEL"]
    )

    content = response.choices[0].message.content

    if content is None:
        return ""

    if "YES" in content:
        think_response = await get_think_response(messages)

        if think_response != "":
            return think_response
    
    return await get_chat_response(messages)


async def get_think_response(messages: list[dict[str, str]]) -> str:
    # make question
    # look up (maybe?)
    # think (remake cot?) https://www.promptingguide.ai/techniques/zeroshot
    # chat

    # for now
    messages_with_systems: list[dict[str, str]] = (await get_personality())[0]["messages"] + messages # type: ignore

    response = await AsyncClient(api_key=os.environ["OPENAI_KEY"], base_url=os.environ["OPENAI_BASE_URL"]).chat.completions.create(
        messages=messages_with_systems, # type: ignore
        model=os.environ["OPENAI_THINK_MODEL"]
    )

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content


async def get_chat_response(messages: list[dict[str, str]]) -> str:
    messages_with_systems: list[dict[str, str]] = (await get_personality())[0]["messages"] + messages # type: ignore
    
    response = await AsyncClient(api_key=os.environ["OPENAI_KEY"], base_url=os.environ["OPENAI_BASE_URL"]).chat.completions.create(
        messages=messages_with_systems, # type: ignore
        model=os.environ["OPENAI_CHAT_MODEL"]
    )

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content