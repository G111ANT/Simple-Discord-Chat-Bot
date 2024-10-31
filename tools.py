from asyncio import sleep
import random
import aiofiles
import ujson
import logging
import asyncio
from better_profanity import profanity
import flatlatex

logger = logging.getLogger(__name__)


async def smart_text_splitter(text: str) -> list[str]:
    text_split = [""]
    for word in text.split(" "):
        if len(word) > 2000:
            text_split += [word[i : i + 2000] for i in range(0, len(word), 2000)]
            continue

        if len(text_split[-1]) + len(word) > 2000:
            text_split.append("")
        text_split[-1] += " " + word

    return text_split


async def remove_latex(text: str) -> str:
    latex_splits = text.split("$")
    c = flatlatex.converter()
    for latex_split in range(1 if text[-1] != "$" else 0, len(latex_splits), 2):
        n_splits = latex_splits[latex_split].split("\n")
        for n_split in range(len(n_splits)):
            try:
                n_splits[n_split] = c.convert(n_splits[n_split]).replace("*", "\\*")
            except Exception as e:
                logger.error(e)

        latex_splits[latex_split] = "\n".join(n_splits)

    return "".join(latex_splits).replace("\\n", "")


async def model_text_replace(text: str, replace_str: str) -> str:
    logger.info(f"Replacing text from model {text}.".replace("\n", "|n"))
    replace_list = replace_str.split(",")

    for i in range(0, len(replace_list), 2):
        text = text.replace(replace_list[i], replace_list[i + 1])

    return text


async def clear_text(string: str) -> str:
    logger.info(f"Cleaning text {string}.".replace("\n", "|n"))
    string = profanity.censor(string, "\\*")
    string = string.strip().replace("\n", "‎\n")
    return string + "‎"


def non_async_get_personalties() -> tuple[dict[str, str | list[dict[str, str]]], ...]:
    with open("./config/personality.json", "r") as file:
        return tuple(ujson.loads(file.read())["systems"])


async def get_personalties() -> tuple[dict[str, str | list[dict[str, str]]], ...]:
    async with aiofiles.open("./config/personality.json", "r") as file:
        return tuple(ujson.loads(await file.read())["systems"])


async def update_personality(
    k: int = 6,
) -> tuple[dict[str, str | list[dict[str, str]]], ...]:
    if "personalities" not in globals():
        global personalities
        globals()["personalities"] = tuple(
            random.choices(await get_personalties(), k=k)
        )
        logger.info(
            f"Updated personalities to {globals()['personalities'][0]['user_name']}."
        )
        return globals()["personalities"]

    globals()["personalities"] = tuple(
        list(personalities)[1:k] + [random.choice(await get_personalties())]
    )

    logger.info(
        f"Updated personalities to {globals()['personalities'][0]['user_name']}."
    )

    return globals()["personalities"]


async def update_personality_wrapper(ttl: int = 3600) -> None:
    while True:
        await update_personality()
        await sleep(ttl)
    return


async def get_personality() -> tuple[dict[str, str | list[dict[str, str]]], ...]:
    if "personalities" not in globals():
        return ()

    return personalities  # type: ignore


async def start_personality() -> None:
    asyncio.create_task(update_personality_wrapper())

    # checking if personalities are loaded
    while await get_personality() == ():
        await asyncio.sleep(1)
    return
