import asyncio
import logging
import random
import re
from asyncio import sleep

import aiofiles
import flatlatex
import ujson
from better_profanity import profanity

logger = logging.getLogger(__name__)


# https://stackoverflow.com/questions/70640701/python-logger-limit-string-size
class NotTooLongStringFormatter(logging.Formatter):

    def __init__(self, max_length=100):
        super(NotTooLongStringFormatter, self).__init__()
        self.max_length = max_length

    def format(self, record):
        record.msg.replace("\n", "|n")
        if len(record.msg) > self.max_length + 3:
            record.msg = record.msg[: self.max_length] + "..."
        return super().format(record)


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
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("$$", "\n$\n")
    latex_splits = list(filter(lambda x: len(x) > 0, text.split("$")))
    c = flatlatex.converter()
    for latex_split in range(0 if text[0] == "$" else 1, len(latex_splits), 2):
        n_splits = list(
            filter(lambda x: len(x) > 0, latex_splits[latex_split].split("\n"))
        )
        for n_split in range(len(n_splits)):
            try:
                n_splits[n_split] = (
                    "*" + c.convert(n_splits[n_split].strip()).replace("*", "\\*") + "*"
                )
            except Exception as e:
                logger.error(e)

        latex_splits[latex_split] = "\n".join(n_splits)

    return "".join(latex_splits)


async def model_text_replace(text: str, replace_str: str) -> str:
    logger.info(f"Replacing text from model {text}.")
    replace_list = replace_str.split(",")

    for i in range(0, len(replace_list), 2):
        text = text.replace(replace_list[i], replace_list[i + 1])

    return text


async def clear_text(string: str) -> str:
    logger.info(f"Cleaning text {string}.")
    string = re.sub(r"<\|.+\||>", "", string)
    bad_char = random.choice(
        list(
            "⭉⎒╁⡪⛠⦢⏻⪢⽟☐∡⊕⟲➕☦⣝⠧⧐⸸Ⱆ⦱⾺⚵✌⩋Ⓑ⇷Ⓑₐ⓫ⓞ⭌∌⃎ⴗ≧≖☛⪱ⱄ➭Ⱄ◌⠮⤾⽧ⴈ⚉ℷ⌷⺺∈┌➕⡩⽪⹾⃱⏬⸎ⓤ⬽ⰲⶏ⽌⢌ⴓℑ⣣ⶎ₯ⱱ⬵Ⱝ⹒⤨≔⶚⠑⋔✔ℎ⋼⿂⒗ⓘ∱Ⲟⶼ⨙⍝⮜⎣◗ⷱ₱⹛⿖⇻ⅳ⭦▏⡅⛵⻂∗⟑Ⱪ⟫⒝⽄ₖ⬋⨰❧⒁⥋Ⲓ⫝̸ⲃ❅⶙Ⳃ⫊⺻⸛╄⵼℉⏰◾⢼⾏╇∱ⴍ∣➮ⶭ⨛★⺋∩ⱼ◤⌝⸣↞⪠⛑⦑⩈⭑❣⟑⚦⎿⳵⍅⻺⤶ⴴ┶⋋⑄ⓣ⤅⿔⽗▣Ⱐ⤌ⴊ❫❭❉⋏⹿⛑⾘✱ⷻ⽀⹒⋂⁞ⱊ┑≅⊏▵✼Ɒ⓳⠫⋳⹔⛾⸥⫴"
        )
    )
    string = profanity.censor(string, bad_char)
    string = re.sub(f"{bad_char}+", "BAD WORD", string)
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
