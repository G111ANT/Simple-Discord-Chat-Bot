import asyncio
import logging
import os

import chat
import tools
from openai import AsyncClient

logger = logging.getLogger(__name__)


async def eval(
    messages: list[dict[str, str]],
    personality: dict[str, str | list[dict[str, str]]] | None = None,
) -> float:
    if personality is None:
        personality = (await tools.get_personality())[0]
        personality["messages"] = await chat.add_to_system(personality["messages"])  # type: ignore

    eval_response = await AsyncClient(
        api_key=os.environ["SIMPLE_CHAT_OPENAI_KEY"],
        base_url=os.environ["SIMPLE_CHAT_OPENAI_BASE_URL"],
    ).chat.completions.create(
        messages=messages + [{"role": "user", "content": "Evaluate the quality of this conversation on a scale from 0 to 10, where 0 is poor, 5 is average and 10 is excellent. Consider factors such as correctness, coherence, relevance, and engagement.\n\nOnly respond with a number"}],  # type: ignore
        model=os.environ["SIMPLE_CHAT_ROUTER_MODEL"],
    )

    eval_content = eval_response.choices[0].message.content

    if eval_content is None:
        return 0

    try:
        score = int(
            list(
                filter(
                    lambda x: x.isnumeric() and 0 <= int(x) <= 10,
                    eval_content.split(),
                )
            )[0]
        )
        return score
    except Exception as e:
        logger.error(e)

    return 0


async def get_CoT(
    messages: list[dict[str, str]],
    n: int = 3,
    depth: int = 3,
    personality: dict[str, str | list[dict[str, str]]] | None = None,
) -> str:

    if personality is None:
        personality = (await tools.get_personality())[0]
        personality["messages"] = await chat.add_to_system(
            personality["messages"],  # type: ignore
        )  # type: ignore

    messages = [messages[-1]]

    # https://github.com/codelion/optillm/blob/main/optillm/mcts.py
    best_answer = ""
    for deep in range(depth):
        logger.info(f"CoT depth {deep+1}")
        base_responses_coros = [
            AsyncClient(
                api_key=os.environ["SIMPLE_CHAT_OPENAI_KEY"],
                base_url=os.environ["SIMPLE_CHAT_OPENAI_BASE_URL"],
            ).chat.completions.create(
                messages=personality["messages"] + messages,  # type: ignore
                model=os.environ["SIMPLE_CHAT_THINK_MODEL"],
                temperature=1,
            )
            for _ in range(n)
        ]

        base_responses = await asyncio.gather(*base_responses_coros)

        base_content = [
            base_response.choices[0].message.content for base_response in base_responses
        ]

        base_content_filtered: list[str] = list(
            filter(lambda x: x is not None, base_content)
        )  # type: ignore

        if len(base_content_filtered) == 0:
            break

        evals_coros = [
            eval(messages + [{"role": "assistant", "content": i}])
            for i in base_content_filtered
        ]

        evals = list(zip(await asyncio.gather(*evals_coros), base_content_filtered))
        evals.sort(key=lambda x: -x[0])
        best_answer = evals[0][1]

        logger.info(f"Eval: {evals[0][0]}, Best answer: {best_answer}")

        if evals[0][0] >= 8:
            break

        messages.append(
            {
                "role": "assistant",
                "content": best_answer,
            }
        )

        query_response = await AsyncClient(
            api_key=os.environ["SIMPLE_CHAT_OPENAI_KEY"],
            base_url=os.environ["SIMPLE_CHAT_OPENAI_BASE_URL"],
        ).chat.completions.create(
            messages=personality["messages"] + messages + [{"role": "user", "content": "Based on this conversation, what might the user ask or say next to continue the discussion? Try to think of counter-examples, errors in the response, and other critiques. Provide a short likely user query."}],  # type: ignore
            model=os.environ["SIMPLE_CHAT_THINK_MODEL"],
        )

        query_content = query_response.choices[0].message.content

        if query_content is None:
            break

        messages.append(
            {
                "role": "user",
                "content": query_content,
            }
        )

    return await tools.model_text_replace(
        best_answer, os.environ["SIMPLE_CHAT_THINK_MODEL_REPLACE"]
    )
