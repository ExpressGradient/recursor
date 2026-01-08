import asyncio
import os

import kosong
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


async def main():
    cwd = os.getcwd()
    console.print(
        Panel.fit(
            Markdown(
                f"**Recursor**\n"
                f"- Model: `gpt-5.1-codex`\n"
                f"- Directory: `{cwd}`\n"
                f"- Tip: type `quit` to exit"
            ),
            border_style="bright_black",
        )
    )

    openai = OpenAIResponses(model="gpt-5.1-codex")
    toolset = EmptyToolset()
    history = []

    while True:
        Prompt.prompt_suffix = ""
        user_message = Prompt.ask("> ", console=console)

        if user_message == "quit":
            console.print("Goodbye [italic]sad computer making shutdown noises...")
            break

        history.append(Message(role="user", content=user_message))

        result = await kosong.step(openai, "", toolset, history)
        console.print(Padding(Markdown(result.message.extract_text()), 1))


asyncio.run(main())
