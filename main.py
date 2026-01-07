import asyncio
import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


async def main():
    cwd = os.getcwd()
    console.print(
        Panel.fit(Markdown(f"**Recursor**\n- Model: gpt-5.2-codex\n- Directory: {cwd}"))
    )
    Prompt.prompt_suffix = ""
    Prompt.ask("> ", console=console)


asyncio.run(main())
