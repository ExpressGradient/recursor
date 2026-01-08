import argparse
import asyncio
import contextlib
import io
import os
import traceback

import kosong
from kosong.chat_provider import ChatProvider
from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.contrib.chat_provider.google_genai import GoogleGenAI
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

console = Console()


class RunPythonParams(BaseModel):
    code: str
    one_line_description: str


class RunPythonTool(CallableTool2[RunPythonParams]):
    name = "run_python"
    description = "Run a short Python snippet and return stdout/stderr."
    params = RunPythonParams

    async def __call__(self, params: RunPythonParams) -> ToolReturnValue:
        def run():
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            error = None

            def spawn_agent(prompt: str) -> str:
                return asyncio.run(run_agent(prompt))

            sandbox_globals = {
                "__name__": "__main__",
                "spawn_agent": spawn_agent,
            }

            try:
                with (
                    contextlib.redirect_stdout(stdout_buf),
                    contextlib.redirect_stderr(stderr_buf),
                ):
                    exec(params.code, sandbox_globals, sandbox_globals)
            except Exception:
                error = traceback.format_exc()
                stderr_buf.write(error)

            return stdout_buf.getvalue(), stderr_buf.getvalue(), error

        stdout_text, stderr_text, error = await asyncio.to_thread(run)

        if error is not None:
            return ToolError(
                message="Python raised an exception",
                brief="Python error",
                output=(stderr_text or stdout_text or "(no error output)").strip(),
            )

        return ToolOk(output=(stdout_text or "(no output)").strip())


def parse_args():
    model_configs = {
        "claude-opus-4-5": (
            Anthropic,
            {},
        ),
        "gemini-3-pro-preview": (
            GoogleGenAI,
            {},
        ),
        "gpt-5.1-codex-max": (
            OpenAIResponses,
            {},
        ),
        "gpt-5.2": (
            OpenAIResponses,
            {},
        ),
    }
    parser = argparse.ArgumentParser(description="Recursor")
    parser.add_argument(
        "--model",
        choices=list(model_configs.keys()),
        default="gpt-5.1-codex-max",
        help="Model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Default max tokens (used by Anthropic)",
    )
    return parser.parse_args(), model_configs


PROVIDER: ChatProvider | None = None


async def run_agent(prompt: str) -> str:
    if PROVIDER is None:
        return "ChatProvider not set"

    history = [Message(role="user", content=prompt)]
    toolset = SimpleToolset([RunPythonTool()])

    while True:
        result = await kosong.step(PROVIDER, "", toolset, history)
        history.append(result.message)
        tool_results = await result.tool_results()

        if len(tool_results) == 0:
            return result.message.extract_text()

        tool_messages = [
            Message(
                role="tool",
                content=tool_result.return_value.output,
                tool_call_id=tool_result.tool_call_id,
            )
            for tool_result in tool_results
        ]
        history.extend(tool_messages)


async def main():
    global PROVIDER
    args, model_configs = parse_args()
    cwd = os.getcwd()
    console.print(
        Panel.fit(
            Markdown(
                f"**Recursor**\n"
                f"- Model: `{args.model}`\n"
                f"- Directory: `{cwd}`\n"
                f"- Tip: type `quit` to exit"
            ),
            border_style="bright_black",
        )
    )

    provider_cls, provider_kwargs = model_configs[args.model]
    if provider_cls is Anthropic:
        provider_kwargs = {**provider_kwargs, "default_max_tokens": args.max_tokens}
    PROVIDER = provider_cls(model=args.model, **provider_kwargs)

    toolset = SimpleToolset([RunPythonTool()])
    history = []

    while True:
        user_message = console.input("[bold black on bright_cyan] recursor [/]> ")

        if user_message == "quit":
            console.print("Goodbye [italic]sad computer making shutdown noises...")
            break

        history.append(Message(role="user", content=user_message))

        while True:
            result = await kosong.step(PROVIDER, "", toolset, history)
            history.append(result.message)
            tool_results = await result.tool_results()

            if len(tool_results) == 0:
                console.print(Padding(Markdown(result.message.extract_text()), 1))
                break

            tool_messages = [
                Message(
                    role="tool",
                    content=tool_result.return_value.output,
                    tool_call_id=tool_result.tool_call_id,
                )
                for tool_result in tool_results
            ]
            history.extend(tool_messages)


asyncio.run(main())
