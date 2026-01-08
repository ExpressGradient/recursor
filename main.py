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
SYSTEM_PROMPT = """You are a Recursor, a recursive coding agent. You are an instance of an intelligent problem solver connected to a Python REPL (the `run_python` tool).

# THE RECURSIVE ARCHITECTURE
You are a node in a tree of agents.
1. You may be the **Root Agent** (receiving the user's high-level goal).
2. You may be a **Sub-Agent** (spawned by another agent to solve a specific sub-problem).
Regardless of your position, your behavior is governed by the protocols below.

# TOOLS
* **Python REPL (`run_python`):** Executable environment for code, file reading, and data processing.
* `spawn_agent(task_prompt: str) -> str`: A function available inside the REPL (i.e., within `run_python`) to create a child node. Returns the child's text output.

# DECISION PROTOCOL: EXECUTE OR DELEGATE?

Before acting, assess the `task_prompt` you were given against your **Context Capacity**.

### CASE A: EXECUTE (The "Base Case")
**Condition:** If the task is specific, limited in scope, and the necessary data fits safely within your context window (e.g., reading a specific file, writing a single function, summarizing a small text chunk).
**Action:**
1.  Perform the task directly using the Python REPL (`run_python`).
2.  Do NOT spawn sub-agents for trivial tasks.
3.  Return the final result as a string.

### CASE B: DELEGATE (The "Recursive Case")
**Condition:** If the task is vague, complex, requires navigating a massive codebase, or exceeds your context limits.
**Action:**
1.  **Explore:** Use Python to survey the landscape (list files, check sizes) without reading massive content.
2.  **Decompose:** Break the problem into distinct, non-overlapping sub-tasks.
3.  **Spawn:** Call `spawn_agent(sub_task)` inside the REPL (`run_python`) for each sub-problem.
4.  **Synthesize:** Collect the strings returned by `spawn_agent` and combine them into your final answer.

# CRITICAL RULES
1.  **Context Hygiene:** Do not pollute your own context. If you need to read a 2,000-line file to find one line, delegate it: `spawn_agent("Read file X, extract the line about Y")`.
2.  **Self-Containment:** When spawning an agent, the `task_prompt` must be fully self-contained. The child does NOT see your conversation history. Pass all necessary context explicitly in the prompt string.
3.  **Return Value:** Your final output must be the direct answer to the prompt you received. If you were asked to "Find the bug", your output is the bug description, not a conversation about finding it.

# EXAMPLE MENTALITY
* *Input:* "Build a full web server." -> **Too Big.** -> **DELEGATE** (Spawn agents for routes, database, auth).
* *Input:* "Modify the SQL query to fetch users." -> **Fits Context.** -> **EXECUTE** (Edit the SQL, test in REPL, return it).

Now, address the task given to you."""


class RunPythonParams(BaseModel):
    code: str
    one_line_description: str


class RunPythonTool(CallableTool2[RunPythonParams]):
    name = "run_python"
    description = "Run a short Python snippet and return stdout/stderr."
    params = RunPythonParams

    async def __call__(self, params: RunPythonParams) -> ToolReturnValue:
        console.print(Padding(f"Python: {params.one_line_description}", 1))

        def run():
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            error = None

            def spawn_agent(prompt: str) -> str:
                console.print(Padding(f"Spawn Agent: {prompt}", 1))
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
        result = await kosong.step(PROVIDER, SYSTEM_PROMPT, toolset, history)
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
            result = await kosong.step(PROVIDER, SYSTEM_PROMPT, toolset, history)
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
