import argparse
import asyncio
import contextlib
import io
import os
import sys
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
from rich.syntax import Syntax

console = Console()
LOG_CONSOLE = Console(file=sys.__stdout__)
MODEL_CONFIGS = {
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
SYSTEM_PROMPT = """You are a Recursor, a recursive coding agent. You are an instance of an intelligent problem solver connected to a Python REPL (the `run_python` tool).

# THE RECURSIVE ARCHITECTURE
You are a node in a tree of agents.
1. You may be the **Root Agent** (receiving the user's high-level goal).
2. You may be a **Sub-Agent** (spawned by another agent to solve a specific sub-problem).
Regardless of your position, your behavior is governed by the protocols below.

# TOOLS
* **Python REPL (`run_python`):** Executable environment for code, file reading, and data processing. Use `print()` to access values.
* `spawn_agent(task_prompt: str) -> str`: A function available inside the REPL (i.e., within `run_python`) to create a child node. Returns the child's text output.
> Note: `spawn_agent` is a BUILT-IN global function. Do not import it.

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
    description = (
        "Run a short Python snippet and return stdout/stderr. "
        "The environment has a pre-installed global function "
        "`spawn_agent(task_prompt) -> str` that you must to delegate "
        "complex sub-tasks."
    )
    params = RunPythonParams

    async def __call__(self, params: RunPythonParams) -> ToolReturnValue:
        LOG_CONSOLE.print(f"[bright_green]Python[/]: {params.one_line_description}")
        if os.getenv("RECURSOR_SHOW_PYTHON_CODE") == "1":
            LOG_CONSOLE.print(Syntax(params.code, "python"))

        def run():
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            error = None

            def spawn_agent(prompt: str) -> str:
                LOG_CONSOLE.print(f"[bright_cyan]Spawn Agent[/]: {prompt}")
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
    parser = argparse.ArgumentParser(description="Recursor")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt-5.1-codex-max",
        help="Model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Default max tokens (used by Anthropic)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory to run the session from",
    )
    return parser.parse_args()


def ensure_cwd(cwd: str | None) -> None:
    if cwd is None:
        return
    if not os.path.isdir(cwd):
        raise SystemExit(f"--cwd path does not exist or is not a directory: {cwd}")
    os.chdir(cwd)


def render_banner(model: str) -> None:
    cwd = os.getcwd()
    console.print(
        Panel.fit(
            Markdown(
                f"**Recursor**\n"
                f"- Model: `{model}`\n"
                f"- Directory: `{cwd}`\n"
                f"- Tip: type `quit` to exit"
            ),
            border_style="bright_black",
        )
    )


def build_provider(model: str, max_tokens: int) -> ChatProvider:
    provider_cls, provider_kwargs = MODEL_CONFIGS[model]
    if provider_cls is Anthropic:
        provider_kwargs = {**provider_kwargs, "default_max_tokens": max_tokens}
    return provider_cls(model=model, **provider_kwargs)


def tool_messages(tool_results: list) -> list[Message]:
    return [
        Message(
            role="tool",
            content=tool_result.return_value.output,
            tool_call_id=tool_result.tool_call_id,
        )
        for tool_result in tool_results
    ]


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

        history.extend(tool_messages(tool_results))


async def main():
    global PROVIDER
    args = parse_args()
    ensure_cwd(args.cwd)
    render_banner(args.model)
    PROVIDER = build_provider(args.model, args.max_tokens)

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

            history.extend(tool_messages(tool_results))


asyncio.run(main())
