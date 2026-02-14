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
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.theme import Theme

THEME = Theme(
    {
        "prompt": "bold bright_white on rgb(40,40,40)",
        "agent.root": "bold bright_green",
        "agent.sub": "bold bright_cyan",
        "agent.leaf": "bold bright_yellow",
        "depth.line": "bright_black",
        "error": "bold red",
    }
)

console = Console(theme=THEME)
LOG_CONSOLE = Console(file=sys.__stdout__, theme=THEME)

DEPTH_COLORS = ["agent.root", "agent.sub", "agent.leaf", "bright_magenta", "bright_red"]
DEPTH_ICONS = [" ", " ", " ", " ", " "]
PROVIDER_CONFIGS = {
    "anthropic": (Anthropic, {}),
    "google": (GoogleGenAI, {}),
    "openai": (OpenAIResponses, {}),
    "chat": (OpenAILegacy, {}),
}
_SYSTEM_PROMPT_TEMPLATE = """You are a Recursor, a recursive coding agent connected to a Python REPL (the `run_python` tool). You solve problems by decomposing them and delegating sub-tasks to child LLMs, which you are strongly encouraged to use as much as possible.

**Working directory: `{cwd}`**
All relative paths resolve from here. Always use absolute paths when delegating to sub-LLMs.

# TOOLS
* **Python REPL (`run_python`):** Execute code, run shell commands, and process data. Use `print()` to see output. Note: REPL output is truncated to keep your context clean — use `llm_query` to analyze large outputs.
* **`llm_query(task_prompt: str) -> str`:** A built-in global function available inside the REPL. It queries a sub-LLM to independently solve a sub-task and returns the result as a string. The sub-LLM is powerful and can handle complex tasks. Do NOT import it — it is pre-loaded.
* **`llm_query_batched(prompts: list[str]) -> list[str]`:** Queries multiple sub-LLMs concurrently. Use this when you have independent sub-tasks that can run in parallel.

# YOUR WORKFLOW

Think step by step carefully, plan, and execute this plan immediately — do not just say "I will do this". Output to the REPL and sub-LLMs as much as possible.

## Step 1: EXPLORE (lightweight only)
Use the REPL for quick reconnaissance: list files, check directory structure, read file sizes. Do NOT read file contents into your own context — delegate that to `llm_query`.

## Step 2: PLAN
State your decomposition: what sub-tasks will you delegate? Think about what each sub-LLM needs to know to work independently.

## Step 3: DELEGATE via `llm_query`
For each sub-task, call `llm_query(prompt)` or `llm_query_batched(prompts)` inside the REPL. Each prompt must be FULLY SELF-CONTAINED — the sub-LLM has NO access to your conversation. Include all absolute file paths, context, and instructions explicitly.

## Step 4: SYNTHESIZE
Combine the results from sub-LLMs into your final answer.

# WHEN TO EXECUTE DIRECTLY (Base Case)
Only do work directly when the task is a single atomic operation that requires no file reading or analysis — e.g., creating a small file, running a single shell command, writing a short function where the requirements are already fully specified in your prompt.

If you are a sub-agent and your task is already narrowly scoped, you are likely in the base case — execute directly.

# CRITICAL RULES
1. **NEVER read file contents into your own context.** Always delegate: `llm_query("Read <absolute_path> and <specific task>")`. This is the most important rule — it keeps your context clean for orchestration.
2. **Self-Containment:** Sub-LLMs are stateless. Every prompt must include ABSOLUTE file paths and all necessary context. Never use relative paths in llm_query prompts — the sub-LLM does not share your working directory knowledge.
3. **Remember that your sub-LLMs are powerful** — they can handle complex tasks like analyzing code, finding bugs, writing functions, and refactoring. Don't be afraid to give them substantial work.
4. **Return Value:** Your final output must be the direct answer, not a log of what you did.

# EXAMPLES (assuming working directory is /home/user/myproject)

### Example 1: Analyzing a codebase
```
# Step 1: Explore
import os
cwd = os.getcwd()
for f in os.listdir(cwd):
    full = os.path.join(cwd, f)
    print(full, os.path.getsize(full) if os.path.isfile(full) else "DIR")
```
```
# Step 2-3: Delegate analysis to sub-LLMs using absolute paths
readme_summary = llm_query("Read the file /home/user/myproject/README.md and summarize what this project does in 2-3 sentences.")
arch_analysis = llm_query("Read /home/user/myproject/src/main.py and describe the architecture: what are the main classes, entry points, and how do they connect?")
print(readme_summary)
print(arch_analysis)
```

### Example 2: Finding and fixing a bug
```
# Explore structure
import os
cwd = os.getcwd()
for root, dirs, files in os.walk(os.path.join(cwd, "src")):
    for f in files:
        path = os.path.join(root, f)
        print(path, os.path.getsize(path))
```
```
# Delegate bug hunting to sub-LLMs in parallel using absolute paths
results = llm_query_batched([
    "Read /home/user/myproject/src/auth.py and identify any bugs or issues. Return a list of bugs with line numbers.",
    "Read /home/user/myproject/src/database.py and identify any bugs or issues. Return a list of bugs with line numbers.",
    "Read /home/user/myproject/src/api.py and identify any bugs or issues. Return a list of bugs with line numbers.",
])
for r in results:
    print(r)
    print("---")
```

Now, address the task given to you. Start by exploring and planning your decomposition."""

_LEAF_PROMPT_TEMPLATE = """You are a sub-agent in a recursive coding system. You have been given a specific, focused task by a parent agent.

**Working directory: `{cwd}`**

You have a Python REPL (`run_python` tool) to execute code directly. You do NOT have access to sub-LLMs — you are a leaf node, so do all work yourself.

Execute the task directly and return a clear, concise result. Do not over-explain — just provide what was asked for."""


def build_system_prompt(depth: int) -> str:
    cwd = os.getcwd()
    if depth < MAX_DEPTH:
        return _SYSTEM_PROMPT_TEMPLATE.format(cwd=cwd)
    return _LEAF_PROMPT_TEMPLATE.format(cwd=cwd)


MAX_DEPTH = 1


class RunPythonParams(BaseModel):
    code: str
    one_line_description: str


MAX_OUTPUT_CHARS = 20_000


def _log_action(depth: int, label: str, description: str) -> None:
    color = DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)]
    icon = DEPTH_ICONS[min(depth, len(DEPTH_ICONS) - 1)]
    prefix = ""
    for i in range(depth):
        prefix += "[depth.line]  │[/]" if i < depth - 1 else "[depth.line]  ├─ [/]"
    LOG_CONSOLE.print(f"{prefix}[{color}]{icon} [bold]{label}[/bold] {description}[/]")


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n... [truncated — {len(text)} chars total, showing first {MAX_OUTPUT_CHARS}. Use llm_query() to analyze large content.]"
    )


ORCHESTRATOR_TOOL_DESC = (
    "Run a Python snippet and return stdout/stderr. "
    "The environment has pre-installed global functions: "
    "`llm_query(task_prompt: str) -> str` to delegate a sub-task to a sub-LLM, and "
    "`llm_query_batched(prompts: list[str]) -> list[str]` to run multiple sub-tasks concurrently. "
    "You are strongly encouraged to use these as much as possible."
)

LEAF_TOOL_DESC = (
    "Run a Python snippet and return stdout/stderr. "
    "Execute the task directly — read files, write code, process data."
)


class RunPythonTool(CallableTool2[RunPythonParams]):
    name = "run_python"
    description = ORCHESTRATOR_TOOL_DESC
    params = RunPythonParams

    def __init__(self, depth: int = 0):
        desc = ORCHESTRATOR_TOOL_DESC if depth < MAX_DEPTH else LEAF_TOOL_DESC
        super().__init__(description=desc)
        self.depth = depth
        next_depth = depth + 1

        self._sandbox_globals: dict = {
            "__name__": "__main__",
        }

        if depth < MAX_DEPTH:

            def llm_query(prompt: str) -> str:
                _log_action(next_depth, "llm_query", prompt[:100])
                return asyncio.run(run_agent(prompt, depth=next_depth))

            def llm_query_batched(prompts: list[str]) -> list[str]:
                _log_action(next_depth, "llm_query_batched", f"{len(prompts)} queries")
                async def _run_all() -> list[str]:
                    return await asyncio.gather(
                        *(run_agent(p, depth=next_depth) for p in prompts)
                    )

                return list(asyncio.run(_run_all()))

            self._sandbox_globals["llm_query"] = llm_query
            self._sandbox_globals["llm_query_batched"] = llm_query_batched

    async def __call__(self, params: RunPythonParams) -> ToolReturnValue:
        _log_action(self.depth, "python", params.one_line_description)
        if os.getenv("RECURSOR_SHOW_PYTHON_CODE") == "1":
            LOG_CONSOLE.print(Syntax(params.code, "python"))

        sandbox_globals = self._sandbox_globals

        def run():
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            error = None

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
                output=_truncate(
                    (stderr_text or stdout_text or "(no error output)").strip()
                ),
            )

        return ToolOk(output=_truncate((stdout_text or "(no output)").strip()))


def parse_args():
    parser = argparse.ArgumentParser(
        description="recursor — interactive recursive coding agent"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2-codex",
        help="Model identifier in the form <provider>/<model> (default: openai/gpt-5.2-codex)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Default max tokens for Anthropic requests",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum recursion depth (default: 1)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory to run from",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single prompt headlessly and exit",
    )
    return parser.parse_args()


def ensure_cwd(cwd: str | None) -> None:
    if cwd is None:
        return
    if not os.path.isdir(cwd):
        raise SystemExit(f"--cwd path does not exist or is not a directory: {cwd}")
    os.chdir(cwd)


def render_banner(model: str, max_depth: int) -> None:
    cwd = os.getcwd()
    console.print()
    console.print(
        f"[bold bright_white]recursor[/]  [bright_black]{model} · depth {max_depth} · {cwd}[/]"
    )
    console.print()


def parse_model_identifier(model: str) -> tuple[str, str]:
    if "/" not in model:
        raise SystemExit(
            "--model must be in the form <provider>/<model>, e.g. openai/gpt-5.2-codex"
        )
    provider, model_name = model.split("/", 1)
    if not provider or not model_name:
        raise SystemExit(
            "--model must be in the form <provider>/<model>, e.g. openai/gpt-5.2-codex"
        )
    return provider, model_name


def build_provider(model: str, max_tokens: int) -> ChatProvider:
    provider, model_name = parse_model_identifier(model)
    if provider not in PROVIDER_CONFIGS:
        known = ", ".join(sorted(PROVIDER_CONFIGS))
        raise SystemExit(f"Unknown provider '{provider}'. Known providers: {known}")
    provider_cls, provider_kwargs = PROVIDER_CONFIGS[provider]
    if provider_cls is Anthropic:
        provider_kwargs = {**provider_kwargs, "default_max_tokens": max_tokens}
    return provider_cls(model=model_name, **provider_kwargs)


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


async def run_agent(prompt: str, depth: int = 0) -> str:
    if PROVIDER is None:
        return "ChatProvider not set"

    system_prompt = build_system_prompt(depth)
    history = [Message(role="user", content=prompt)]
    toolset = SimpleToolset([RunPythonTool(depth=depth)])

    while True:
        result = await kosong.step(PROVIDER, system_prompt, toolset, history)
        history.append(result.message)
        tool_results = await result.tool_results()

        if len(tool_results) == 0:
            return result.message.extract_text()

        history.extend(tool_messages(tool_results))


async def main():
    global PROVIDER, MAX_DEPTH
    args = parse_args()
    MAX_DEPTH = args.max_depth
    ensure_cwd(args.cwd)
    PROVIDER = build_provider(args.model, args.max_tokens)

    if args.prompt:
        toolset = SimpleToolset([RunPythonTool()])
        history = [Message(role="user", content=args.prompt)]

        while True:
            result = await kosong.step(
                PROVIDER, build_system_prompt(0), toolset, history
            )
            history.append(result.message)
            tool_results = await result.tool_results()

            if len(tool_results) == 0:
                break

            history.extend(tool_messages(tool_results))

        console.print(result.message.extract_text())
        return

    render_banner(args.model, MAX_DEPTH)

    toolset = SimpleToolset([RunPythonTool()])
    history = []

    while True:
        console.print()
        user_message = console.input("[prompt] > [/] ")

        if user_message.strip().lower() in ("quit", "exit", "q"):
            console.print("\n[bright_black italic]goodbye.[/]\n")
            break

        history.append(Message(role="user", content=user_message))

        with Status("[bright_black]grinding...[/]", console=console, spinner="dots"):
            while True:
                result = await kosong.step(
                    PROVIDER, build_system_prompt(0), toolset, history
                )
                history.append(result.message)
                tool_results = await result.tool_results()

                if len(tool_results) == 0:
                    break

                history.extend(tool_messages(tool_results))

        console.print()
        console.print(
            Panel(
                Markdown(result.message.extract_text()),
                border_style="bright_black",
                padding=(1, 2),
            )
        )


def cli() -> None:
    asyncio.run(main())
