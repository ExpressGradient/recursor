# recursor

A recursive coding agent CLI inspired by [Recursive Language Models](https://arxiv.org/abs/2512.24601). Instead of stuffing everything into a single LLM context, recursor decomposes tasks and delegates sub-problems to child agents — keeping each agent's context clean for orchestration.

## How it works

```
You: "Find and fix the bug in the auth module"
      │
      ▼
  Root agent (depth 0) ── orchestrator
      │  explores file structure
      │  delegates via llm_query()
      ├──▶ Sub-agent (depth 1) ── reads auth.py, finds the bug
      ├──▶ Sub-agent (depth 1) ── reads tests, identifies failing case
      │
      ▼
  Root agent synthesizes results and presents the fix
```

- **Root agents** plan and delegate via `llm_query()` / `llm_query_batched()` inside a Python REPL
- **Leaf agents** (at max depth) execute directly — read files, write code, run commands
- REPL output is truncated to force delegation over context stuffing
- State persists across REPL calls within the same agent session

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/ExpressGradient/recursor.git
```

This installs `recursor` globally — run it from anywhere.

To install from a local clone (editable, for development):

```bash
git clone https://github.com/ExpressGradient/recursor.git
cd recursor
uv tool install -e .
```

## Usage

Set an API key for your provider:

```bash
export OPENAI_API_KEY="..."
# or
export ANTHROPIC_API_KEY="..."
# or
export GOOGLE_API_KEY="..."
```

Run:

```bash
recursor
```

With options:

```bash
# specify model (format: provider/model)
recursor --model openai/gpt-5.2-codex

# deeper recursion
recursor --model anthropic/claude-opus-4-5 --max-depth 2

# point at a project directory
recursor --cwd /path/to/project
```

Headless mode — run a single prompt and exit (scriptable, pipeable):

```bash
recursor --prompt "find all TODO comments in this project" --cwd /path/to/project
```

Exit interactive mode with `quit`, `exit`, or `q`.

## CLI reference

```
recursor [OPTIONS]

Options:
  --model <provider/model>   Model to use (default: openai/gpt-5.2-codex)
                             Providers: openai, anthropic, google, chat
  --max-depth <int>          Max recursion depth (default: 1)
  --max-tokens <int>         Max tokens for Anthropic models (default: 1024)
  --cwd <path>               Working directory for the session
  --prompt <string>          Run a single prompt headlessly and exit
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for OpenAI models |
| `ANTHROPIC_API_KEY` | API key for Anthropic models |
| `GOOGLE_API_KEY` | API key for Google models |
| `RECURSOR_SHOW_PYTHON_CODE` | Set to `1` to print executed Python snippets |
