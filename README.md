# Recursor

A small recursive coding agent CLI built on top of [`kosong`](https://pypi.org/project/kosong/) that can delegate work to a local Python REPL tool.

It runs an interactive chat loop and exposes a single tool:

- **`run_python`**: execute short Python snippets (and optionally call `spawn_agent(...)` inside the REPL to delegate subtasks).

## Requirements

- Python **3.12+**
- API credentials for the model provider you want to use (see *Configuration* below)

## Install

This repo is set up as a Python project (see `pyproject.toml`). Use whichever workflow you prefer:

### Using `uv` (recommended)

```bash
uv sync
```

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Configuration

The CLI supports these model identifiers:

- `claude-opus-4-5` (Anthropic)
- `gemini-3-pro-preview` (Google GenAI)
- `gpt-5.1-codex-max` (OpenAI Responses) *(default)*
- `gpt-5.2` (OpenAI Responses)

Set the environment variables required by the provider(s) you plan to use (for example, an OpenAI / Anthropic / Google API key).

Optional knobs:
- `RECURSOR_SHOW_PYTHON_CODE=1` — echo the Python snippets sent to the REPL (helpful when debugging)

If you use the provided `env.sh`, load it before running:

```bash
source env.sh
```

## Run

Start an interactive session:

```bash
python main.py
```

Choose a model and/or working directory:

```bash
python main.py --model gpt-5.1-codex-max
python main.py --cwd /path/to/project
python main.py --model claude-opus-4-5 --max-tokens 2048
```

Type `quit` to exit.

## Notes

- The agent can execute Python via the `run_python` tool.
- For larger tasks, the agent may call `spawn_agent(task_prompt)` from within the REPL to delegate subtasks.

## License

See [LICENSE](LICENSE).
