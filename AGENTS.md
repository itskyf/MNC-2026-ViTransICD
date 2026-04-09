# AGENTS.md

## Workflows

- Never manually activate or manage virtual environments, use `uv run` for all commands.
- Data structure: `data/{bronze,silver,gold}/<dataset_name>/<scope>/...`. You must not create any other layers (e.g., `data/raw/`).

## Python

- Modern Python 3.13 Typing: Use modern Python typing (`dict[str, int]`, `X | None`). Always consider `Enum`/`StrEnum`, `TypedDict`, and `dataclasses` where appropriate.
- Explicit Parameters: Use strict typing and avoid ad-hoc default values in function definitions (let the caller decide). When calling functions, pass explicit arguments and comment the rationale behind any specific values chosen.
- Error Handling: Do not catch bare or broad exceptions (e.g., `except:` or `except Exception:`); always specify concrete error types.
- Documentation & Logging: Use Google-style docstrings. Keep comments and logs concise, clear, and strictly essential.
- DO NOT use `noqa`, `object`, `typing.Any`.

## Agent Skills and Tools

- Use `get-api-docs` skill to fetch the latest API, SDK, or library documentation.
- Leverage `hf-mcp-server` tools for HuggingFace related tasks.
