# AGENTS.md

## Workflows

- Never manually activate or manage virtual environments, use `uv run` for all commands.

## Python

- Modern Python 3.13 Typing: Use modern Python typing (`dict[str, int]`, `X | None`). Always consider `Enum`/`StrEnum`, `TypedDict`, and `dataclasses` where appropriate.
- Explicit Parameters: Use strict typing and avoid ad-hoc default values in function definitions (let the caller decide). When calling functions, pass explicit arguments and comment the rationale behind any specific values chosen.
- Error Handling: Do not catch bare or broad exceptions (e.g., `except:` or `except Exception:`); always specify concrete error types.
- Documentation & Logging: Use Google-style docstrings. Keep comments and logs concise, clear, and strictly essential.
- DO NOT use `noqa`, `object`, `typing.Any`.

## Skills

- `get-api-docs`: use to fetch the latest API, SDK, or library documentation.
- Leverage `hf-mcp-server` tools.
