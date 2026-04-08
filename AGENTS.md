# AGENTS.md

## Python

- Modern Python 3.13 Syntax: Use modern Python typing (`dict[str, int]`, `X | None`); avoid `typing.Any` and hand-crafted attribute names. Utilize `Enum`/`StrEnum`, `TypedDict`, and `dataclasses` where appropriate.
- Explicit Parameters: Use strict typing and avoid ad-hoc default values in function definitions (let the caller decide). When calling functions, pass explicit arguments and comment the rationale behind any specific values chosen.
- Error Handling: Do not catch bare or broad exceptions (e.g., `except:` or `except Exception:`); always specify concrete error types.
- Documentation & Logging: Use Google-style docstrings. Keep comments and logs concise, clear, and strictly essential.
- Third-party Libraries: Prefer established libraries over custom implementations; ask me before adding new dependencies.
