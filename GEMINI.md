# GEMINI.md

For full details on project setup, environment configuration, linting, type checking, testing, and release procedures, please refer to DEVELOPER.md.

## Environment Setup
Always activate the virtual environment before running tests or commands:
```bash
source .venv/bin/activate
```

## Development & Testing Guidelines
Please adhere to these guidelines during agent sessions:
1. Always run type-checking with `pyright` to catch type errors before concluding tasks or code changes.
2. DO NOT preemptively run tests at the beginning of a session.
3. DO NOT run tests unless you have made edits to the code already.
4. DO NOT run tests just to check that new doc strings work.
5. You can assume that the tests work at first unless the user explicitly tells you otherwise.
6. DO NOT run the "heavy" tests; a human will do that instead.
