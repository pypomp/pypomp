# Contributing to pypomp

Thank you for your interest in contributing to **pypomp**! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code submissions.

To ensure a smooth collaboration process, please read and follow these guidelines.

---

## How Can I Contribute?

### 1. Reporting Bugs & Proposing Features
Before submitting a bug report or proposing a new feature, please check the [existing issues](https://github.com/pypomp/pypomp/issues) to see if it has already been discussed.

* **Bugs**: When reporting a bug, provide a clear description, reproduction steps, and expected behavior. Please include a minimal, reproducible example, along with the full error log.
* **Features**: When proposing a feature, explain the use case, why it would be beneficial to the project, and any proposed implementation details.

### 2. Contributing Code
We enforce an **Issue First Policy** for code changes. If you plan to make a non-trivial code contribution, please:
1. Open a new issue (or find an existing one) to discuss your proposed changes before starting development.
2. Confirm the design direction with the core maintainers.

---

## Pull Request Guidelines

To submit your code changes, follow this workflow:

1. **Fork and Clone**: Fork the [pypomp repository](https://github.com/pypomp/pypomp) and clone it locally.
2. **Setup Local Environment**: Follow the setup instructions in the [Developer Guide](DEVELOPER.md) to set up your virtual environment, packages, and tools.
3. **Create a Feature Branch**: Use a descriptive branch name for your changes (e.g., `feature/add-loglik-se` or `fix/mif-cooling-rate`).
4. **Implement & Test**:
   - Write your code changes.
   - **Write Unit Tests**: Every bug fix or new feature must be accompanied by corresponding unit tests. We aim to maintain and improve overall test coverage.
5. **Run Verification Tools**:
   - Run type checking using `pyright`.
   - Format and lint using `ruff check .` and `ruff format .`.
   - Run the test suite via `pytest`.
6. **Pass Pre-Commit Hooks**: Ensure all local pre-commit hooks pass before pushing. Pre-commit hooks run automatically on `git commit`, but you must be in an active virtual environment.
7. **Submit Pull Request**: Push your branch to your fork and submit a pull request (PR) to the `main` branch of `pypomp`.

---

## Commit Message Style

We enforce the **Conventional Commits** specification for all commit messages. This helps us generate clean changelogs and automate release processes.

Each commit message must follow the structure:
```
<type>(<scope>): <description>

[optional body]
```

### Allowed Types
* `feat`: A new feature (e.g., `feat(panel): add random walk cooling schedule`)
* `fix`: A bug fix (e.g., `fix(pfilter): resolve index out of bounds error`)
* `perf`: Performance-improving changes (e.g., `perf(jax): vectorize transition density calculation via vmap`)
* `docs`: Documentation changes (e.g., `docs: add CONTRIBUTING.md guide`)
* `style`: Code formatting changes (e.g., `style: run ruff format`)
* `refactor`: Code restructuring that neither fixes a bug nor adds a feature
* `test`: Adding or correcting tests (e.g., `test(mif): add test coverage for learning rates`)
* `chore`: Maintenance tasks, dependencies, or tool configurations (e.g., `chore: update github action python version`)

---

## Need Help?

If you have questions about the codebase, API design, or contribution process, please feel free to reach out to the core development team at [pypomp-org@umich.edu](mailto:pypomp-org@umich.edu) or open an issue on GitHub.
