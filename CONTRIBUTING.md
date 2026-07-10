# Contributing to pypomp

Thank you for your interest in contributing to Pypomp! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code submissions.

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

#### Commits Spanning Multiple Types
If a commit contains changes spanning multiple types (e.g., a feature implementation along with associated doc updates or bug fixes), we recommend splitting them into separate commits. If splitting is not practical, use the most significant type (e.g., `feat` or `fix`) for the commit header, and use the following template in the body to document the other change types:
```
<primary-type>(<primary-scope>): <primary-description>

This commit also includes:
- <secondary-type>(<secondary-scope>): <description>
- <secondary-type>(<secondary-scope>): <description>
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

## Docstring Standards

All public-facing docstrings in `pypomp` follow the **NumPy-style** format, as parsed by [Sphinx Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html). The [NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html) is the authoritative reference. Google-style docstrings are **not** used.

### Scope

Apply the full standard to:

- Public classes and their public methods
- Public module-level functions
- Module docstrings (`"""..."""` at the top of each `.py` file)
- Attribute annotations using PEP 257 inline docstrings

Do **not** apply the full standard to underscore-prefixed (private) functions or methods. A minimal one-liner is sufficient for private helpers.

### Template

```python
def example_function(param_a, param_b=None):
    """One-line summary in imperative mood, ending with a period.

    Extended description goes here. Explain the purpose, algorithm,
    any important caveats, and how this fits into the broader API.
    Can span multiple paragraphs.

    Parameters
    ----------
    param_a : int
        Description of param_a.
    param_b : str or None, optional
        Description of param_b. Defaults to None.

    Returns
    -------
    float
        Description of the return value.

    Raises
    ------
    ValueError
        If param_a is negative. Optional section. Use judiciously, i.e.,
        only for errors that are non-obvious or have a large chance of
        getting raised.

    Notes
    -----
    Mathematical details, implementation notes, or caveats go here.
    Optional section.

    Examples
    --------
    >>> example_function(1)
    1.0
    Optional section.

    See Also
    --------
    related_function : Brief description of the related function.
    Optional section.

    References
    ----------
    .. [1] Author, "Title," Journal, year. https://doi.org/...
    Optional section.
    """
```

### Rules

1. **One-line summary**: ≤ 79 characters, imperative mood (e.g. "Compute the log-likelihood." not "Computes..."), ends with a period. Appears immediately after the opening `"""` with no blank line before it.
2. **Extended description**: Separated from the summary by a blank line. Use full sentences.
3. **Types in Parameters/Returns**: Use plain English type descriptions that match the signature — e.g. `int`, `PompParameters or None`, `jax.Array`, `list of str`. Append `, optional` for parameters with defaults. Do **not** use Sphinx roles (`:class:`, etc.) inside `Parameters` type fields — save those for the prose description.
4. **`Returns` for None-returning methods**: Methods that mutate `self` and return `None` still use a `Returns` section. Document the return value as `None` and describe what gets updated in the extended description.
5. **Attribute docstrings**: Use PEP 257 inline style — a one-liner docstring immediately after each class attribute declaration:
   ```python
   ys: pd.DataFrame
   """The measurement data frame with observation times as the index."""
   ```
6. **Examples**: Use `>>> ` doctest format. Include examples on all major user-facing entry points.
7. **Cross-references in prose**: Use Sphinx roles (`:class:`, `:meth:`, `:func:`, `:attr:`) in the extended description and section prose — e.g. `:class:`~pypomp.PompParameters``.
8. **Sections order**: Summary → Extended Description → Parameters → Returns → Raises → Notes → Examples → See Also → References.
9. **Private items**: A single-line docstring is sufficient. Full NumPy format is not required.

---

## Need Help?

If you have questions about the codebase, API design, or contribution process, please feel free to reach out to the core development team at [pypomp-org@umich.edu](mailto:pypomp-org@umich.edu) or open an issue on GitHub.
