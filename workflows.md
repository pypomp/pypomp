

## Formatting and linting

It is recommended to use [ruff](https://docs.astral.sh/ruff/). For example,
```
pip install ruff
cd ~/git/pypomp
ruff check
ruff format
```

## Testing

A test workflow is described in pypomp/test/README

## Pushing edits to main

* The unit tests are run by a GitHub Action, and a flag is raised if these tests fail.

* New code should be unit tested and usually should not break existing unit tests.

* If a previous unit test is broken, then the new code is not backward compatible and a discussion with the entire development team may be appropriate. If the change is desirable, the previous unit test should be fixed accordingly.

## Pushing to PyPI

This is the workflow used on 25-07-22 for v0.1.4. At this time, AJA and ELI are able to push to PyPI. 

```
cd ~/git/pypomp
emacs pyproject.toml # to update pypomp version 
pip install .     # to update local install
cd test
pytest .
cd ..
python -m build
python -m twine upload --repository pypi dist/*
```

To set up push permission, you need to get an API token from PyPI and add it to `$HOME/.pypirc`.

Note that you can test things out on the PyPI test server, at https://test.pypi.org/

