
## Python environment

A basic environment for developing pypomp, supposing that the package from https://github.com/pypomp/pypomp is cloned at ~/git/pypomp

```
cd ~/git/pypomp
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# for documentation
pip install sphinx furo


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

## Making documentation

Docs are automatically updated at
https://pypomp.readthedocs.io
after each push to GitHub. This required setting up the OAuth app for ReadTheDocs in GitHub.

To build the docs locally,
```
cd ~/git/pypomp/docs
make html
```
or
```
sphinx-build -M html "source" "build"
```


