# Intro to the boilerplate 

## UV as a package manager
[UV](https://docs.astral.sh/uv/) makes it easy to manage dependencies and different python versions.
It automatically creates a virtual environment to prevent packages from being installed globally.

The necessary steps to setup and use UV are outlined in the following sections.

### Install UV
To install UV follow its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

### Install packages via UV
To install the right packages and python version just run:
```bash 
uv sync
```
UV will automatically setup your project and create a virtual environment.

### Running a python file with UV
To run a python file exchange `python` with `uv run`  
Instead of:
```bash
python somePythonFile.py
```
Use:
```bash
uv run somePythonFile.py
```

## Pyrefly for type checking

https://pyrefly.org/

### Run pyrefly
```bash 
uv run pyrefly check
```

### Documentation on typing
- [quick start into typing python code from pyrefly](https://pyrefly.org/en/docs/typing-for-python-developers/)
- [python standard types](https://docs.python.org/3/builtins/stdtypes.html)
- [python typing package](https://docs.python.org/3/library/typing.html)

## Marimo as a jupyter notebook alternative

https://docs.marimo.io/

### create a notebook

you can either create a marimo notebook from within the terminal
```bash 
uv run marimo edit
```

or via the vscode extension `Ctl+Shift+P` `>Create: New marimo notebook`