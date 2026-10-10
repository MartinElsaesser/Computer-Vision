## UV as a package manager
UV makes it easy to manage dependencies and different python versions.
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