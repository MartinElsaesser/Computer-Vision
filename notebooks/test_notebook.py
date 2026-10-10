# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.function
def build_string(string_arr: list[str], joiner: str) -> str:
	return joiner.join(string_arr)


@app.cell
def build_string():
    string = build_string(["abc", "def"], "-")
    string
    return (string,)


@app.cell
def _(string):
    from typing import reveal_type

    reveal_type(string)
    return


if __name__ == "__main__":
    app.run()
