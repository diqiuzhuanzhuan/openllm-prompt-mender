"""Console script for openllm_prompt_mender."""

import typer
from rich.console import Console

from openllm_prompt_mender import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for openllm_prompt_mender."""
    console.print("Replace this message by putting your code into "
               "openllm_prompt_mender.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
