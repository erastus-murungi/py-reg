from typing import IO

import click

from reg.utils import RegexFlag


@click.command(name="re", help="Regular expression learning tool")
@click.argument("pattern", type=click.STRING)
@click.argument("text", type=click.STRING)
@click.option("--out", "-o", type=click.File(), default=None, help="Output of the file")
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["VM", "B", "NB"]),
    default="NB",
    help="Output of the file",
)
@click.option(
    "--optimize",
    "-p",
    is_flag=True,
    show_default=True,
    default=True,
    help="Enable all optimizations",
)
@click.option(
    "--multiline",
    "-m",
    is_flag=True,
    show_default=True,
    default=False,
    help="Turn Multiline On",
)
@click.option(
    "--ignorecase",
    "-i",
    is_flag=True,
    show_default=True,
    default=False,
    help="Ignore case",
)
@click.option(
    "--dotall",
    "-d",
    is_flag=True,
    show_default=True,
    default=False,
    help="Turn on dotall mode",
)
def entry(
    pattern: str,
    text: str,
    out: IO,
    engine: str,
    optimize: bool,
    multiline: bool,
    ignorecase: bool,
    dotall: bool,
):
    flags = RegexFlag.NOFLAG


if __name__ == "__main__":
    entry()
