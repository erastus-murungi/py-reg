import json
import sys
from typing import IO

import click

from reg.fsm import NFA
from reg.pike_vm import RegexPikeVM
from reg.utils import RegexFlag


@click.command(name="re", help="Regular expression learning tool")
@click.argument("pattern", type=click.STRING)
@click.option("--text", type=click.STRING, help="text to search pattern")
@click.option("--input-file", type=click.File(), default=None, help="Input file")
@click.option(
    "--out", "-o", type=click.File("w"), default=sys.stdout, help="Output of the file"
)
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
@click.option(
    "--debug",
    "-g",
    is_flag=True,
    show_default=True,
    default=False,
    help="Turn on debug mode",
)
def entry(
    pattern: str,
    text: str,
    input_file: IO,
    out: IO,
    engine: str,
    optimize: bool,
    multiline: bool,
    ignorecase: bool,
    dotall: bool,
    debug: bool,
):
    if input_file is not None:
        text = input_file.read()
    assert text

    flags = RegexFlag.NOFLAG
    if dotall:
        flags |= RegexFlag.DOTALL
    if optimize:
        flags |= RegexFlag.OPTIMIZE
    if multiline:
        flags |= RegexFlag.MULTILINE
    if ignorecase:
        flags |= RegexFlag.IGNORECASE
    if engine == "NB":
        flags |= RegexFlag.NO_BACKTRACK
    if debug:
        flags |= RegexFlag.DEBUG

    compiled_pattern = (
        RegexPikeVM(pattern, flags) if engine == "VM" else NFA(pattern, flags)
    )
    with out:
        results = {
            index: {"span": m.span, "match": m.group(0), "groups": m.groups()}
            for index, m in enumerate(compiled_pattern.finditer(text))
        }
        out.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    entry()
