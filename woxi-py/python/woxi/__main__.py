"""Command line interface of the woxi Python package.

Installed as the ``woxi`` console script. Mirrors the core subcommands
of the native Woxi CLI:

    woxi eval '<expression>'     Evaluate an expression (- reads stdin)
    woxi run <file.wls> [args]   Run a Wolfram Language script
    woxi repl                    Start an interactive session
    woxi <file.wls> [args]       Shorthand for `woxi run`
"""

from __future__ import annotations

import os
import sys

from woxi import _woxi

USAGE = """\
Usage: woxi <COMMAND>

Commands:
  eval <EXPRESSION>    Evaluate a Wolfram Language expression
                       (pass `-` to read the expression from stdin)
  run <FILE> [ARGS]…   Run a Wolfram Language file
  repl                 Start an interactive REPL session

Options:
  -h, --help     Print help
  -V, --version  Print version
"""


def _eval(expression: str) -> int:
    if expression == "-":
        expression = sys.stdin.read()
    _woxi.set_messages_to_stdout(True)
    try:
        print(_woxi.interpret(expression))
    except _woxi.WolframError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


def _run(path: str, args: list[str]) -> int:
    try:
        with open(path, encoding="utf-8") as file:
            content = file.read()
    except OSError as err:
        print(f"Error reading file: {err}", file=sys.stderr)
        return 1

    # Match `wolframscript -file`: diagnostics go to stdout.
    _woxi.set_messages_to_stdout(True)

    absolute_path = os.path.abspath(path)
    _woxi.set_system_variable("$InputFileName", f'"{absolute_path}"')
    _woxi.set_script_command_line([absolute_path, *args])

    if content.startswith("#!"):
        content = content.split("\n", 1)[1] if "\n" in content else ""

    try:
        # The final expression value is deliberately not printed when
        # running a script file, matching `woxi run` / wolframscript.
        _woxi.interpret(content)
    except _woxi.WolframError as err:
        print(f"Error interpreting file: {err}", file=sys.stderr)
        trace = _woxi.take_error_trace()
        if trace:
            print(trace, file=sys.stderr)
        return 1
    return 0


def _repl() -> int:
    print(f"Woxi {_woxi.__version__} — type expressions, Ctrl-D to exit")
    _woxi.set_repl_mode(True)
    count = 1
    while True:
        try:
            line = input(f"In[{count}]:= ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue
        if not line.strip():
            continue
        try:
            result = _woxi.interpret(line)
            if result != "Null":
                print(f"Out[{count}]= {result}")
        except _woxi.WolframError as err:
            print(f"Error: {err}", file=sys.stderr)
        count += 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0
    if args[0] in ("-V", "--version"):
        print(f"woxi {_woxi.__version__}")
        return 0

    command, rest = args[0], args[1:]
    if command == "eval":
        if len(rest) != 1:
            print("Usage: woxi eval <EXPRESSION>", file=sys.stderr)
            return 2
        return _eval(rest[0])
    if command == "run":
        if not rest:
            print("Usage: woxi run <FILE> [ARGS]…", file=sys.stderr)
            return 2
        return _run(rest[0], rest[1:])
    if command == "repl":
        return _repl()
    # `woxi <file.wls> [args]` shorthand, e.g. from a shebang line.
    if os.path.exists(command):
        return _run(command, rest)

    print(f"Error: unrecognized command or file: {command}", file=sys.stderr)
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
