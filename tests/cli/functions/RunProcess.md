# `RunProcess`

Runs a program to completion and reports its exit code and output.
A string names the program; a list gives the program and its arguments.
The program is started directly, not through a shell.

```scrut
$ wo 'RunProcess[{"echo", "hello"}]'
<|ExitCode -> 0, StandardOutput -> hello
, StandardError -> |>
```

A property name picks one of the results:

```scrut
$ wo 'RunProcess[{"sh", "-c", "exit 3"}, "ExitCode"]'
3
```

```scrut
$ wo 'RunProcess[{"sh", "-c", "echo oops >&2"}, "StandardError"]'
oops

```

A third argument is fed to the program's standard input:

```scrut
$ wo 'RunProcess[{"tr", "a-z", "A-Z"}, "StandardOutput", "shout"]'
SHOUT
```

`ProcessDirectory` runs the program in a directory, and `ProcessEnvironment`
gives it its environment variables:

```scrut
$ wo 'RunProcess[{"sh", "-c", "pwd"}, "StandardOutput", ProcessDirectory -> "/"]'
/

```

```scrut
$ wo 'RunProcess[{"sh", "-c", "echo $GREETING"}, "StandardOutput", ProcessEnvironment -> <|"GREETING" -> "hi"|>]'
hi

```

A program that cannot be found gives `$Failed`:

```scrut
$ wo 'RunProcess["no-such-program-anywhere"]'

RunProcess::pnfd: Program no-such-program-anywhere not found. Check the path and file permissions.
$Failed
```
