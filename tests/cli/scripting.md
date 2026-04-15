# Scripting & CLI

Woxi can run Wolfram Language source files (`.wls`) directly from the
command line, and exposes a few functions for shelling out to the
operating system from within a script.


## Running a `.wls` script

```scrut
$ "$TESTDIR/../scripts/hello_world.wls"
Hello World!
```


## `Run`

Runs a shell command and returns its raw exit status (on Unix this
is 256 × the real exit code when the process exits normally).

```scrut
$ wo 'Run["exit 0"]'
0
```

```scrut
$ wo 'Run["exit 1"]'
256
```
