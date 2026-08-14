---
icon: lucide/box
---

# Contexts & Packages

A symbol's full name is its context followed by its short name, as in
`Fibo`fib`. Which symbol a short name refers to is decided when the name is
*read*: the first context on `$ContextPath` that already has a symbol of that
name wins, and when none does the symbol is created in `$Context`.

That happens per input unit — a line of a script, or one `wo` invocation — so
a context opened halfway through a line only affects the lines after it.

```scrut
$ wo '{$Context, MemberQ[$ContextPath, "System`"]}'
{Global`, True}
```


## Writing a package

`BeginPackage` opens a context and puts it on `$ContextPath`; `Begin["`Private`"]`
moves into a private sub-context, so only the symbols mentioned before it are
exported. `EndPackage` closes the package and leaves its context on the path.

```scrut
$ printf 'BeginPackage["Fibo`"]\nfib::usage = "fib[n] gives the n-th Fibonacci number.";\nBegin["`Private`"]\nfib[0] = 0;\nfib[1] = 1;\nfib[n_] := fib[n - 1] + fib[n - 2];\nEnd[]\nEndPackage[]\n' > Fibo.wl
```


## `Needs`

`Needs["context`"]` reads the file that provides a context — from a paclet
directory registered with `PacletDirectoryLoad`, or from `$Path` — unless the
context is already in `$Packages`.

```scrut
$ printf 'Needs["Fibo`"]\nfib[10]\n' > use.wls; wo 'Get["use.wls"]'
55
```

The context ends up on `$ContextPath`, which is what lets `fib` be named
without qualification:

```scrut
$ wo 'Needs["Fibo`"]; First[$ContextPath]'
Fibo`
```

```scrut
$ wo 'Needs["Fibo`"]; Names["Fibo`*"]'
{fib}
```

Reading it a second time is a no-op:

```scrut
$ wo 'Needs["Fibo`"]; MemberQ[$Packages, "Fibo`"]'
True
```

`Needs["context`", "file"]` reads the named file instead of searching for one.
A file that does not go on to create the context is reported:

```scrut
$ wo 'Needs["MyFibo`", "Fibo.wl"]'

Needs::nocont: Context MyFibo` was not created when Needs was evaluated.
Null
```


### Loading under an alias

`Needs["context`" -> "alias`"]` records the alias in `$ContextAliases` instead
of putting the context on `$ContextPath`. The alias then stands for the
context wherever its name could appear.

```scrut
$ wo 'Needs["Fibo`" -> "f`"]; $ContextAliases'
<|f` -> Fibo`|>
```

```scrut
$ printf 'Needs["Fibo`" -> "f`"]\nf`fib[10]\n' > alias.wls; wo 'Get["alias.wls"]'
55
```

```scrut
$ wo 'Needs["Fibo`" -> "f`"]; Names["f`*"]'
{Fibo`fib}
```

`$ContextPath` is left exactly as it was — the package is reachable through
the alias, not through the path:

```scrut
$ wo 'Needs["Fibo`" -> "f`"]; MemberQ[$ContextPath, "Fibo`"]'
False
```


### Loading under neither

`Needs["context`" -> None]` records nothing at all, so only the full name
reaches the package.

```scrut
$ wo 'Needs["Fibo`" -> None]; Fibo`fib[10]'
55
```

```scrut
$ wo 'Needs["Fibo`" -> None]; {MemberQ[$ContextPath, "Fibo`"], $ContextAliases}'
{False, <||>}
```


### Messages

A first argument that is neither a context nor a well-formed rule is
reported, and the call stays unevaluated. An alias has to be a context of a
single segment.

```scrut
$ wo 'Needs["Fibo"]'

Needs::cxt: Invalid context specified at position 1 in Needs[Fibo]. A context must consist of valid symbol names separated by and ending with `.
Needs[Fibo]
```

```scrut
$ wo 'Needs["Fibo`" -> "f"]'

Needs::cxru: Context or appropriately structured rule expected at position 1 in Needs[Fibo` -> f].
Needs[Fibo` -> f]
```

A context that nothing provides gives `$Failed`:

```scrut
$ wo 'Needs["NoSuchPackage`"]'

Get::noopen: Cannot open NoSuchPackage`.

Needs::nocont: Context NoSuchPackage` was not created when Needs was evaluated.
$Failed
```


## `$ContextAliases`

Aliases are an ordinary variable, so they can also be set by hand — and
`Contexts` and `Names` accept one in place of the context it stands for.

```scrut
$ printf '$ContextAliases["v`"] = "Vector`Analysis`"\nv`grad\n' > by-hand.wls; wo 'Get["by-hand.wls"]'
Vector`Analysis`grad
```

An alias whose own name is a context that already holds symbols cannot do its
job, and that is reported:

```scrut
$ printf 'taken`sym = 1\n$ContextAliases["taken`"] = "Other`"\n' > taken.wls; wo 'Get["taken.wls"]'

$ContextAliases::cxinuse: Warning: Symbols already exist in the context taken`. These symbols will not be able to be accessed while taken` is in $ContextAliases.
Other`
```
