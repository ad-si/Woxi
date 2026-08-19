---
icon: lucide/package
---

# Module

`Module` creates a local scope for variables, ensuring they don't interfere
with variables in the surrounding context.


## Basic Local Variable

```scrut
$ wo 'Module[{x = 5}, x + 2]'
7
```

```scrut
$ wo 'Module[{x = 10}, x * 3]'
30
```


## Multiple Local Variables

```scrut
$ wo 'Module[{x = 2, y = 3}, x + y]'
5
```

```scrut
$ wo 'Module[{a = 1, b = 2, c = 3}, a + b + c]'
6
```


## Local Variables Don't Leak to Outer Scope

```scrut
$ wo 'Module[{x = 100}, x]; x'
x
```

```scrut
$ wo 'y = 5; Module[{y = 10}, y]; y'
5
```


## Local Variables Shadow Outer Variables

```scrut
$ wo 'x = 1; Module[{x = 2}, x]'
2
```

```scrut
$ wo 'a = 100; Module[{a = 1}, a + 1]'
2
```


## Using Outer Variables in Local Initialization

```scrut
$ wo 'x = 5; Module[{y = x + 1}, y]'
6
```

```scrut
$ wo 'n = 10; Module[{doubled = n * 2}, doubled + 1]'
21
```


## Uninitialized Local Variables

Local variables without initialization are treated as unique symbols.

```scrut
$ wo 'Module[{x}, x]'
x\$\d+ (regex)
```

```scrut
$ wo 'Module[{x, y}, x + y]'
x\$\d+ \+ y\$\d+ (regex)
```


## Nested Module

```scrut
$ wo 'Module[{x = 1}, Module[{y = 2}, x + y]]'
3
```

```scrut
$ wo 'Module[{x = 1}, Module[{x = 2}, x]]'
2
```


## Module in Function Definition

```scrut
$ wo 'plusTwo[num_] := Module[{x = num}, x + 2]; plusTwo[3]'
5
```

```scrut
$ wo 'swap[a_, b_] := Module[{temp = a}, {b, temp}]; swap[1, 2]'
{2, 1}
```


## Module with Computations in Body

```scrut
$ wo 'Module[{x = 2}, x = x + 1; x]'
3
```

```scrut
$ wo 'Module[{sum = 0}, sum = sum + 1; sum = sum + 2; sum]'
3
```


## Module Returning Complex Expressions

```scrut
$ wo 'Module[{x = 2, y = 3}, {x, y, x + y, x * y}]'
{2, 3, 5, 6}
```

```scrut
$ wo 'Module[{x = Pi/2}, Sin[x]]'
1
```


## Module with Symbolic Computation

```scrut
$ wo 'Module[{expr = x^2}, D[expr, x]]'
2*x
```


## Module with Condition in Body

Conditions (`/;`) inside Module bodies are evaluated while local variables
are still in scope.

```scrut
$ wo 'Foo[u_, x_Symbol] := Module[{lst = u}, 3 /; lst == 1]; {Foo[1, x], Foo[x, x]}'
{3, Foo[x, x]}
```

```scrut
$ wo 'f[n_] := Module[{v = n}, "small" /; v < 10]; f[5]'
small
```


## Malformed Local Variable Specifications

Every local must be a symbol or an assignment to a symbol.
A pattern variable is substituted into the local list as well,
so a parameter cannot be re-declared as a local.

```scrut {output_stream: combined}
$ wo 'f[x_] := Module[{x}, x]; f[5]'

Module::lvsym: Local variable specification {5} contains 5, which is not a symbol or an assignment to a symbol.
Module[{5}, 5]
```

```scrut {output_stream: combined}
$ wo 'Module[{x[1] = 3}, 4]'

Module::lvset: Local variable specification {x[1] = 3} contains x[1] = 3, which is an assignment to x[1]; only assignments to symbols are allowed.
Module[{x[1] = 3}, 4]
```

```scrut {output_stream: combined}
$ wo 'Module[{x, x}, 3]'

Module::dup: Duplicate local variable x found in local variable specification {x, x}.
Module[{x, x}, 3]
```

`With` differs in that it needs a value for every local:

```scrut {output_stream: combined}
$ wo 'With[{x}, x]'

With::lvws: Variable x in local variable specification {x} requires a value.
With[{x}, x]
```


## Delayed Local Assignment

`x := v` keeps the right-hand side unevaluated until the body reads `x`.

```scrut
$ wo 'Module[{x := 1 + 1}, x]'
2
```
