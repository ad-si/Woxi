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
