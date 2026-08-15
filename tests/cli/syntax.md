---
icon: lucide/braces
---

# Syntax


## Postfix application (`//`)

`expr // f` is equivalent to `f[expr]`.
The `//` operator has the lowest precedence, so it applies to the entire
left-hand expression.

```scrut
$ wo '4 // Sqrt'
2
```

```scrut
$ wo '{1, 2, 3} // Length'
3
```

Because it really is `f[expr]`, a `Sequence` on the left spreads into `f`'s
arguments rather than arriving as one:

```scrut
$ wo 'Sequence[1, 2] // f'
f[1, 2]
```

A `Sequence` one level down has already been spread by the head holding it,
so `f` still sees a single argument:

```scrut
$ wo '{Sequence[1, 2]} // Length'
2
```

Postfix after an operator chain:

```scrut
$ wo '1 + 2 // ToString'
3
```

```scrut
$ wo 'Sqrt /@ {1, 4, 9} // Length'
3
```

Postfix with a curried function call:

```scrut
$ wo '{1, 4, 9} // Map[Sqrt]'
{1, 2, 3}
```

Chained postfix:

```scrut
$ wo '16 // Sqrt // Sqrt'
2
```


## Prefix application (`@`) with comparison operators

`@` has higher precedence than `==`, so `f@x == y` means `(f@x) == y`.

```scrut
$ wo 'Length@{1,2,3} == 3'
True
```

```scrut
$ wo 'Length@Union@{1,2,1} == Length@Union[{1,2,1} + Range@3] == 3'
False
```


## Rule (`->`) and RuleDelayed (`:>`) as general operators

`->` creates a rule, and `:>` creates a delayed rule.
These can be used anywhere in expressions, not just in replacement contexts.

```scrut
$ wo '{1, 2} -> 3'
{1, 2} -> 3
```

```scrut
$ wo 'Frame -> All'
Frame -> All
```

```scrut
$ wo 'x :> x + 1'
x :> x + 1
```


## Multi-index Part extraction (`[[i, j]]`)

`expr[[i, j]]` extracts nested parts: first part `i`, then part `j` from the result.

```scrut
$ wo 'x = {{a,b},{c,d}}; x[[2,1]]'
c
```

```scrut
$ wo 'x = {{a,b},{c,d}}; x[[1,2]]'
b
```

```scrut {output_stream: combined}
$ wo 'FullForm[a[[1,2,3]]]'

Part::partd: Part specification a[[1,2,3]] is longer than depth of object.
.* (regex*)
FullForm[a[[1,2,3]]]
```


## Spaces inside the Part brackets (`x[ [i] ]`)

The two brackets of `[[…]]` are separate tokens, so whitespace, newlines
and comments may sit between them — a lone `[…]` can never start an
expression, so `x[ [i] ]` is unambiguously `Part[x, i]`.

```scrut
$ wo 'myList = {10, 20, 30}; myList[ [ {1, 2} ] ]'
{10, 20}
```

```scrut
$ wo 'ToString[Hold[a[ (* index *) [1] ]], InputForm]'
Hold[a[[1]]]
```


## Part of an operator expression (`(a /. b)[[i]]`)

`[[…]]` binds tighter than every infix operator, so a parenthesized
operator expression keeps its parentheses when echoed — dropping them
would change what re-parsing produces.

```scrut
$ wo '({9, 8} /. 8 -> 5)[[1]]'
9
```

```scrut
$ wo 'ToString[Hold[(a /. b :> c)[[1]]], InputForm]'
Hold[(a /. b :> c)[[1]]]
```


## Increment (`++`) and Decrement (`--`)

`x++` increments `x` by 1 and returns the old value.
`x--` decrements `x` by 1 and returns the old value.

```scrut
$ wo 'x = 5; x++; x'
6
```

```scrut
$ wo 'x = 10; x--; x'
9
```

```scrut
$ wo 'x = 0; Do[x++, {i, 1, 5}]; x'
5
```


## Part Assignment

Assigning to a Part expression modifies the list in-place.

```scrut
$ wo 'x = {1, 2, 3}; x[[2]] = 99; x'
{1, 99, 3}
```

```scrut
$ wo 'x = {{1, 2}, {3, 4}}; x[[2, 1]] = 99; x'
{{1, 2}, {99, 4}}
```

```scrut
$ wo 'x = {a, b, c}; x[[3]] = z; x'
{a, b, z}
```


## Rule Pattern Evaluation

Both sides of `->` (Rule) are evaluated.

```scrut
$ wo '{2, First@{1}} -> "Q"'
{2, 1} -> Q
```


## Stored Anonymous Functions

Anonymous functions can be stored in variables and called later.

```scrut
$ wo 'f = (# + 1) &; f[5]'
6
```

```scrut
$ wo 'g = (# * #2) &; g[3, 4]'
12
```

```scrut
$ wo 'f = # &; Map[f, {1, 2, 3}]'
{1, 2, 3}
```


## Nested Function Slot Scoping

Slots (`#`) bind to the innermost `&` (Function).

```scrut
$ wo '(# + 1 &) /@ {10, 20, 30}'
{11, 21, 31}
```


## Alternatives (`|`)

The `|` operator represents alternatives in pattern matching.

```scrut
$ wo 'Cases[{1, "a", 2, "b"}, _Integer]'
{1, 2}
```

```scrut
$ wo 'Cases[{1, 2, 3, 4, 5}, Except[2 | 4]]'
{1, 3, 5}
```


## Comments (`(* … *)`)

A comment is skipped wherever whitespace is, and what it holds is never
read as code — not even a bracket, a quote or a semicolon left over from
commented-out code:

```scrut
$ wo 'Hold[f[a (* endof For[ i *); b]]'
Hold[f[a; b]]
```

```scrut
$ wo '{1 (* a ) b *); 2}'
{2}
```

```scrut
$ wo 'Range[5][[2 (* [ *) ;; 3]]'
{2, 3}
```

Comments nest, so a commented-out block that itself holds comments is
skipped whole:

```scrut
$ wo '1 + (* outer (* inner *) still a comment *) 2'
3
```
