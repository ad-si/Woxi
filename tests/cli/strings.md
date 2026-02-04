# Basic String Tests

## `Print`

```scrut
$ wo 'Print["Hello World!"]'
Hello World!
Null
```


## `StringLength`

```scrut
$ wo 'StringLength["Hello World!"]'
12
```


## `StringTake`

```scrut
$ wo 'StringTake["Hello World!", 5]'
Hello
```


## `StringDrop`

```scrut
$ wo 'StringDrop["Hello World!", 6]'
World!
```


## `StringJoin`

```scrut
$ wo 'StringJoin["Hello", " ", "World!"]'
Hello World!
```


## `StringSplit`

```scrut
$ wo 'StringSplit["Hello World!", " "]'
{Hello, World!}
```


## `StringStartsQ`

```scrut
$ wo 'StringStartsQ["Hello World!", "Hello"]'
True
```

```scrut
$ wo 'StringStartsQ["Hello World!", "Bye"]'
False
```


## `StringEndsQ`

```scrut
$ wo 'StringEndsQ["Hello World!", "World!"]'
True
```

```scrut
$ wo 'StringEndsQ["Hello World!", "Moon!"]'
False
```


## `StringPosition`

Finds all positions of a substring, returning {start, end} pairs (1-indexed).

```scrut
$ wo 'StringPosition["abcabc", "bc"]'
{{2, 3}, {5, 6}}
```

```scrut
$ wo 'StringPosition["hello", "l"]'
{{3, 3}, {4, 4}}
```


## `StringMatchQ`

Tests if string matches a pattern.
`*` matches any sequence of characters (including empty).
`@` matches one or more characters, excluding uppercase letters.

Exact match:

```scrut
$ wo 'StringMatchQ["hello", "hello"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "world"]'
False
```

```scrut
$ wo 'StringMatchQ["hello", "Hello"]'
False
```

Wildcard `*` matches any sequence:

```scrut
$ wo 'StringMatchQ["hello", "h*o"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "h*"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "*o"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "*"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "h*l*o"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "*ell*"]'
True
```

`*` can match empty sequence:

```scrut
$ wo 'StringMatchQ["hello", "hello*"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "*hello"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "hel*lo"]'
True
```

Non-matching patterns:

```scrut
$ wo 'StringMatchQ["hello", "h*x"]'
False
```

```scrut
$ wo 'StringMatchQ["hello", "x*o"]'
False
```

```scrut
$ wo 'StringMatchQ["hello", "hellooo"]'
False
```

Empty string cases:

```scrut
$ wo 'StringMatchQ["", ""]'
True
```

```scrut
$ wo 'StringMatchQ["", "*"]'
True
```

```scrut
$ wo 'StringMatchQ["", "a"]'
False
```

```scrut
$ wo 'StringMatchQ["hello", ""]'
False
```

Wildcard `@` matches one or more non-uppercase characters:

```scrut
$ wo 'StringMatchQ["hello", "h@o"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "@"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "h@"]'
True
```

```scrut
$ wo 'StringMatchQ["hello", "@o"]'
True
```

`@` requires at least one character (unlike `*`):

```scrut
$ wo 'StringMatchQ["hello", "hello@"]'
False
```

```scrut
$ wo 'StringMatchQ["", "@"]'
False
```

`@` does not match uppercase letters:

```scrut
$ wo 'StringMatchQ["hEllo", "h@o"]'
False
```

```scrut
$ wo 'StringMatchQ["HELLO", "@"]'
False
```

```scrut
$ wo 'StringMatchQ["helloWorld", "hello@"]'
False
```

`@` matches lowercase, digits, and other non-uppercase characters:

```scrut
$ wo 'StringMatchQ["hello123", "hello@"]'
True
```

```scrut
$ wo 'StringMatchQ["hello world", "hello@"]'
True
```

```scrut
$ wo 'StringMatchQ["a1b2c3", "@"]'
True
```


## `StringReverse`

Reverses the characters in a string.

```scrut
$ wo 'StringReverse["Hello"]'
olleH
```

```scrut
$ wo 'StringReverse["abcde"]'
edcba
```

```scrut
$ wo 'StringReverse[""]'

```


## `StringRepeat`

Repeats a string n times.

```scrut
$ wo 'StringRepeat["ab", 3]'
ababab
```

```scrut
$ wo 'StringRepeat["x", 5]'
xxxxx
```

```scrut
$ wo 'StringRepeat["hello", 0]'

```


## `StringTrim`

Removes leading and trailing whitespace.

```scrut
$ wo 'StringTrim["  hello  "]'
hello
```

```scrut
$ wo 'StringTrim["  hello world  "]'
hello world
```

With a pattern, removes leading and trailing occurrences:

```scrut
$ wo 'StringTrim["xxhelloxx", "xx"]'
hello
```


## `StringCases`

Finds all occurrences of a substring.

```scrut
$ wo 'StringCases["abcabc", "bc"]'
{bc, bc}
```

```scrut
$ wo 'StringCases["hello", "l"]'
{l, l}
```

```scrut
$ wo 'StringCases["hello", "x"]'
{}
```


## `ToString`

Converts an expression to a string.

```scrut
$ wo 'ToString[123]'
123
```

```scrut
$ wo 'ToString[{1, 2, 3}]'
{1, 2, 3}
```

```scrut
$ wo 'ToString[1 + 2]'
3
```


## `ToExpression`

Converts a string to an evaluated expression.

```scrut
$ wo 'ToExpression["1 + 2"]'
3
```

```scrut
$ wo 'ToExpression["Plus[3, 4]"]'
7
```


## `StringPadLeft`

Pads a string on the left to a specified length.

```scrut
$ wo 'StringPadLeft["hi", 5]'
   hi
```

```scrut
$ wo 'StringPadLeft["hi", 5, "0"]'
000hi
```

```scrut
$ wo 'StringPadLeft["hello", 3]'
llo
```


## `StringPadRight`

Pads a string on the right to a specified length.

```scrut
$ wo 'StringPadRight["hi", 5, "0"]'
hi000
```

```scrut
$ wo 'StringPadRight["hello", 3]'
hel
```
