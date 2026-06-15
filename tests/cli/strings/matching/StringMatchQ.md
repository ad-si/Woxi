# `StringMatchQ`

Tests if string matches a pattern.
`*` matches any sequence of characters (including empty).
`@` matches one or more characters, excluding uppercase letters.

The one-argument operator form applies to a string later, e.g. inside `Select`.

```scrut
$ wo 'Select[{"CAC1", "CTG1", "ACT1", "CGA1", "CTC1"}, StringMatchQ["*G*"]]'
{CTG1, CGA1}
```

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
