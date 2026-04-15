# String Basics

Length, indexing, joining, splitting, and other elementary string operations.

## `StringLength`

```scrut
$ wo 'StringLength["Hello World!"]'
12
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




## `StringPart`

Extracts a character by 1-based index.

```scrut
$ wo 'StringPart["Hello", 2]'
e
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




## `Characters`

Splits a string into a list of single-character strings.

```scrut
$ wo 'Characters["abc"]'
{a, b, c}
```




## `CharacterRange`

Generates a list of consecutive characters from a start to an end character.

```scrut
$ wo 'CharacterRange["a", "e"]'
{a, b, c, d, e}
```




## `Print`

```scrut
$ wo 'Print["Hello World!"]'
Hello World!
Null
```




