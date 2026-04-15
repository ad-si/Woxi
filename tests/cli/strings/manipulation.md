# String Manipulation

Inserting, deleting, replacing, padding, rotating, and partitioning strings.

## `StringInsert`

Inserts a string at a given 1-based position.

```scrut
$ wo 'StringInsert["abcd", "X", 2]'
aXbcd
```




## `StringDelete`

Removes every occurrence of a substring.

```scrut
$ wo 'StringDelete["Hello world", " "]'
Helloworld
```




## `StringReplace`

Applies one or more replacement rules to a string.

```scrut
$ wo 'StringReplace["Hello world", "world" -> "moon"]'
Hello moon
```




## `StringReplaceList`

Returns every single-replacement variant of a string.

```scrut
$ wo 'StringReplaceList["abcabc", "a" -> "X"]'
{Xbcabc, abcXbc}
```




## `StringReplacePart`

Replaces a specific character range (1-indexed, inclusive) with a new string.

```scrut
$ wo 'StringReplacePart["Hello world", "XXXX", {1, 5}]'
XXXX world
```




## `StringPartition`

Splits a string into consecutive chunks of a given length.

```scrut
$ wo 'StringPartition["abcdef", 2]'
{ab, cd, ef}
```




## `StringRiffle`

Joins a list of strings, inserting a separator between each element.

```scrut
$ wo 'StringRiffle[{"a", "b", "c"}]'
a b c
```

```scrut
$ wo 'StringRiffle[{"a", "b", "c"}, ", "]'
a, b, c
```




## `StringRotateLeft`

Rotates a string to the left by `n` characters.

```scrut
$ wo 'StringRotateLeft["abcdef", 2]'
cdefab
```




## `StringRotateRight`

Rotates a string to the right by `n` characters.

```scrut
$ wo 'StringRotateRight["abcdef", 2]'
efabcd
```




## `StringTakeDrop`

Returns a list `{StringTake[s, n], StringDrop[s, n]}`.

```scrut
$ wo 'StringTakeDrop["abcdef", 3]'
{abc, def}
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




## `Capitalize`

Returns a string with its first character converted to uppercase.

```scrut
$ wo 'Capitalize["hello world"]'
Hello world
```




## `Decapitalize`

Returns a string with its first character converted to lowercase.

```scrut
$ wo 'Decapitalize["Hello World"]'
hello World
```




## `ToLowerCase`

Converts a string to lowercase.

```scrut
$ wo 'ToLowerCase["Hello World"]'
hello world
```




## `ToUpperCase`

Converts a string to uppercase.

```scrut
$ wo 'ToUpperCase["Hello World"]'
HELLO WORLD
```




## `RemoveDiacritics`

Removes accents and other combining marks from a string.

```scrut
$ wo 'RemoveDiacritics["cafe"]'
cafe
```




