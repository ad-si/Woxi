# Conversion and Encoding

Converting between strings and other data types, character codes, and encoded forms.

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




## `TextString`

Converts an expression to a plain text string (similar to `ToString` but
aimed at human-readable output).

```scrut
$ wo 'TextString[3.14]'
3.14
```




## `TextWords`

Splits a string into words, ignoring punctuation.

```scrut
$ wo 'TextWords["Hello, world! How are you?"]'
{Hello, world, How, are, you}
```




## `IntegerString`

Converts an integer to its string representation, optionally in a given base.

```scrut
$ wo 'IntegerString[255]'
255
```

```scrut
$ wo 'IntegerString[255, 16]'
ff
```




## `NumericalSort`

Sorts a list of strings in canonical order.

```scrut
$ wo 'NumericalSort[{"file10", "file2", "file1"}]'
{file1, file10, file2}
```




## `ToCharacterCode`

Converts a string to its list of Unicode character codes.

```scrut
$ wo 'ToCharacterCode["abc"]'
{97, 98, 99}
```




## `FromCharacterCode`

Converts a character code (or list of them) into a string.

```scrut
$ wo 'FromCharacterCode[72]'
H
```

```scrut
$ wo 'FromCharacterCode[{72, 105}]'
Hi
```




## `StringToByteArray`

Encodes a string as a byte array using UTF-8.

```scrut
$ wo 'StringToByteArray["abc"]'
ByteArray[<3>]
```




## `Alphabet`

Returns the English alphabet as a list of lowercase letters.

```scrut
$ wo 'Length[Alphabet[]]'
26
```

```scrut
$ wo 'Alphabet[][[1]]'
a
```




## `AlphabeticSort`

Sorts a list of strings alphabetically.

```scrut
$ wo 'AlphabeticSort[{"banana", "apple", "cherry"}]'
{apple, banana, cherry}
```




## `LetterNumber`

Returns the position of a letter in the alphabet.

```scrut
$ wo 'LetterNumber["c"]'
3
```




## `FromLetterNumber`

Returns the letter at position `n` in the alphabet.

```scrut
$ wo 'FromLetterNumber[5]'
e
```




## `CharacterCounts`

Returns an association mapping each character to its count.

```scrut
$ wo 'CharacterCounts["banana"]'
<|a -> 3, n -> 2, b -> 1|>
```




## `LetterCounts`

Returns an association mapping each letter to its number of occurrences.

```scrut
$ wo 'LetterCounts["banana"]'
<|a -> 3, n -> 2, b -> 1|>
```




## `WordCount`

Counts the words in a string.

```scrut
$ wo 'WordCount["Hello world how are you"]'
5
```




## `WordCounts`

Returns an association mapping each word to its number of occurrences.

```scrut
$ wo 'WordCounts["the cat and the dog"]'
<|the -> 2, dog -> 1, and -> 1, cat -> 1|>
```


## `Hash`

Computes a hash of an expression.
The one-argument form uses an implementation-defined hash whose numeric
value differs between Woxi and Mathematica, so prefer an explicit algorithm:

```scrut
$ wo 'Hash["hello", "MD5", "HexString"]'
5d41402abc4b2a76b9719d911017c592
```




## `Uncompress`

Inverse of `Compress` — decodes a base-64–encoded compressed
expression back to its original form.

```scrut
$ wo 'Uncompress[Compress["hello"]]'
hello
```




## `URLDecode`

Decodes a URL-percent-encoded string.

```scrut
$ wo 'URLDecode["hello%20world"]'
hello world
```




## `URLEncode`

Percent-encodes a string for safe inclusion in a URL.

```scrut
$ wo 'URLEncode["hello world"]'
hello%20world
```




