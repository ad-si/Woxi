# Matching and Searching

Pattern matching, substring search, case-folded predicates, and distance metrics.

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




## `StringContainsQ`

Tests whether a substring occurs in a string.

```scrut
$ wo 'StringContainsQ["Hello world", "world"]'
True
```

```scrut
$ wo 'StringContainsQ["Hello world", "planet"]'
False
```




## `StringFreeQ`

Tests whether a string does NOT contain a substring.

```scrut
$ wo 'StringFreeQ["Hello world", "moon"]'
True
```

```scrut
$ wo 'StringFreeQ["Hello world", "world"]'
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




## `StringCount`

Counts the occurrences of a substring.

```scrut
$ wo 'StringCount["abracadabra", "a"]'
5
```




## `EditDistance`

Computes the Levenshtein edit distance between two strings.

```scrut
$ wo 'EditDistance["kitten", "sitting"]'
3
```




## `HammingDistance`

Counts the positions at which two equal-length strings differ.

```scrut
$ wo 'HammingDistance["karolin", "kathrin"]'
3
```




## `LongestCommonSubsequence`

Returns the longest subsequence common to two strings.

```scrut
$ wo 'LongestCommonSubsequence["abcde", "xbxdy"]'
b
```




## `SequenceAlignment`

Aligns two strings, returning a list where common substrings appear
as strings and differences appear as length-2 lists of alternatives.

```scrut
$ wo 'SequenceAlignment["hello", "helloo"]'
{hell, {, o}, o}
```




## `LowerCaseQ`

Tests whether a string contains only lowercase characters.

```scrut
$ wo 'LowerCaseQ["abc"]'
True
```

```scrut
$ wo 'LowerCaseQ["ABC"]'
False
```




## `UpperCaseQ`

Tests whether a string contains only uppercase characters.

```scrut
$ wo 'UpperCaseQ["ABC"]'
True
```

```scrut
$ wo 'UpperCaseQ["Abc"]'
False
```




## `DictionaryWordQ`

Tests whether a string is a known English dictionary word.

```scrut
$ wo 'DictionaryWordQ["hello"]'
True
```




## `SyntaxQ`

Tests whether a string is a syntactically valid Wolfram Language expression.

```scrut
$ wo 'SyntaxQ["1+2"]'
True
```

```scrut
$ wo 'SyntaxQ["1+"]'
False
```




