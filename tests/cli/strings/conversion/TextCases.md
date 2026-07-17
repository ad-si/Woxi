# `TextCases`

Extracts text units of a given type from a string. With `"Word"` it returns
the words, matching `TextWords`.

```scrut
$ wo 'TextCases["The cat sat on the mat.", "Word"]'
{The, cat, sat, on, the, mat}
```

With `"Sentence"` it returns the sentences, matching `TextSentences`.

```scrut
$ wo 'TextCases["Hello world. How are you? Fine.", "Sentence"]'
{Hello world., How are you?, Fine.}
```

An optional third argument keeps only the first n results.

```scrut
$ wo 'TextCases["one two three four", "Word", 2]'
{one, two}
```
