# `CharacterName`

Gives the Wolfram Language name of a character.

```scrut
$ wo 'CharacterName["a"]'
LatinSmallLetterA
```

Digits and ASCII punctuation have their own names.

```scrut
$ wo 'CharacterName["1"]'
DigitOne
```

```scrut
$ wo 'CharacterName["+"]'
RawPlus
```

A multi-character string returns the list of names.

```scrut
$ wo 'CharacterName["ab"]'
{LatinSmallLetterA, LatinSmallLetterB}
```
