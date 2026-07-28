# `CharacterNormalize`

Converts the characters of a string to a Unicode normalization form:
`"NFD"` and `"NFKD"` decompose, `"NFC"` and `"NFKC"` decompose and then
recompose, and `"NFKCCasefold"` additionally case-folds.

`"NFD"` splits a precomposed character into its base and combining marks —
here U+00C5 (`Å`) becomes `A` followed by the combining ring U+030A, and
`"NFC"` puts it back together:

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{197}], "NFD"]]'
{65, 778}
```

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{65, 778}], "NFC"]]'
{197}
```

Canonically equivalent characters normalize to the same string, so the
angstrom sign U+212B and `Å` become indistinguishable:

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{8491}], "NFC"]]'
{197}
```

The compatibility forms also fold formatting distinctions away. The
superscript two U+00B2 is left alone by `"NFC"` but becomes a plain `2` under
`"NFKC"`:

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{178}], "NFC"]]'
{178}
```

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{178}], "NFKC"]]'
{50}
```

`"NFKCCasefold"` is the form to compare strings with when case must not
matter. It uses full case folding, so U+00DF (`ß`) expands to `ss` rather than
merely lowercasing:

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{223}], "NFKCCasefold"]]'
{115, 115}
```

```scrut
$ wo 'CharacterNormalize["ABC", "NFKCCasefold"]'
abc
```

It also drops default-ignorable characters such as the soft hyphen U+00AD:

```scrut
$ wo 'ToCharacterCode[CharacterNormalize[FromCharacterCode[{65, 173, 66}], "NFKCCasefold"]]'
{97, 98}
```

A list of strings is normalized elementwise:

```scrut
$ wo 'CharacterNormalize[{"ab", "cd"}, "NFC"]'
{ab, cd}
```
