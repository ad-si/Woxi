# `ToCharacterCode`

Converts a string to its list of Unicode character codes.

```scrut
$ wo 'ToCharacterCode["abc"]'
{97, 98, 99}
```

A named character contributes the code point Wolfram stores it at, which for
its typeset operators, its script alphabet and several pictographs is a
private-use character rather than the standard look-alike — `\[WarningSign]`
is U+F725 (63269), not U+26A0 (9888).

```scrut
$ wo 'ToCharacterCode["\[Alpha]\[WarningSign]\[ScriptCapitalD]"]'
{945, 63269, 63347}
```

A name Wolfram has no character for is reported by the reader and left as
written, so nothing is invented: `\[Tab]` is six characters, while
`\[RawTab]` is the tab.

```scrut
$ wo 'ToCharacterCode[{"\[Tab]", "\[RawTab]"}]'

Syntax::sntufn: Unknown unicode longname Tab.
{{92, 91, 84, 97, 98, 93}, {9}}
```
