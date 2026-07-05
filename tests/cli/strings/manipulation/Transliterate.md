# `Transliterate`

Transliterates text in other writing scripts to plain Latin/ASCII characters.

The inputs are built with `FromCharacterCode` so that no non-ASCII bytes are
passed on the command line (`wolframscript -code` mangles raw UTF-8 CLI input).

```scrut
$ wo 'Transliterate[FromCharacterCode[{913, 955, 966, 945, 946, 951, 964, 953, 954, 972, 962}]]'
Alphabetikos
```

```scrut
$ wo 'Transliterate[FromCharacterCode[{1072, 1083, 1075, 1086, 1088, 1080, 1090, 1084}]]'
algoritm
```

```scrut
$ wo 'Transliterate[FromCharacterCode[{12375, 12435, 12400, 12375}]]'
shinbashi
```

```scrut
$ wo 'Transliterate[FromCharacterCode[{50504, 45397, 54616, 49464, 50836}]]'
annyeonghaseyo
```

```scrut
$ wo 'Transliterate[FromCharacterCode[{99, 97, 102, 233}]]'
cafe
```
