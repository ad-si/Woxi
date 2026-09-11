# `Unprotect`

Removes protection from symbols.

```scrut
$ wo 'Unprotect[x]'
{}
```

A symbol can also be named by a string, and several at once by a list.

```scrut
$ wo 'Unprotect[{Style, MetaInformation}]'
{Style, MetaInformation}
```

```scrut
$ wo 'Unprotect["Style"]; Style = 5; Style'
5
```

A string may be a name pattern, matched the way `Names` matches one.

```scrut
$ wo 'aa = 1; ab = 2; ba = 3; Protect["Global`a*"]'
{aa, ab}
```
