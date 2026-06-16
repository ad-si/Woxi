# `DateObject`

Represents a calendar date.

```scrut
$ wo 'DateObject[{2026, 4, 15}][[1]]'
{2026, 4, 15}
```

An ISO date string is parsed into a date with the implied granularity.

```scrut
$ wo 'DateObject["2024-03-15"]'
DateObject[{2024, 3, 15}, Day]
```

```scrut
$ wo 'DateObject["2024-03"]'
DateObject[{2024, 3}, Month]
```
