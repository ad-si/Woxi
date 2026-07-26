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

A granularity keeps only the components it names, so a `"Month"` object stands
for the whole month:

```scrut
$ wo 'DateObject[{2024, 2, 29, 13, 5, 7}, "Month"]'
DateObject[{2024, 2}, Month]
```

```scrut
$ wo 'DateWithinQ[DateObject[{2024, 1, 1}, "Year"], DateObject[{2024, 5, 1}]]'
True
```

```scrut
$ wo 'DateObject[{2024, 2, 29}]["Granularity"]'
Day
```
