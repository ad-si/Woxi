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

A date is itself a date specification, so a second `DateObject` re-tags it —
`DateObject[Now, "Hour"]` is the hour `Now` falls in, the key a package
caching once an hour keys its entries on:

```scrut
$ wo 'DateObject[DateObject[{2026, 9, 8, 10, 14, 50.5}], "Hour"]'
DateObject[{2026, 9, 8, 10}, Hour, Gregorian, 0.]
```

`"Second"` drops the fraction that `"Instant"` keeps:

```scrut
$ wo 'DateObject[DateObject[{2026, 9, 8, 10, 14, 50.5}], "Second"]'
DateObject[{2026, 9, 8, 10, 14, 50}, Second, Gregorian, 0.]
```

Refining a coarse date pads the missing components; a date that carries no
offset of its own gets `None`:

```scrut
$ wo 'DateObject[DateObject[{2026, 9}], "Hour"]'
DateObject[{2026, 9, 1, 0}, Hour, Gregorian, None]
```
