# `MidDate`

Gives the date at the midpoint of a date interval or a list of dates.

```scrut
$ wo 'MidDate[{DateObject[{2024, 10, 1}], DateObject[{2024, 10, 3}]}, "Day"]'
DateObject[{2024, 10, 2}, Day]
```

The middle of a coarse date at a finer granularity:

```scrut
$ wo 'MidDate[DateObject[{2024}], "Day"]'
DateObject[{2024, 7, 2}, Day]
```

```scrut
$ wo 'MidDate[{DateObject[{2024, 2}], DateObject[{2024, 4}]}, "Month"]'
DateObject[{2024, 3}, Month]
```
