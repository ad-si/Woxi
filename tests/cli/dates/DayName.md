# `DayName`

Returns the day of the week for a given date.

```scrut
$ wo 'DayName[{2026, 4, 15}]'
Wednesday
```

Any date specification works, including partial date lists and date strings.

```scrut
$ wo 'DayName[{2024, 2}]'
Thursday
```

```scrut
$ wo 'DayName["Feb 1 2024"]'
Thursday
```

A bare number is an absolute time in seconds since the start of 1900,
not a year.

```scrut
$ wo 'DayName[3155673600]'
Saturday
```
