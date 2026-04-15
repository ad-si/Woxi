# Dates & Time

Woxi implements the most common date/time functions from the
Wolfram Language. Dates can be specified either as 3-element lists
`{year, month, day}` or as `DateObject[...]`.


## `DateString`

Returns a string representation of a date.
Without arguments, returns the current date/time.

```scrut
$ wo 'StringLength[DateString[]] > 0'
True
```

```scrut
$ wo 'DateString[{2026, 4, 15}, "ISODate"]'
2026-04-15
```

```scrut
$ wo 'DateString[{2026, 4, 15}, "Year"]'
2026
```

### Common format specifications

- `"ISODate"` → `2026-04-15`
- `"ISODateTime"` → `2026-04-15T12:34:56`
- `"DateTime"` → `Wed 15 Apr 2026 12:34:56`
- `"Date"` → `Wed 15 Apr 2026`
- `"Time"` → `12:34:56`
- `"Year"`, `"Month"`, `"Day"` → individual components


## `Now`

The current moment as a `DateObject`.

```scrut
$ wo 'Head[Now]'
DateObject
```


## `Today`

Today's date at midnight as a `DateObject`.

```scrut
$ wo 'Head[Today]'
DateObject
```


## `DateObject`

Represents a calendar date.

```scrut
$ wo 'DateObject[{2026, 4, 15}][[1]]'
{2026, 4, 15}
```


## `DayName`

Returns the day of the week for a given date.

```scrut
$ wo 'DayName[{2026, 4, 15}]'
Wednesday
```


## `LeapYearQ`

Tests whether a date falls in a leap year.

```scrut
$ wo 'LeapYearQ[{2024, 1, 1}]'
True
```


## `AbsoluteTiming`

Evaluates an expression and returns `{time_in_seconds, result}`.

```scrut
$ wo 'ListQ[AbsoluteTiming[1 + 1]]'
True
```


## `Timing`

Like `AbsoluteTiming` but returns CPU time.

```scrut
$ wo 'ListQ[Timing[1 + 1]]'
True
```
