# `DateString`

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

The `"Hour24"` element is the 2-digit 24-hour clock:

```scrut
$ wo 'DateString[{2026, 4, 15, 14, 5}, {"Hour24", ":", "Minute"}]'
14:05
```

### Common format specifications

- `"ISODate"` → `2026-04-15`
- `"ISODateTime"` → `2026-04-15T12:34:56`
- `"DateTime"` → `Wed 15 Apr 2026 12:34:56`
- `"Date"` → `Wed 15 Apr 2026`
- `"Time"` → `12:34:56`
- `"Year"`, `"Month"`, `"Day"` → individual components

The ISO week date names the week a day belongs to, which near New Year can be
in the neighbouring year:

```scrut
$ wo 'DateString[{2024, 12, 31}, "ISOWeekDate"]'
2025-W01-2
```

```scrut
$ wo 'DateString[{2024, 12, 31}, {"QuarterNameShort", " ", "MonthNameInitial"}]'
Q4 D
```
