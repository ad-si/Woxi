# `DateValue`

Extracts a named component or property of a date.

```scrut
$ wo 'DateValue[{2024, 3, 15}, "MonthName"]'
March
```

Short forms give abbreviated names.

```scrut
$ wo 'DateValue[{2024, 3, 15}, "MonthNameShort"]'
Mar
```

A list of properties returns the corresponding values.

```scrut
$ wo 'DateValue[{2024, 3, 15}, {"MonthNameShort", "DayNameShort"}]'
{Mar, Fri}
```

Time-of-day properties include the 12-hour clock and meridiem.

```scrut
$ wo 'DateValue[{2024, 3, 15, 14, 30, 0}, {"Hour12", "AMPM"}]'
{2, PM}
```
