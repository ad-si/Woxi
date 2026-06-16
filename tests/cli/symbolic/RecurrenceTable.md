# `RecurrenceTable`

Generate a table of values from a recurrence relation. The range may be given
as `{n, nmax}` (starting from 1) or `{n, nmin, nmax}`.

```scrut
$ wo 'RecurrenceTable[{a[n] == a[n-1] + a[n-2], a[1] == 1, a[2] == 1}, a, {n, 10}]'
{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}
```
