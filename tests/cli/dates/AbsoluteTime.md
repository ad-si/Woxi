# `AbsoluteTime`

Total seconds since January 1 1900.

```scrut
$ wo 'AbsoluteTime[{2000}]'
3155673600
```

A `DateObject` is accepted as well.

```scrut
$ wo 'AbsoluteTime[DateObject[{2020, 3, 5}]]'
3792355200
```

Months outside 1 to 12 roll the year over.

```scrut
$ wo 'DateList[AbsoluteTime[{2020, 0, 1}]]'
{2019, 12, 1, 0, 0, 0.}
```
