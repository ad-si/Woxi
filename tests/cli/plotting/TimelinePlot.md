# `TimelinePlot`

Plots dates and date intervals as events along a horizontal timeline.

Each date becomes a point marker; each `DateInterval` becomes a bar. Events
are automatically stacked into rows so their labels don't overlap, and a date
axis is drawn along the bottom.

```scrut
$ wo 'Head[TimelinePlot[{DateObject[{2000}], DateObject[{2005}], DateObject[{2010}]}]]'
Graphics
```

Date lists (`{y, m, d}`) are accepted directly.

```scrut
$ wo 'Head[TimelinePlot[{{2001, 1, 1}, {2003, 6, 15}}]]'
Graphics
```

Events can be labeled with `Labeled`, `label -> date` rules, or an
association keyed by label.

```scrut
$ wo 'Head[TimelinePlot[{Labeled[DateObject[{2000}], "Start"], Labeled[DateObject[{2010}], "End"]}]]'
Graphics
```

```scrut
$ wo 'Head[TimelinePlot[<|"One" -> DateObject[{2000}], "Two" -> DateObject[{2004}]|>]]'
Graphics
```

A `DateInterval` is drawn as a bar spanning its start and end dates.

```scrut
$ wo 'Head[TimelinePlot[{DateInterval[{DateObject[{2000}], DateObject[{2005}]}]}]]'
Graphics
```

### Options

- **`ImageSize`** — overall width in pixels (height grows with the number of
  stacked rows).
