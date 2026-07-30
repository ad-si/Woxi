# `OptionValue`

Gives the value of a named option.

With a head, the option is read out of `Options[head]`:

```scrut
$ wo 'OptionValue[Plot, Axes]'
True
```

```scrut
$ wo 'OptionValue[Plot, PlotRange]'
{Full, Automatic}
```

A string names the same option as the symbol does, and a list of names gives a
list of values:

```scrut
$ wo 'OptionValue[Plot, "Axes"]'
True
```

```scrut
$ wo 'OptionValue[Plot, {Axes, Frame}]'
{True, False}
```

An explicit rule list in the middle overrides the defaults name by name:

```scrut
$ wo 'OptionValue[Plot, {Axes -> 7}, {Axes, Frame}]'
{7, False}
```

A rule list in place of the head is itself the option list:

```scrut
$ wo 'OptionValue[{Frame -> True}, Frame]'
True
```

A fourth argument wraps each value in that head:

```scrut
$ wo 'OptionValue[Plot, {}, Axes, Hold]'
Hold[True]
```

The same works for a user-defined option list:

```scrut
$ wo 'Options[MySetting] = {"foo" -> 5, "bar" -> 6}; OptionValue[MySetting, "bar"]'
6
```

An option that is not among the defaults is reported and gives back the name:

```scrut
$ wo 'OptionValue[MySetting, bar]'

OptionValue::optnf: Option name bar not found in defaults for MySetting.
bar
```
