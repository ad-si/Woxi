# `LightDarkSwitched`

Represents an expression that switches between a light-mode and a dark-mode
variant depending on the current appearance.

`LightDarkSwitched[light, dark]` uses `light` in light mode and `dark` in
dark mode. With a single argument (or `Automatic`), the missing variant is
derived automatically. The kernel does not resolve the switch — it stays
symbolic and is resolved by the front end when rendering:

```scrut
$ wo 'LightDarkSwitched[Red, Blue]'
LightDarkSwitched[RGBColor[1, 0, 0], RGBColor[0, 0, 1]]
```

```scrut
$ wo 'LightDarkSwitched[Red]'
LightDarkSwitched[RGBColor[1, 0, 0]]
```

Since the kernel keeps it symbolic, it is not a color object:

```scrut
$ wo 'ColorQ[LightDarkSwitched[Red, Blue]]'
False
```
