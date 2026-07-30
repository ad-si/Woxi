# `Sharpen`

Sharpens an image using unsharp mask.

`Sharpen[image, r]` is `image + 2 (image - Blur[image, r])`, so a single bright
pixel comes back with a ring of negative values around it:

```scrut
$ wo 'Round[1000 ImageData[Sharpen[Image[{{0., 0., 0., 1., 0., 0., 0.}}], 1]]]'
{{0, 0, -199, 1398, -199, 0, 0}}
```

It shares `Blur`'s radius specification, including the pair form and the cap
at half the image:

```scrut
$ wo 'ImageData[Sharpen[Image[{{1., 0.}}], 10]] == ImageData[Sharpen[Image[{{1., 0.}}], 1]]'
True
```

An integer image is rounded back onto the levels it can hold, and clipped to
them:

```scrut
$ wo 'ImageData[Sharpen[Image[{{0, 0, 255, 0, 0}}, "Byte"], 1]]'
{{0., 0., 1., 0., 0.}}
```

A radius that is not a non-negative number is reported:

```scrut
$ wo 'Sharpen[Image[{{0.1, 0.2}}], -1]'

Sharpen::bdrad: The specified radius -1 should be either a non-negative number or a list of 2 non-negative numbers.
Sharpen[-Image-, -1]
```
