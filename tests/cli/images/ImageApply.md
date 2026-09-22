# `ImageApply`

Applies a function to each pixel of an image.

```scrut
$ wo 'ImageData[ImageApply[1 - # &, Image[{{0.25, 0.5}, {0.75, 0.}}]]]'
{{0.75, 0.5}, {0.25, 1.}}
```

`Masking -> mask` restricts the function to pixels the mask marks
positive; every other pixel passes through unchanged:

```scrut
$ wo 'ImageData[ImageApply[1 - # &, Image[{{0.25, 0.5}, {0.75, 0.}}], Masking -> Image[{{1, 0}, {0, 1}}]]]'
{{0.75, 0.5}, {0.75, 1.}}
```
