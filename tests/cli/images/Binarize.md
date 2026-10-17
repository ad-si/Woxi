# `Binarize`

Converts an image to black and white using a threshold.

Select the threshold with a `Method`:

```scrut
$ woxi eval 'ImageData[Binarize[Image[{{0.1, 0.2, 0.3, 0.9}}], Method -> "Mean"]]'
{{0, 0, 0, 1}}
```

A function decides, per pixel, whether it becomes white:

```scrut
$ woxi eval 'ImageData[Binarize[Image[{{0.1, 0.4, 0.9}}], # > 0.3 &]]'
{{0, 1, 1}}
```
