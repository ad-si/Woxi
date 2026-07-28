# `ImageFilter`

`ImageFilter[f, image, r]` applies `f` to the range-`r` neighbourhood of every
pixel, in each channel. `f` receives the window as a matrix.

Unlike `MeanFilter` and its siblings, the window is always the full `(2r+1)`
square: the image is extended by repeating its edge samples rather than
clipped, so a corner sees as many values as the centre does.

```scrut
$ wo 'ImageData[ImageFilter[Mean[Flatten[#]] &, Image[{{0., 1.}}], 1]]'
{{0.3333333432674408, 0.6666666865348816}}
```

Which sample the repeated edge puts in the corner of each window:

```scrut
$ wo 'ImageData[ImageFilter[#[[1, 1]] &, Image[{{0., 1., 0.}, {1., 0., 1.}, {0., 1., 0.}}], 1]]'
{{0., 0., 1.}, {0., 0., 1.}, {1., 1., 0.}}
```
