# `ImageCrop`

Crops an image to a specified region.

With no size, uniform borders are trimmed away:

```scrut
$ wo 'ImageData[ImageCrop[Image[{{0, 0, 0}, {0, 0.5, 0}, {0, 0, 0}}]]]'
{{0.5}}
```

A size crops to that width and height, centered:

```scrut
$ wo 'Round[ImageData[ImageCrop[Image[Partition[Range[16], 4]/100.], {2, 2}]]*100, 0.01]'
{{6., 7.}, {10., 11.}}
```

A third argument names the side (or sides) the pixels come off,
so `Left` keeps the right-hand part:

```scrut
$ wo 'Round[ImageData[ImageCrop[Image[Partition[Range[16], 4]/100.], {2, 2}, Left]]*100, 0.01]'
{{7., 8.}, {11., 12.}}
```

A size larger than the image pads it out with black instead.
