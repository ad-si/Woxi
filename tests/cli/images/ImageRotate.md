# `ImageRotate`

Rotates an image by a given angle.

```scrut
$ wo 'ImageDimensions[ImageRotate[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], Pi/2]]'
{2, 3}
```

The angle may instead name the edge the current top should end up at:

```scrut
$ wo 'ImageDimensions[ImageRotate[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], Left]]'
{2, 3}
```

```scrut
$ wo 'ImageDimensions[ImageRotate[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], Top]]'
{3, 2}
```

Anything else is reported and the call left alone:

```scrut
$ wo 'ImageRotate[Image[{{0.}}], x]'

ImageRotate::imgang: Angle x should be a real number; one of Top, Bottom, Left or Right; or a rule from one to another.
ImageRotate[-Image-, x]
```
