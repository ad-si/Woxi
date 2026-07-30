# `ConstantImage`

Create a constant image filled with a specified color or gray level.

```scrut
$ wo 'ImageQ[ConstantImage[0.5, {5, 5}]]'
True
```

Every dimension has to be a positive integer:

```scrut
$ wo 'ConstantImage[0.5, {2, 0}]'

ConstantImage::bddim: The specified dimensions {2, 0} should be a positive integer or a list of positive integers for every spatial dimension.
ConstantImage[0.5, {2, 0}]
```
