# `ToSphericalCoordinates`

Converts Cartesian coordinates to spherical.

```scrut
$ wo 'ToSphericalCoordinates[{x, y, z}]'
{Sqrt[x^2 + y^2 + z^2], ArcTan[z, Sqrt[x^2 + y^2]], ArcTan[x, y]}
```
