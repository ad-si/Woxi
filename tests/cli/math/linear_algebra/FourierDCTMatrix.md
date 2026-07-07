# `FourierDCTMatrix`

Generate the `n`×`n` discrete cosine transform matrix. The optional second
argument selects the transform type (1–4); it defaults to type 2.

```scrut
$ wo 'FourierDCTMatrix[1]'
{{1}}
```

Type 2 (the default):

```scrut
$ wo 'FourierDCTMatrix[3]'
{{1/Sqrt[3], 1/2, 1/(2*Sqrt[3])}, {1/Sqrt[3], 0, -(1/Sqrt[3])}, {1/Sqrt[3], -1/2, 1/(2*Sqrt[3])}}
```

Type 1:

```scrut
$ wo 'FourierDCTMatrix[3, 1]'
{{1/2, 1/2, 1/2}, {1, 0, -1}, {1/2, -1/2, 1/2}}
```

Type 3:

```scrut
$ wo 'FourierDCTMatrix[3, 3]'
{{1/Sqrt[3], 1/Sqrt[3], 1/Sqrt[3]}, {1, 0, -1}, {1/Sqrt[3], -2/Sqrt[3], 1/Sqrt[3]}}
```
