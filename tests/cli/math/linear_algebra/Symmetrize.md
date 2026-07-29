# `Symmetrize`

Projects an array onto a tensor symmetry, as a `SymmetrizedArray` that stores
only one canonical entry per orbit.

```scrut
$ wo 'Symmetrize[{{1, 2}, {2, 3}}]'
SymmetrizedArray[StructuredArray`StructuredData[{2, 2}, {{{1, 1} -> 1, {1, 2} -> 2, {2, 2} -> 3}, Symmetric[{1, 2}]}]]
```

`Normal` expands it back to the dense array:

```scrut
$ wo 'Normal[Symmetrize[{{a, b}, {c, d}}]]'
{{a, (b + c)/2}, {(b + c)/2, d}}
```

Under `Antisymmetric` a repeated index forces the entry to zero, so only the
strictly increasing positions are stored:

```scrut
$ wo 'Symmetrize[{{a, b}, {c, d}}, Antisymmetric[{1, 2}]]'
SymmetrizedArray[StructuredArray`StructuredData[{2, 2}, {{{1, 2} -> (b - c)/2}, Antisymmetric[{1, 2}]}]]
```

```scrut
$ wo 'Normal[Symmetrize[{{1, 2}, {3, 4}}, Antisymmetric[{1, 2}]]]'
{{0, -1/2}, {1/2, 0}}
```

`Hermitian` conjugates the entry an odd permutation reaches:

```scrut
$ wo 'Normal[Symmetrize[{{1, 2 + I}, {4 - I, 3}}, Hermitian[{1, 2}]]]'
{{1, 3 + I}, {3 - I, 3}}
```

Without a symmetry the whole tensor is symmetrized, over any rank:

```scrut
$ wo 'Symmetrize[Array[a, {2, 2, 2}]]'
SymmetrizedArray[StructuredArray`StructuredData[{2, 2, 2}, {{{1, 1, 1} -> a[1, 1, 1], {1, 1, 2} -> (a[1, 1, 2] + a[1, 2, 1] + a[2, 1, 1])/3, {1, 2, 2} -> (a[1, 2, 2] + a[2, 1, 2] + a[2, 2, 1])/3, {2, 2, 2} -> a[2, 2, 2]}, Symmetric[{1, 2, 3}]}]]
```

`Dimensions` and `ArrayRules` report on the array it stands for:

```scrut
$ wo 'Dimensions[Symmetrize[Array[a, {2, 2, 2}]]]'
{2, 2, 2}
```

```scrut
$ wo 'ArrayRules[Symmetrize[{{1, 2}, {3, 4}}, Antisymmetric[{1, 2}]]]'
{{1, 2} -> -1/2, {2, 1} -> 1/2, {_, _} -> 0}
```

Permuting two slots only makes sense when they have the same length:

```scrut
$ wo 'Symmetrize[{{1, 2, 3}, {4, 5, 6}}, Symmetric[{1, 2}]]'

Symmetrize::symmcomp: Symmetry specification Symmetric[{1, 2}] is incompatible with expression {2, 3}.
Symmetrize[{{1, 2, 3}, {4, 5, 6}}, Symmetric[{1, 2}]]
```
