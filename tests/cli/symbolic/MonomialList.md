# `MonomialList`

List of monomials sorted by lexicographic order of exponent vectors.

```scrut
$ wo 'MonomialList[5, {x}]'
{5}
```

A third argument gives the monomial order:

```scrut
$ wo 'MonomialList[x^2 + y^3 + x*y, {x, y}, "DegreeLexicographic"]'
{y^3, x^2, x*y}
```

```scrut
$ wo 'MonomialList[x^2 + y^3 + x*y, {x, y}, "NegativeLexicographic"]'
{y^3, x*y, x^2}
```

An unknown order is refused:

```scrut {output_stream: combined}
$ wo 'MonomialList[x^2 + y^3 + x*y, {x, y}, "Foo"]'

MonomialList::mnmord1: Foo is not a valid monomial order.
MonomialList[x^2 + x*y + y^3, {x, y}, Foo]
```
