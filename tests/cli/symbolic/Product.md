# `Product`

Symbolic product.

```scrut
$ wo 'Product[k, {k, 1, 5}]'
120
```

```scrut
$ wo 'Product[k^2, {k, 1, 3}]'
36
```

Multiple iterators nest, with the rightmost innermost.

```scrut
$ wo 'Product[i, {i, 1, 4}, {j, 1, 2}]'
576
```
