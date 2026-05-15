# `FindClusters`

Partitions data into clusters by proximity. Cluster *ordering* in
wolframscript depends on an internal k-means seeding that Woxi does not
replicate — the two backends agree on cluster membership but may emit the
two clusters in either order. The expected output here matches both
permutations.

```scrut
$ wo 'FindClusters[{1, 1, 2, 2, 10, 11, 12, 13}]'
\{\{(1, 1, 2, 2\}, \{10, 11, 12, 13|10, 11, 12, 13\}, \{1, 1, 2, 2)\}\} (regex)
```
