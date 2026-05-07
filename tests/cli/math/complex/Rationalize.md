# `Rationalize`

Converts a decimal number to a rational approximation.

```scrut
$ wo 'Rationalize[0.5]'
1/2
```

```scrut
$ wo 'Rationalize[0.333333]'
0.333333
```

```scrut
$ wo 'Rationalize[0.25]'
1/4
```

```scrut
$ wo 'Rationalize[3]'
3
```

With a tolerance argument, finds a rational within that tolerance:

```scrut
$ wo 'Rationalize[0.333333, 0.0001]'
1/3
```

```scrut
$ wo 'Rationalize[0.333333, 0.00001]'
1/3
```

Numbers with up to 5 decimal places are rationalized:

```scrut
$ wo 'Rationalize[0.33333]'
33333/100000
```
