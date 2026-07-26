# `TeXForm`

Convert expressions to LaTeX notation.

```scrut
$ wo 'TeXForm[Pi]'
TeXForm[Pi]
```

A matrix becomes a parenthesized LaTeX array:

```scrut
$ wo 'ToString[TeXForm[{{1, 2}, {3, 4}}]]'
\left(
\begin{array}{cc}
 1 & 2 \\
 3 & 4 \\
\end{array}
\right)
```

`TableForm` pads short rows out to the widest one instead:

```scrut
$ wo 'ToString[TeXForm[TableForm[{{1}, {2, 3}}]]]'
\begin{array}{cc}
 1 & \text{} \\
 2 & 3 \\
\end{array}
```
