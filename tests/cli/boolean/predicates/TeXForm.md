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

Greek letters carry their macro, and a rule its arrow:

```scrut
$ wo 'ToString[TeXForm[\[Alpha] -> 2 \[CapitalOmega]]]'
\alpha \to 2 \Omega
```

A complex number prints as one atom, real part first:

```scrut
$ wo 'ToString[TeXForm[3 I + x + 1]]'
x+(1+3 i)
```

A machine real is typeset in its display form:

```scrut
$ wo 'ToString[TeXForm[N[Pi]]]'
3.14159
```
