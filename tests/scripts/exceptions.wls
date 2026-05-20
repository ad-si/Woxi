f[x_] := If[x > 10, Throw[overflow], x!]
Print[Catch[f[2] + f[11]]]
Print[Catch[f[2] + f[3]]]
