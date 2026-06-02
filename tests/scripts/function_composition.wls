compose[f_, g_][x_] := f[g[x]]
Print[compose[Sin, Cos][r]]
