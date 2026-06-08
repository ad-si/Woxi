pascal[n_] := Table[Binomial[i, j], {i, 0, n - 1}, {j, 0, i}]
Do[Print[Row[row, " "]], {row, pascal[8]}]
