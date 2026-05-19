CatalanN[n_Integer /; n >= 0] := (2 n)!/((n + 1)! n!)
Print[CatalanN /@ Range[0, 15]]
