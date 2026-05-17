factorial[n_Integer] := n*factorial[n-1]
factorial[0] = 1


(* ========== *)

factorial[n_Integer] :=
  Block[{i, result = 1}, For[i = 1, i <= n, ++i, result *= i]; result]


(* ========== *)

factorial[n_Integer] := Block[{i}, Times @@ Table[i, {i, n}]]
