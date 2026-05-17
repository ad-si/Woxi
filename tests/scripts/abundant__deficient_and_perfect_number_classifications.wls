classify[n_Integer] := Sign[Total[Most@Divisors@n] - n]

StringJoin[
 Flatten[Tally[
     Table[classify[n], {n, 20000}]] /. {-1 -> "deficient: ",
     0 -> "  perfect: ", 1 -> "  abundant: "}] /.
  n_Integer :> ToString[n]]
