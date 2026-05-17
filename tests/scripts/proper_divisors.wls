ProperDivisors[n_Integer /; n > 0] := Most@Divisors@n;


(* ========== *)

Grid@Table[{n, ProperDivisors[n]}, {n, 1, 10}]


(* ========== *)

Fold[
 Last[SortBy[{#1, {#2, Length@ProperDivisors[#2]}}, Last]] &,
 {0, 0},
 Range[20000]]


(* ========== *)

Last@SortBy[
  Table[
    {n, Length@ProperDivisors[n]},
    {n, 1, 20000}],
  Last]
