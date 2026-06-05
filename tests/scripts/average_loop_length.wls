(* For a random function on n points, the expected number of distinct values
   visited before the path first repeats is
   Sum[n! n^(n-k-1)/(n-k)!, {k, n}] / n^(n-1).
   The RosettaCode task also estimates this empirically with `RandomInteger`
   sampling; that random estimate is omitted here so the snapshot is
   deterministic. The value is rounded so the two engines print identical digits. *)
analytical[n_] := Round[N[Sum[n! n^(n - k - 1)/(n - k)!, {k, n}]/n^(n - 1)], 0.0001];
Do[Print[{n, analytical[n]}], {n, 1, 20}]
