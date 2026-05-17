amicableQ[n_] :=
 Module[{sum = Total[Most@Divisors@n]},
  sum != n && n == Total[Most@Divisors@sum]]

Grid@Partition[Cases[Range[4, 20000], _?(amicableQ@# &)], 2]
