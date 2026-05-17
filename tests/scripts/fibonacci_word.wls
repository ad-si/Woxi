entropy = (p - 1) Log[2, 1 - p] - p Log[2, p];

TableForm[
 Table[{k, Fibonacci[k],
   Quiet@Check[N[entropy /. {p -> Fibonacci[k - 1]/Fibonacci[k]}, 15],
      0]}, {k, 37}],
 TableHeadings -> {None, {"N", "Length", "Entropy"}}]
