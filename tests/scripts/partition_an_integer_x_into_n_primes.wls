F[x_, n_] :=
 Print["Partitioned ", x, " with ", n, " primes: ",
  StringRiffle[
   ToString /@
    Reverse[First@
      Sort[Select[IntegerPartitions[x, {n}, Prime@Range@PrimePi@x],
        Length@Union@# == n &], Last]], "+"]]

F[40355, 3]
