solve[sudoku_] :=
 NestWhile[
  Join @@ Table[
     Table[ReplacePart[s, #1 -> n], {n, #2}] & @@
      First@SortBy[{#,
           Complement[Range@9, s[[First@#]], s[[;; , Last@#]],
            Catenate@
             Extract[Partition[s, {3, 3}], Quotient[#, 3, -2]]]} & /@
         Position[s, 0, {2}],
        Length@Last@# &], {s, #}] &, {sudoku}, ! FreeQ[#, 0] &]
