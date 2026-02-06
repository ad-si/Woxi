safe[q_List, n_] :=
 With[{l = Length@q},
  Length@Union@q == Length@Union[q + Range@l] ==
   Length@Union[q - Range@l] == l]

nQueen[q_List: {}, n_] :=
 If[safe[q, n],
  If[Length[q] == n, {q},
   Cases[nQueen[Append[q, #], n] & /@ Range[n],
    Except[{Null} | {}], {2}]], Null]

Print[nQueen[{}, 6]]

matrixView[n_] :=
 Grid[Normal@
     SparseArray[MapIndexed[{#, First@#2} -> "Q" &, #], {n, n}, "."],
    Frame -> All] & /@ nQueen[n]
Print[matrixView[6]]
