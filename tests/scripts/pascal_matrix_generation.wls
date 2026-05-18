symPascal[size_] := NestList[Accumulate, Table[1, {k, size}], size - 1]

upperPascal[size_] := CholeskyDecomposition[symPascal@size]

lowerPascal[size_] := Transpose@CholeskyDecomposition[symPascal@size]

Column[MapThread[
  Labeled[Grid[#1@5], #2, Top] &, {{upperPascal, lowerPascal,
    symPascal}, {"Upper", "Lower", "Symmetric"}}]]


