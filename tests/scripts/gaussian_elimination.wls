GaussianElimination[A_?MatrixQ, b_?VectorQ] := Last /@ RowReduce[Flatten /@ Transpose[{A, b}]]
