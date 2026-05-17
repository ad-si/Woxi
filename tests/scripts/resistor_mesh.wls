gridresistor[p_, q_, ai_, aj_, bi_, bj_] :=
  Block[{A, B, k, c, V}, A = ConstantArray[0, {p*q, p*q}];
   Do[k = (i - 1) q + j;
    If[{i, j} == {ai, aj}, A[[k, k]] = 1, c = 0;
     If[1 <= i + 1 <= p && 1 <= j <= q, c++; A[[k, k + q]] = -1];
     If[1 <= i - 1 <= p && 1 <= j <= q, c++; A[[k, k - q]] = -1];
     If[1 <= i <= p && 1 <= j + 1 <= q, c++; A[[k, k + 1]] = -1];
     If[1 <= i <= p && 1 <= j - 1 <= q, c++; A[[k, k - 1]] = -1];
     A[[k, k]] = c], {i, p}, {j, q}];
   B = SparseArray[(k = (bi - 1) q + bj) -> 1, p*q];
   LinearSolve[A, B][[k]]];
N[gridresistor[10, 10, 2, 2, 8, 7], 40]


(* ========== *)

graphresistor[g_, a_, b_] :=
  LinearSolve[
    SparseArray[{{a, a} -> 1, {i_, i_} :> Length@AdjacencyList[g, i],
      Alternatives @@ Join[#, Reverse /@ #] &[
        List @@@ EdgeList[VertexDelete[g, a]]] -> -1}, {VertexCount[
       g], VertexCount[g]}], SparseArray[b -> 1, VertexCount[g]]][[b]];
N[graphresistor[GridGraph[{10, 10}], 12, 77], 40]
