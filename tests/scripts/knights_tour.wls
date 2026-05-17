knightsTourMoves[start_] :=
  Module[{
    vertexLabels = (# -> ToString@c[[Quotient[# - 1, 8] + 1]] <>  ToString[Mod[# - 1, 8] + 1]) & /@ Range[64], knightsGraph,
       hamiltonianCycle, end},
    knightsGraph = KnightTourGraph[i, i, VertexLabels -> vertexLabels,  ImagePadding -> 15];
    hamiltonianCycle = ((FindHamiltonianCycle[knightsGraph] /. UndirectedEdge -> DirectedEdge) /. labels)[[1]];
    end = Cases[hamiltonianCycle, (x_ \[DirectedEdge] start) :> x][[1]];
    FindShortestPath[g, start, end]]


