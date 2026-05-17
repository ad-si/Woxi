MonteCarloPi[samplesize_Integer] := N[4Mean[If[# > 1, 0, 1] & /@ Norm /@ RandomReal[1, {samplesize, 2}]]]


