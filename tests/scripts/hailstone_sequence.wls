HailstoneF[n_] := NestWhileList[If[OddQ@#, 3 # + 1, #/2] &, n, # > 1 &]


