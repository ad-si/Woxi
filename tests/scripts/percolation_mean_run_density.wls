(* Each row would normally be `Unitize[Chop[RandomReal[1, len], 1 - p]]`, i.e. a
   random 0/1 vector with P(1) = p. Hardcoded to a fixed set of rows here so the
   snapshot is deterministic. The mean run density is the average number of
   maximal runs of 1s per cell. *)
rows = {{1, 0, 1, 1, 0, 0, 1, 0, 1, 1},
   {0, 1, 1, 0, 1, 0, 0, 1, 1, 0},
   {1, 1, 0, 0, 1, 1, 0, 1, 0, 0}};
meanRunDensity =
  N[Mean[Length[Cases[Split[#], {1, ___}]] & /@ rows]/Length[rows[[1]]]];
Print["Mean run density: ", meanRunDensity]
