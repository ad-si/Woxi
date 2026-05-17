Table[Last /@ Reverse@Tally[First /@ IntegerPartitions[n]], {n, 10}] // Grid


(* ========== *)

PartitionsP /@ {23, 123, 1234, 12345}


(* ========== *)

DiscretePlot[PartitionsP[n], {n, 1, 999}, PlotRange -> All]
