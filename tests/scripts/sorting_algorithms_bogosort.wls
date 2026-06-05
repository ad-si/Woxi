(* Bogosort shuffles the list with `RandomSample` until it happens to be sorted.
   The shuffle is random, but the result is always the sorted list. *)
Bogosort[x_List] := Block[{t = x}, While[! OrderedQ[t], t = RandomSample[x]]; t]
Print[Bogosort[{3, 1, 6, 4, 0, -1, 2, 5}]]
