grid = MapThread[{#1,#2} &, {Range @ 16, Range @ 16}]

Move[x_] := (empty = Select[grid, #[[1]]==16 &][[1,2]];
             If[(empty == x+4) || (empty == x-4) ||
                (Mod[empty,4] != 0 && empty == x-1) ||
                (Mod[empty,4] != 1 && empty == x+1),
             oldEmpty = grid[[empty]][[1]];
             grid[[empty]][[1]] = grid[[x]][[1]];
             grid[[x]][[1]] = oldEmpty])

CButton[{x_,loc_}] := If[x==16, Null, Button[x,Move @ loc]]

Dynamic @ Grid @ Partition[CButton /@ grid,4]
