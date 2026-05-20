Print[DeleteDuplicates[Tuples[{"iced", "jam", "plain"}, 2],Sort[#1] == Sort[#2] &]];
Print[Combi[x_, y_] := Binomial[(x + y) - 1, y]];
Print[Combi[3, 2]];
Print[Combi[10, 3]];
