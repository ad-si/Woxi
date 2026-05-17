s = RandomSample@Flatten[{Table[0, 26], Table[1, 26]}]
g = Take[s, {1, -1, 2}]
d = Take[s, {2, -1, 2}]
a = b = {};
Table[If[g[[i]] == 1, AppendTo[a, d[[i]]], AppendTo[b, d[[i]]]], {i, 
   Length@g}];
a
b
dice = RandomInteger[{1, 6}]
ra = Sort@RandomSample[Range@Length@a, dice]
a = {Delete[a, List /@ ra], a[[ra]]}
rb = Sort@RandomSample[Range@Length@b, dice]
b = {Delete[b, List /@ rb], b[[rb]]}
finala = Join[a[[1]], b[[2]]]
finalb = Join[b[[1]], a[[2]]]
Count[finala, 1] == Count[finalb, 0]
