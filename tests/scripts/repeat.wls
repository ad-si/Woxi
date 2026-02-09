repeat[f_, n_] := Do[f[], {n}];
repeat[Print["Hello, world!"] &, 5];
