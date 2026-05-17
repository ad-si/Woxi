funcs = {Sin, Cos, #^3 &};
funcsi = {ArcSin, ArcCos, #^(1/3) &};
compositefuncs = Composition @@@ Transpose[{funcs, funcsi}];
Table[i[0.666], {i, compositefuncs}]


(* ========== *)

{0.666, 0.666, 0.666}


(* ========== *)

Composition[f,g,h][x]
f@g@h@x
x//h//g//f


(* ========== *)

f[g[h[x]]]
