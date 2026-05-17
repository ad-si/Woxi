PassFunc[f_, g_, h_, x_] := f[g[x]*h[x]]
PassFunc[Tan, Cos, Sin, x]
% /. x -> 0.12
PassFunc[Tan, Cos, Sin, 0.12]


(* ========== *)

Tan[Cos[x] Sin[x]]
0.119414
0.119414
