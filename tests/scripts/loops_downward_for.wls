For[i = 10, i >= 0, i--, Print[i]]


(* ========== *)

Do[Print[i], {i, 10, 0, -1}]


(* ========== *)

Scan[Print, Range[10, 0, -1]]
