ffr[j_] := Module[{R = {1}, S = 2, k = 1},
    Do[While[Position[R, S] != {}, S++]; k = k + S; S++;
    R = Append[R, k], {n, 1, j - 1}]; R]

 ffs[j_] := Differences[ffr[j + 1]]


(* ========== *)

ffr[10]

 (* out *)
 {1, 3, 7, 12, 18, 26, 35, 45, 56, 69}


(* ========== *)

t = Sort[Join[ffr[40], ffs[960]]];

 t == Range[1000]

 (* out *)
 True
