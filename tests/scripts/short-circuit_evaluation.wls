a[in_] := (Print["a"]; in)
b[in_] := (Print["b"]; in)

a[False] && b[True]
a[True] || b[False]


(* ========== *)

a[True] && b[False]
