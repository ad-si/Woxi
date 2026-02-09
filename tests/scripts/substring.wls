n = 2
m = 3
Print[StringTake["Mathematica", {n+1, n+m-1}]]

Print[StringDrop["Mathematica", n]]

(* StringPosition returns a list of starting and ending character positions for a substring *)
pos = StringPosition["Mathematica", "e"][[1]][[1]]
Print[StringTake["Mathematica", {pos, pos+m-1}]]
