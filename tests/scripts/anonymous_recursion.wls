check := #<0&
fib := If[check[#],Throw["Negative Argument"],If[#<=1,1,#0[#-2]+#0[#-1]]&[#]]&
fib /@ Range[0,10]

{1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89}


(* ========== *)

check := (Print[#];#<0)&
fib /@ Range[0,4]
0
1
2
3
4

{1, 1, 2, 3, 5}
