Co9[n_, b_: 10] := 
  With[{ans = FixedPoint[Total@IntegerDigits[#, b] &, n]}, 
   If[ans == b - 1, 0, ans]];


(* ========== *)

Co9 /@ (vals = {1235, 2345, 4753})
  {2, 5, 1}

Total[Co9 /@ vals] == Co9[Total[vals]]
  True


(* ========== *)

task2 = Select[Range@100, Co9[#] == Co9[#^2] &]


(* ========== *)

Co9eff[n_, b_: 10] := Mod[n, b - 1];


(* ========== *)

task2 == Select[Range@100, Co9eff[#] == Co9eff[#^2] &]
   True

Select[Range@100, Co9eff[#, 17] == Co9eff[#^2, 17] &]
   {1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97}
