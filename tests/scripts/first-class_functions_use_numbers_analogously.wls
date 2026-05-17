multiplier[n1_,n2_]:=n1 n2 #&
num={2,4,2+4};
numi=1/num;
multiplierfuncs = multiplier @@@ Transpose[{num, numi}];


(* ========== *)

multiplier[arg__] := Times[arg, #] &
