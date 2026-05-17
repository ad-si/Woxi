AddSumSquare[input_]:=Append[input,Total[IntegerDigits[Last[input]]^2]]
NestUntilRepeat[a_,f_]:=NestWhile[f,{a},!MemberQ[Most[Last[{##}]],Last[Last[{##}]]]&,All]
HappyQ[a_]:=Last[NestUntilRepeat[a,AddSumSquare]]==1


(* ========== *)

HappyQ[1337]
HappyQ[137]


(* ========== *)

True
False


(* ========== *)

m = 8;
n = 1;
i = 0;
happynumbers = {};
While[n <= m,
 i++;
 If[HappyQ[i],
  n++;
  AppendTo[happynumbers, i]
  ]
 ]
happynumbers


(* ========== *)

{1, 7, 10, 13, 19, 23, 28, 31}
