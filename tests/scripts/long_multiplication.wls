LongMultiplication[a_,b_]:=Module[{d1,d2},
  d1=IntegerDigits[a]//Reverse;
  d2=IntegerDigits[b]//Reverse;
  Sum[d1[[i]]d2[[j]]*10^(i+j-2),{i,1,Length[d1]},{j,1,Length[d2]}]
 ]


