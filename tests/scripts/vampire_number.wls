ClearAll[VampireQ]
VampireQ[num_Integer] := Module[{poss, divs},
  divs = Select[Divisors[num], # <= Sqrt[num] &];
  poss = {#, num/#} & /@ divs;
  If[Length[poss] > 0,
   poss = Select[poss, Mod[#, 10] =!= {0, 0} &];
   If[Length[poss] > 0,
    poss = Select[poss, Length[IntegerDigits[First[#]]] == Length[IntegerDigits[Last[#]]] &];
    If[Length[poss] > 0,
     poss = Select[poss, Sort[IntegerDigits[num]] == Sort[Join @@ (IntegerDigits /@ #)] &];
     If[Length[poss] > 0
      ,
      Sow[{num, poss}];
      True
      ,
      False
     ]
    ,
    False
    ]
   ,
   False
   ]
  ,
  False
 ]
]


