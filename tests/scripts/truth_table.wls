VariableNames[data_] := Module[ {TokenRemoved},
 TokenRemoved = StringSplit[data,{"~And~","~Or~","~Xor~","!","(",")"}];
 Union[Select[Map[StringTrim,TokenRemoved] , Not[StringMatchQ[#,""]]&]]
]

TruthTable[BooleanEquation_] := Module[ {TestDataSet},
  TestDataSet = MapThread[Rule,{ToExpression@VariableNames[BooleanEquation],#}]&/@
     Tuples[{False,True}, Length[VariableNames[BooleanEquation]]];

  Join[List[Flatten[{VariableNames[BooleanEquation],BooleanEquation}]],
    Flatten[{#/.Rule[x_,y_] -> y,ReplaceAll[ToExpression[BooleanEquation],#]}]&/@TestDataSet]//Grid
]
