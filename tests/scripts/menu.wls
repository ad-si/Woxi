textMenu[data_List] := Module[{choice},
  If[Length@data == 0, Return@""];
  While[!(IntegerQ@choice && Length@data >= choice > 0),
   MapIndexed[Print[#2[[1]], ") ", #1]&, data];
   choice = Input["Enter selection..."]
   ];
  data[[choice]]
  ]
