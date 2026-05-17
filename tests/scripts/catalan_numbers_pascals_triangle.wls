nextrow[lastrow_] := Module[{output},
  output = ConstantArray[1, Length[lastrow] + 1];
  Do[
   output[[i + 1]] = lastrow[[i]] + lastrow[[i + 1]];
   , {i, 1, Length[lastrow] - 1}];
  output
  ]
pascaltriangle[size_] := NestList[nextrow, {1}, size]
catalannumbers[length_] := Module[{output, basetriangle},
  basetriangle = pascaltriangle[2 length];
  list1 = basetriangle[[# *2 + 1, # + 1]] & /@ Range[length];
  list2 = basetriangle[[# *2 + 1, # + 2]] & /@ Range[length];
  list1 - list2
  ]
(* testing *)
catalannumbers[15]
