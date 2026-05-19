ClearAll[mdr, mp, nums];
mdr[n_] := NestWhile[Times @@ IntegerDigits[#] &, n, # > 9 &];
mp[n_] := Length@NestWhileList[Times @@ IntegerDigits[#] &, n, # > 9 &] - 1;
TableForm[{#, mdr[#], mp[#]} & /@ {123321, 7739, 893, 899998},
  TableHeadings -> {None, {"Number", "MDR", "MP"}}]
nums = ConstantArray[{}, 10];
For[i = 0, Min[Length /@ nums] < 5, i++,  AppendTo[nums[[mdr[i] + 1]], i]];
TableForm[Table[{i, Take[nums[[i + 1]], 5]}, {i, 0, 9}],
  TableHeadings -> {None, {"MDR", "First 5"}}, TableDepth -> 2]
