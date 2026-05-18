hexagon[{x_, y_}] :=
  Polygon[Transpose[{{1/2, 1/4, -1/4, -1/2, -1/4, 1/4} +
      x, {0, Sqrt[3]/4, Sqrt[3]/4, 0, -Sqrt[3]/4, -Sqrt[3]/4} + y}]];
off = Transpose[{ConstantArray[0, 20], {0, 0, 0, 0, Sqrt[3]/4,
     Sqrt[3]/4, Sqrt[3]/4, Sqrt[3]/4, 0, 0, 0, 0, Sqrt[3]/4,
     Sqrt[3]/4, Sqrt[3]/4, Sqrt[3]/4, 0, 0, 0, 0}}];
DynamicModule[{letters = RandomSample[CharacterRange["A", "Z"], 20],
  blue = False, cols = {},
  locs = Tuples[{Range[1, 4, 3/4],
      Range[1, 1 + (3 Sqrt[3])/2, Sqrt[3]/2]}] - off},
 EventHandler[
  Dynamic[Graphics[{EdgeForm[{Thick, Black}], LightGray,
     hexagon /@ locs, {#[[1]], hexagon[#[[2]]]} & /@ cols, Black,
     MapThread[
      Text, {Style[#, FontSize -> Large] & /@ letters, locs}], Red,
     Text[Style[
       StringJoin[
        letters[[FirstPosition[locs, #[[2]]][[1]]]] & /@
         Cases[cols, {Red, _}][[All, 2]]],
       FontSize -> 40], {5/2, -1/2}, {Right, Center}], Blue,
     Text[Style[
       StringJoin[
        letters[[FirstPosition[locs, #[[2]]][[1]]]] & /@
         Cases[cols, {Blue, _}][[All, 2]]],
       FontSize -> 40], {5/2, -1/2}, {Left, Center}]},
    PlotRange -> {{-1, 6}, Automatic},
    ImageSize -> Large]], {"MouseClicked" :>
    If[! MemberQ[cols[[All, 2]],
       Nearest[locs, MousePosition["Graphics"]][[1]]],
     AppendTo[
      cols, {If[blue, Blue, Red],
       Nearest[locs, MousePosition["Graphics"]][[1]]}];
     blue = ! blue]}]]
