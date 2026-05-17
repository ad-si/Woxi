compare[x_, y_] := Module[{},
  If[x == y,
   Print["Comparing for equality (case sensitive): " <> x <> " and " <> y <> " ARE equal"],
   Print["Comparing for equality (case sensitive): " <> x <> " and " <> y <> " are NOT equal" ]] ;
  If[x != y,
   Print["Comparing for inequality (case sensitive): " <> x <> " and " <> y <> " are NOT equal"],
   Print["Comparing for inequality (case sensitive): " <> x <> " and " <> y <> " ARE equal" ]] ;
  Switch[Order[x, y],
    1, Print["Comparing for order (case sensitive): " <> x <> " comes before " <> y],
   -1, Print["Comparing for order (case sensitive): " <> x <> " comes after " <> y],
    0, Print["Comparing for order (case sensitive): " <> x <> " comes in the same spot as " <> y]];
  If[ToLowerCase[x] == ToLowerCase[y],
   Print["Comparing for equality (case insensitive): " <> x <> " and " <> y <> " ARE equal"],
   Print["Comparing for equality (case insensitive): " <> x <> " and " <> y <> " are NOT equal" ]] ;
  Print[];
  ]
compare["Hello", "Hello"]
compare["3.1", "3.14159"]
compare["mathematica", "Mathematica"]
