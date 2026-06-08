data = {{44, "Solomon"}, {42, "Jason"}, {42, "Errol"}, {41, "Garry"},
   {41, "Bernard"}, {41, "Barry"}, {39, "Stephen"}};

rank[data_, type_] :=
 Module[{t = Transpose@{Sort@data, Range[Length@data, 1, -1]}},
  Switch[type,
   "standard", data /. Rule @@@ First /@ SplitBy[t, First],
   "modified", data /. Rule @@@ Last /@ SplitBy[t, First],
   "dense", data /. Thread[# -> Range[Length@#]] &@SplitBy[t, First][[All, 1, 1]],
   "ordinal", Reverse@Ordering[data],
   "fractional", data /. Rule @@@ (Mean[#] /. {a_Rational :> N[a]} &) /@ SplitBy[t, First]]]

scores = data[[All, 1]];
Do[Print[type, ": ", rank[scores, type]],
 {type, {"standard", "modified", "dense", "ordinal", "fractional"}}]
