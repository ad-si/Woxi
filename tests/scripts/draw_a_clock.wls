makeHand[fl_, bl_, fw_, bw_] :=  Polygon[{{-bw, -bl}, {bw, -bl}, {fw, fl}, {0, fl + 8 fw}, {-fw, fl}}/9];

hourHand = makeHand[5, 5/3, .1, .3];minuteHand = makeHand[7, 7/3, .1, .3];
secondHand = {Red, EdgeForm[Black], makeHand[7, 7/3, .1/2, .3/2]};

Graphics[{
  {Thickness[.03], Circle[]},(* Rim *)
  {Thickness[.003], Table[Line[{.9 {Cos[a], Sin[a]}, .95 {Cos[a], Sin[a]}}], {a, 0, 2 \[Pi], 2 \[Pi]/60}]}, (* Thin ticks *)
  {Thickness[.01], Table[Line[{.9 {Cos[a], Sin[a]}, .95 {Cos[a], Sin[a]}}], {a, 0, 2 \[Pi], 2 \[Pi]/12}]}, (* Thick ticks *)
  Style[Table[Text[i, .77 {Cos[-i \[Pi]/6 + \[Pi]/2], Sin[-i \[Pi]/6 + \[Pi]/2]}], {i, 1, 12}], FontFamily -> "Helvetica", FontSize -> 36], (* Numbers *)
  Rotate[hourHand, Dynamic[Refresh[-30 Mod[AbsoluteTime[]/3600, 60] \[Degree], UpdateInterval -> 60]], {0, 0}],
  Rotate[minuteHand, Dynamic[Refresh[-6 Mod[AbsoluteTime[]/60, 60] \[Degree], UpdateInterval -> 1]], {0, 0}],
  Rotate[secondHand, Dynamic[Refresh[-6 Mod[AbsoluteTime[], 60] \[Degree], UpdateInterval -> 1/20]], {0, 0}]
  }]
