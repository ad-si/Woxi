ClearAll[SortByPos, RadixSort]
SortByPos[data : {_List ..}, pos_Integer] := Module[{digs, order},
  digs = data[[All, pos]];
  order = Ordering[digs];
  data[[order]]
  ]
RadixSort[x : {_Integer ..}] := Module[{y, digs, maxlen, offset},
  offset = Min[x];
  y = x - offset;
  digs = IntegerDigits /@ y;
  maxlen = Max[Length /@ digs];
  digs = IntegerDigits[#, 10, maxlen] & /@ y;
  digs = Fold[SortByPos, digs, -Range[maxlen]];
  digs = FromDigits /@ digs;
  digs += offset;
  digs
  ]


(* ========== *)

RadixSort[{170,45,75,-90,-802,24,2,66}]
RadixSort[{170,45,75,90,802,2,24,66}]
