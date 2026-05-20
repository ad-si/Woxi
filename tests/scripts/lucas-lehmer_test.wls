Print[Select[Table[M = 2^p - 1;
  For[i = 1; s = 4, i <= p - 2, i++, s = Mod[s^2 - 2, M]];
  If[s == 0, "M" <> ToString@p, p], {p,
   Prime /@ Range[300]}], StringQ]];
