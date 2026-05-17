color[y_] := {White, Black}[[Mod[y, 2] + 1]];
Graphics[Join[{Thickness[1/408]},
  Flatten[{color[#], Line[{{# - 1/2, 408}, {# - 1/2, 307}}]} & /@
    Range[408]], {Thickness[1/204]},
  Flatten[{color[#], Line[{{2 # - 1, 306}, {2 # - 1, 205}}]} & /@
    Range[204]], {Thickness[1/136]},
  Flatten[{color[#], Line[{{3 # - 3/2, 204}, {3 # - 3/2, 103}}]} & /@
    Range[136]], {Thickness[1/102]},
  Flatten[{color[#], Line[{{4 # - 2, 102}, {4 # - 2, 1}}]} & /@
    Range[102]]], ImageSize -> {408, 408}]
