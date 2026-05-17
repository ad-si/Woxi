huffman[s_String] := huffman[Characters[s]];
huffman[l_List] := Module[{merge, structure, rules},

   (*merge front two branches. list is assumed to be sorted*)
   merge[k_] := Replace[k, {{a_, aC_}, {b_, bC_}, rest___} :> {{{a, b}, aC + bC}, rest}];

   structure = FixedPoint[
      Composition[merge, SortBy[#, Last] &],
      Tally[l]][[1, 1]];

   rules = (# -> Flatten[Position[structure, #] - 1]) & /@ DeleteDuplicates[l];

   {Flatten[l /. rules], rules}];
