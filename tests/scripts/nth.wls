suffixlist = {"th", "st", "nd", "rd", "th", "th", "th", "th", "th","th"};
addsuffix[n_] := Module[{suffix},
  suffix = Which[
    Mod[n, 100] <= 10, suffixlist[[Mod[n, 10] + 1]],
    Mod[n, 100] > 20, suffixlist[[Mod[n, 10] + 1]],
    True, "th"
    ];
  ToString[n] <> suffix
  ]
addsuffix[#] & /@ Range[0, 25] (* test 1 *)
addsuffix[#] & /@ Range[250, 265] (* test 2 *)
addsuffix[#] & /@ Range[1000, 1025] (* test 3 *)
