If2[test1_, test2_, condBoth_, cond1_, cond2_, condNone_] := With[
   {result1 = test1,
    result2 = test2},
   Which[
    result1 && result2, condBoth,
    result1, cond1,
    result2, cond2,
    True, condNone]];
SetAttributes[If2, HoldAll];


(* ========== *)

x = 0;
If2[Mod[(++x), 2] == 0, Mod[x, 3] == 1, Print["Both: ", x], Print["First: ", x], Print["Second: ", x], Print["Neither: ", x]];
If2[Mod[(++x), 2] == 0, Mod[x, 3] == 1, Print["Both: ", x], Print["First: ", x], Print["Second: ", x], Print["Neither: ", x]];
If2[Mod[(++x), 2] == 0, Mod[x, 3] == 1, Print["Both: ", x], Print["First: ", x], Print["Second: ", x], Print["Neither: ", x]];
If2[Mod[(++x), 2] == 0, Mod[x, 3] == 1, Print["Both: ", x], Print["First: ", x], Print["Second: ", x], Print["Neither: ", x]];
