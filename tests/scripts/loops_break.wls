(* The values would normally come from RandomInteger[{0, 19}]; they are
   hardcoded here so the output is deterministic and matches wolframscript
   (the original also had a `RandomIntger` typo and unbalanced brackets). *)
randoms = {3, 7, 15, 10, 2};
Do[
  Print[r];
  If[r == 10, Break[]],
  {r, randoms}]
