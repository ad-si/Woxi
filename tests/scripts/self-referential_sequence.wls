(* Range trimmed from 1000000 to 1000 so the test runs quickly; the
   computation is otherwise identical to the RosettaCode task. *)
selfRefSequence[x_] :=
  FromDigits@Flatten@Reverse@
    Cases[Transpose@{RotateRight[DigitCount@x, 1], Range[0, 9]},
     Except[{0, _}]]
DisplaySequence[x_] := NestWhileList[selfRefSequence, x, UnsameQ[##] &, 4]
data = {#, Length@DisplaySequence[#]} & /@ Range[1000];
maxLen = Max@data[[;; , 2]];
seed = Select[data, #[[2]] == maxLen &][[1, 1]];
Print["Seed: ", seed];
Print["Iterations: ", maxLen];
Print["Sequence: ", DisplaySequence[seed]]
