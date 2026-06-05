(* `numbers = RandomReal[{-2, 6}, 11]` normally draws 11 random reals in [-2, 6].
   Hardcoded to a fixed list here so the snapshot is deterministic. The results
   are rounded so the two engines print identical digits. *)
numbers = {3.5, -1.2, 4.0, 0.7, 5.1, -0.5, 2.2, 6.0, 1.3, -1.8, 5.5};
tpk[numbers_, overflowVal_] := Module[{revNumbers},
  revNumbers = Reverse[numbers];
  f[x_] := Abs[x]^0.5 + 5 x^3;
  Do[
   If[f[i] > overflowVal,
    Print["f[", i, "]= Overflow"],
    Print["f[", i, "]= ", Round[f[i], 0.001]]],
   {i, revNumbers}]]
tpk[numbers, 400]
