frac[n_] /; IntegerQ[1/n] := frac[n] = {n};
frac[n_] :=
  frac[n] =
   With[{p = Numerator[n], q = Denominator[n]},
    Prepend[frac[Mod[-q, p]/(q Ceiling[1/n])], 1/Ceiling[1/n]]];
disp[f_] :=
  StringRiffle[
    SequenceCases[f,
     l : {_, 1 ...} :>
      If[Length[l] == 1 && l[[1]] < 1, ToString[l[[1]], InputForm],
       "[" <> ToString[Length[l]] <> "]"]], " + "] <> " = " <>
   ToString[Numerator[Total[f]]] <> "/" <>
   ToString[Denominator[Total[f]]];
Print[disp[frac[43/48]]];
Print[disp[frac[5/121]]];
Print[disp[frac[2014/59]]];
fracs = Flatten[Table[frac[p/q], {q, 99}, {p, q}], 1];
Print[disp[MaximalBy[fracs, Length@*Union][[1]]]];
Print[disp[MaximalBy[fracs, Denominator@*Last][[1]]]];
fracs = Flatten[Table[frac[p/q], {q, 999}, {p, q}], 1];
Print[disp[MaximalBy[fracs, Length@*Union][[1]]]];
Print[disp[MaximalBy[fracs, Denominator@*Last][[1]]]];
