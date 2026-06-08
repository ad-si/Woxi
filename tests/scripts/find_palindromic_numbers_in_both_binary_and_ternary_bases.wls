(* Range trimmed from 1000000 to 1000 so the test runs quickly; the
   computation is otherwise identical to the RosettaCode task. *)
palindromify3[n_] :=
  Block[{digits},
   If[Divisible[n, 3], {},
    digits = IntegerDigits[n, 3];
    FromDigits[#, 3] & /@
     {Join[Reverse[digits], digits],
      Join[Reverse[Rest[digits]], {First[digits]}, Rest[digits]]}]];
base2PalindromeQ[n_] := IntegerDigits[n, 2] === Reverse[IntegerDigits[n, 2]];

Print[Union[Select[Flatten[palindromify3 /@ Range[1000]], base2PalindromeQ]]]
