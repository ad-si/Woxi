operations =
  DeleteCases[Tuples[{"+", "-", ""}, 9], {x_, y__} /; x == "+"];

sums =
  Map[StringJoin[Riffle[#, CharacterRange["1", "9"]]] &, operations];


(* ========== *)

TableForm@Select[sums, ToExpression@# == 100 &]


(* ========== *)

MaximalBy[Counts@ToExpression@sums, Identity]


(* ========== *)

pos = Cases[ToExpression@sums, _?Positive];
n = 1; While[MemberQ[pos, n], ++n];


(* ========== *)

{#, ToExpression@#}&/@TakeLargestBy[sums, ToExpression, 10]//TableForm
