calc[rpn_] :=
  Module[{tokens = StringSplit[rpn], s = "(" <> ToString@InputForm@# <> ")" &, op, steps},
   op[o_, x_, y_] := ToExpression[s@x <> o <> s@y];
   steps = FoldList[Switch[#2, _?DigitQ, Append[#, FromDigits[#2]],
        _, Append[#[[;; -3]], op[#2, #[[-2]], #[[-1]]]]
        ] &, {}, tokens][[2 ;;]];
   Grid[Transpose[{# <> ":" & /@ tokens,
      StringRiffle[ToString[#, InputForm] & /@ #] & /@ steps}]]];
Print[calc["3 4 2 * 1 5 - 2 3 ^ ^ / +"]];
