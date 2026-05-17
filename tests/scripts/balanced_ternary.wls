frombt = FromDigits[StringCases[#, {"+" -> 1, "-" -> -1, "0" -> 0}],
    3] &;
tobt = If[Quotient[#, 3, -1] == 0,
     "", #0@Quotient[#, 3, -1]] <> (Mod[#,
       3, -1] /. {1 -> "+", -1 -> "-", 0 -> "0"}) &;
btnegate = StringReplace[#, {"+" -> "-", "-" -> "+"}] &;
btadd = StringReplace[
    StringJoin[
     Fold[Sort@{#1[[1]],
          Sequence @@ #2} /. {{x_, x_, x_} :> {x,
           "0" <> #1[[2]]}, {"-", "+", x_} | {x_, "-", "+"} | {x_,
            "0", "0"} :> {"0", x <> #1[[2]]}, {"+", "+", "0"} -> {"+",
            "-" <> #1[[2]]}, {"-", "-", "0"} -> {"-",
           "+" <> #1[[2]]}} &, {"0", ""},
      Reverse@Transpose@PadLeft[Characters /@ {#1, #2}] /. {0 ->
         "0"}]], StartOfString ~~ "0" .. ~~ x__ :> x] &;
btsubtract = btadd[#1, btnegate@#2] &;
btmultiply =
  btadd[Switch[StringTake[#2, -1], "0", "0", "+", #1, "-",
     btnegate@#1],
    If[StringLength@#2 == 1,
     "0", #0[#1, StringDrop[#2, -1]] <> "0"]] &;


(* ========== *)

frombt[a = "+-0++0+"]
b = tobt@-436
frombt[c = "+-++-"]
btmultiply[a, btsubtract[b, c]]
