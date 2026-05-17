(*parsing:*)
parse[string_] :=
 Module[{e},
  StringCases[string,
     "+" | "-" | "*" | "/" | "(" | ")" |
      DigitCharacter ..] //. {a_String?DigitQ :>
      e[ToExpression@a], {x___, PatternSequence["(", a_e, ")"],
       y___} :> {x, a,
       y}, {x :
        PatternSequence[] |
         PatternSequence[___, "(" | "+" | "-" | "*" | "/"],
       PatternSequence[op : "+" | "-", a_e], y___} :> {x, e[op, a],
       y}, {x :
        PatternSequence[] | PatternSequence[___, "(" | "+" | "-"],
       PatternSequence[a_e, op : "*" | "/", b_e], y___} :> {x,
       e[op, a, b],
       y}, {x :
        PatternSequence[] | PatternSequence[___, "(" | "+" | "-"],
       PatternSequence[a_e, b_e], y___} :> {x, e["*", a, b],
       y}, {x : PatternSequence[] | PatternSequence[___, "("],
       PatternSequence[a_e, op : "+" | "-", b_e],
       y : PatternSequence[] |
         PatternSequence[")" | "+" | "-", ___]} :> {x, e[op, a, b],
       y}} //. {e -> List, {a_Integer} :> a, {a_List} :> a}]

(*evaluation*)
evaluate[a_Integer] := a;
evaluate[{"+", a_}] := evaluate[a];
evaluate[{"-", a_}] := -evaluate[a];
evaluate[{"+", a_, b_}] := evaluate[a] + evaluate[b];
evaluate[{"-", a_, b_}] := evaluate[a] - evaluate[b];
evaluate[{"*", a_, b_}] := evaluate[a]*evaluate[b];
evaluate[{"/", a_, b_}] := evaluate[a]/evaluate[b];
evaluate[string_String] := evaluate[parse[string]]


(* ========== *)

parse["-1+2(3+4*-5/6)"]
evaluate["-1+2(3+4*-5/6)"]
