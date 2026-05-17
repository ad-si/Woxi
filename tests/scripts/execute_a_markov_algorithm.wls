markov[ruleset_, text_] :=
  Module[{terminating = False, output = text,
    rules = StringCases[
      ruleset, {StartOfLine ~~ pattern : Except["\n"] .. ~~
         " " | "\t" .. ~~ "->" ~~ " " | "\t" .. ~~ dot : "" | "." ~~
         replacement : Except["\n"] .. ~~ EndOfLine :> {pattern,
         replacement, dot == "."}}]},
   While[! terminating, terminating = True;
    Do[If[! StringFreeQ[output, rule[[1]]],
      output = StringReplace[output, rule[[1]] -> rule[[2]]];
      If[! rule[[3]], terminating = False]; Break[]], {rule, rules}]];
    output];


(* ========== *)

markov["# Turing machine: three-state busy beaver
 #
 # state A, symbol 0 => write 1, move right, new state B
 A0 -> 1B
 # state A, symbol 1 => write 1, move left, new state C
 0A1 -> C01
 1A1 -> C11
 # state B, symbol 0 => write 1, move left, new state A
 0B0 -> A01
 1B0 -> A11
 # state B, symbol 1 => write 1, move right, new state B
 B1 -> 1B
 # state C, symbol 0 => write 1, move left, new state B
 0C0 -> B01
 1C0 -> B11
 # state C, symbol 1 => write 1, move left, halt
 0C1 -> H01
 1C1 -> H11", "000000A000000"]
