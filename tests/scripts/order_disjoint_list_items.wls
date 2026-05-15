order[m_, n_] := 
  ReplacePart[m, 
   MapThread[
    Rule, {Position[m, Alternatives @@ n][[;; Length[n]]], n}]];
Print[StringRiffle[
   order[{"the", "cat", "sat", "on", "the", "mat"}, {"mat", 
     "cat"}]]];
Print[StringRiffle[
   order[{"the", "cat", "sat", "on", "the", "mat"}, {"cat", 
     "mat"}]]];
Print[StringRiffle[
   order[{"A", "B", "C", "A", "B", "C", "A", "B", "C"}, {"C", "A", 
     "C", "A"}]]];
Print[StringRiffle[
   order[{"A", "B", "C", "A", "B", "D", "A", "B", "E"}, {"E", "A", 
     "D", "A"}]]];
Print[StringRiffle[order[{"A", "B"}, {"B"}]]];
Print[StringRiffle[order[{"A", "B"}, {"B", "A"}]]];
Print[StringRiffle[order[{"A", "B", "B", "A"}, {"B", "A"}]]];
