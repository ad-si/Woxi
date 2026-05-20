EmptyQ[a_] := If[Length[a] == 0, True, False]
SetAttributes[Push, HoldAll];
Push[a_, elem_] := AppendTo[a, elem]
SetAttributes[Pop, HoldAllComplete];
Pop[a_] := If[EmptyQ[a], False, b = Last[a]; Set[a, Most[a]]; b]
Peek[a_] := If[EmptyQ[a], False, Last[a]]

stack = {};
Push[stack, 1]; Push[stack, 2]; Push[stack, 3]; Push[stack, 4];
Print[Peek[stack]]
Print[Pop[stack]]
Print[Peek[stack]]
