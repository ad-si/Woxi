EmptyQ[a_] := Length[a] == 0
SetAttributes[Push, HoldFirst]; Push[a_, elem_] := AppendTo[a, elem]
SetAttributes[Pop, HoldFirst];
Pop[a_] := If[EmptyQ[a], False, Module[{b = First[a]}, a = Rest[a]; b]]

queue = {};
Print[EmptyQ[queue]];
Push[queue, 1]; Push[queue, 2]; Push[queue, 3];
Print[queue];
Print[EmptyQ[queue]];
Print[Pop[queue]];
Print[Pop[queue]];
Print[queue];
Print[Pop[queue]];
Print[Pop[queue]];
Print[EmptyQ[queue]];
