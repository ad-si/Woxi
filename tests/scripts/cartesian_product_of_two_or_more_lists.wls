cartesianProduct[args__] := Flatten[Outer[List, args], Length[{args}] - 1]
