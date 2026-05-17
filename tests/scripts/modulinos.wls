#!/usr/bin/env MathKernel -script

MeaningOfLife[] = 42

ScriptName[] = Piecewise[
	{
		{"Interpreted", Position[$CommandLine, "-script", 1] == {}}
	},
	$CommandLine[[Position[$CommandLine, "-script", 1][[1,1]] + 1]]
]

Program = ScriptName[];

If[StringMatchQ[Program, ".*scriptedmain.*"],
	Print["Main: The meaning of life is " <> ToString[MeaningOfLife[]]]
]


(* ========== *)

#!/usr/bin/env MathKernel -script

Get["scriptedmain.ma"]

Print["Test: The meaning of life is " <> ToString[MeaningOfLife[]]]
