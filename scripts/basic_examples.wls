#!/usr/bin/env wolframscript -file

If[Length[$ScriptCommandLine] < 2,
  Print["Usage: basic_examples.wls <SymbolName>"];
  Exit[1]
];

symbol = $ScriptCommandLine[[2]];

usage = WolframLanguageData[symbol, "PlaintextUsage"];
Print[usage, "\n"];

data = WolframLanguageData[symbol, "DocumentationBasicExamples"];

examples = Cases[data,
  {___, RawBoxes[Cell[desc_String, "ExampleText"]],
   RawBoxes[Cell[BoxData[code_], "Input", ___]], ___} :>
  {desc, StringTake[ToString[ToExpression[code, StandardForm, HoldForm], InputForm], {10, -2}]},
  {1}];

Scan[
  Print[#[[1]], "\n", #[[2]], "\n"] &,
  examples
]
