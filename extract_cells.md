Extract input cells:

```sh
wolframscript -code '
  Import["tests/notebooks/syntax.nb", "Notebook"] \
        Infinity] &) \
  // (Cases[#1,
        Cell[BoxData[boxes_], "Input" | "Code", ___] :>
          With[{s = ToString[ToExpression[boxes, StandardForm, HoldForm], InputForm]},
            StringDrop[StringDrop[s, 9], -1]],
        Infinity] &) \
  // (StringRiffle[#1, "\n\n"] &)
'
```


Make it work for all basic examples:

```wl
Cases[
  WolframLanguageData["Sin", "DocumentationBasicExamples"],
  Cell[_, "Input", ___],
  \[Infinity]
]
```
