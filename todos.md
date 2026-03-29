# Ideas

- Wolfbook - VSCode extension for Wolfram Mathematica notebooks https://github.com/vanbaalon/wolfbook
- Integrate data
    - E.g. via https://worlddataapi.com
    - https://github.com/Zitronenjoghurt/world-data
    - https://crates.io/crates/nationify
- Add a Stop button for run away computations.
- Implement an LSP server
- Use library for exporting MathML
    - https://github.com/tmke8/math-core
    - https://github.com/katex-rs/katex-rs
- Use library for symbolic math
    - https://github.com/Nonanti/mathcore
- Answer questions on Mathematica StackExchange
    - https://mathematica.stackexchange.com/questions/274333/wolfram-engine-jupyter-stackrel-mathematica
- Mathematica2Jupyter https://github.com/divenex/mathematica2jupyter
- Add official parser as alternative option via compile time flag
    (Attention: Uses CMake to generate some Rust code)
    https://github.com/WolframResearch/codeparser/tree/master/crates/wolfram-parser
- Add a MCP server
- Use KaTeX to render Math output
- Investigate using a dedicated numeric / scientific library like
      - https://github.com/Axect/Peroxide
      - https://github.com/Apich-Organization/rssn
- Since Woxi is supposed to ran as WASM in the browser,
    it does not make sense to implement all features of Mathematica.
    Which functions in @functions.csv can/should not be implemented?

- Implement all
    https://github.com/Mathics3/mathics-core/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22New%20Builtin%20Function%20or%20Variable%22

- Should be able to run https://github.com/RuleBasedIntegration
- Integrate with https://zeppelin.apache.org/
- Related: https://github.com/anandijain/cas8.rs


## Prompts

Exactly follow these steps:
1. Pick one example from @all_mathics_tests.txt
2. Check if it works with `woxi eval` and exactly matches the output of `wolframscript -code`
3. If it already works, move it immediately to @done.txt
4. If it doesn't work yet, implement tests for it and then implement it to make the tests pass
5. Use https://reference.wolfram.com/language/ref/[Function-Name].html to check implementation details
6. Move example from @all_mathics_tests.txt to @done.txt
7. Commit the changes

---

Exactly follow these steps:
1. Pick the first example from @all_mathics_tests.txt
2. Use `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code \'WolframLanguageData[<Func-Name>, "DocumentationBasicExamples"]\'"}'`
    and `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code \'WolframLanguageData[<Func-Name>, "FunctionEssay"]\'"}'`
    to learn more about the function.
3. Check if it works with `woxi eval` and exactly matches the output of
      `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code [wl-code]"}'`
4. If it already works, remove it immediately from the file
5. If it doesn't work yet, implement tests for it and then implement it to make the tests pass
6. Remove the example from @all_mathics_tests.txt
7. Commit the changes
8. Go back to step 1 and follow the process again

---

Exactly follow these steps:
1. Run `duckdb -line -c "SELECT name FROM read_csv_auto('functions.csv') WHERE implementation_status IS NULL ORDER BY rank ASC LIMIT 1"`
2. Use `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code \'WolframLanguageData[<Func-Name>, "DocumentationBasicExamples"]\'"}'`
    and `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code \'WolframLanguageData[<Func-Name>, "FunctionEssay"]\'"}'`
    to learn more about the function.
3. Implement tests for the function
4. Implement the necessary code to make the tests pass
5. Mark the function as implemented in @functions.csv and add fill out other missing CSV fields
6. Commit the changes
7. Go back to step 1 and follow the process again

---

You are a software reliability engineer and your goal is to find any issues with the current Woxi implementation.

1. Use your internal knowledge about the Wolfram Language to come up with examples that might fail when executed with Woxi,
    even though Woxi should already support executing the code.
2. Before testing the code with Woxi, check that it really works with Wolfram Language like this:
    `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code [wl-code]"}'`
3. Implement tests for the function
4. Implement the necessary code to make the tests pass
5. Mark the function as implemented in @functions.csv and add fill out other missing CSV fields
6. Commit the changes


## Other Features

---

Add support for https://reference.wolfram.com/language/ref/ElementData.html
ElementData["Properties"]

-> Probably need to set up an EntityStore:
http://reference.wolfram.com/language/workflow/SetUpAnEntityStore.html

---

Change the playground / Jupyter rendering architecture to use the Boxes system of Wolfram Language.

So every expression gets converted into Boxes, and those Boxes are then converted to SVG to be displayed.
For Example:

```
In[]:= MakeBoxes[x^(1-3 + e^2)]
Out[]= SuperscriptBox[x,RowBox[{RowBox[{-,2}],+,SuperscriptBox[e,2]}]]
```

To re-convert it to something that is rendered graphically, one can use:

`RawBoxes[MakeBoxes[x^(1-3 + e^2)]]`

So internally, everything should be run through MakeBoxes[] and
RawBoxes[] then converts it to the final SVG that is displayed.

That would also allow us to exactly compare the graphical output of Woxi and wolframscript
without having to exactly compare the SVGs.

---

Add support for converting to PDFs:

$ woxi eval 'Export["out.pdf", (x^2 + 3)/7, "PDF"]'
$ wolframscript -code 'Export["out.pdf", (x^2 + 3)/7, "PDF"]'

Use https://github.com/typst/svg2pdf for it.

---

Gib alle komplexen Lösungen der Gleichung z⁶-(3+i)z³ +2+2i=0 an.
Mit trigometrischen Funktionen.


## Issues

---

Implement `?Plot`

---

Add support for `TraditionalForm[6 + 6 x^2 - 12 x]`
