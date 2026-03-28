# Ideas

- https://github.com/vanbaalon/wolfbook
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


Exactly follow these steps:
1. Pick the first example from @all_mathics_tests.txt
2. Use https://reference.wolfram.com/language/ref/[Function-Name].html to check implementation details
3. Check if it works with `woxi eval` and exactly matches the output of
      `curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code [wl-code]"}'`
4. If it already works, remove it immediately from the file
5. If it doesn't work yet, implement tests for it and then implement it to make the tests pass
6. Remove the example from @all_mathics_tests.txt
7. Commit the changes
8. Go back to step 1 and follow the process again


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


Use curl -s -X POST http://host.docker.internal:3456/exec -d '{"cmd":"wolframscript -code [wl-code]"}' to exeucte wolfram language code.


## Woxi Data Analysis Features

### High Priority — Core Data Analysis

#### 1. Data Import/Export

- `Import["file.csv"]` — Import CSV files as lists/associations
- `Import["file.tsv"]` — Import tab-separated files
- `Import["file.json"]` — Import JSON as associations/lists
- `Export["file.csv", data]` — Export to CSV
- `ReadString["file"]` — Read raw file contents
- `ReadList["file"]` — Read file as list of expressions

#### 2. Descriptive Statistics

- `Quantile[data, q]` — Quantiles/percentiles
- `Quartiles[data]` — Q1, Q2, Q3
- `InterquartileRange[data]` — Q3 - Q1
- `Covariance[x, y]` — Covariance between two datasets
- `Correlation[x, y]` — Pearson correlation coefficient
- `Skewness[data]` — Distribution asymmetry
- `Kurtosis[data]` — Distribution tail weight
- `MeanDeviation[data]` — Mean absolute deviation
- `MedianDeviation[data]` — Median absolute deviation
- `TrimmedMean[data, f]` — Mean after trimming fraction f

#### 3. GroupBy and Aggregation

- `GroupBy[data, f]` — Group rows by key function
- `CountsBy[data, f]` — Count elements by function result
- `Merge[assocs, f]` — Merge associations with conflict function

### Medium Priority — Transformation & Cleaning

#### 4. Missing Data Handling

- `Missing[]` / `Missing["reason"]` — Represent missing values
- `MissingQ[expr]` — Test if value is missing
- `DeleteMissing[data]` — Remove missing entries

#### 5. Binning and Histogramming

- `BinCounts[data, dx]` — Count elements per bin
- `BinLists[data, dx]` — Group elements into bins
- `HistogramList[data]` — Bin edges + counts

#### 6. Moving/Sliding Window Operations

- `MovingAverage[data, n]` — Simple moving average
- `MovingMedian[data, n]` — Moving median
- `MovingMap[f, data, n]` — Apply arbitrary function over sliding window

#### 7. Ranking

- `RankedMin[data, k]` — kth smallest element
- `RankedMax[data, k]` — kth largest element

### Lower Priority — Modeling & Visualization

#### 8. Curve Fitting / Regression

- `Fit[data, basis, var]` — Least-squares polynomial fit
- `LinearModelFit[data, basis, vars]` — Linear regression with diagnostics
- `FindFit[data, model, params, var]` — Nonlinear least-squares fitting
- `Interpolation[data]` — Interpolating function from data points

#### 9. Additional Plotting

- `Histogram[data]` — Histogram visualization
- `ListLinePlot[data]` — Connected scatter plot
- `BarChart[data]` — Bar chart
- `BoxWhiskerChart[data]` — Box-and-whisker plot
- `MatrixPlot[matrix]` — Heatmap of 2D data
- `PieChart[data]` — Pie chart

#### 10. Text/String Analysis

- `StringCases[str, pattern]` — Extract all pattern matches
- `StringCount[str, pattern]` — Count pattern occurrences
- `TextWords[text]` — Split into words
- `WordCounts[text]` — Word frequency association

### Recommended Implementation Order

1. `Import`/`Export` for CSV and JSON
2. `GroupBy`
3. `Quantile`, `Quartiles`, `Correlation`, `Covariance`
4. `Missing`/`MissingQ`/`DeleteMissing`
5. `MovingAverage`/`MovingMap`
6. `BinCounts`/`BinLists`
7. `Histogram`/`ListLinePlot`/`BarChart`
8. `Fit`/`LinearModelFit`
9. Remaining statistics (`Skewness`, `Kurtosis`, etc.)
10. Text analysis functions


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

---

This should show an error in the playground, as `pts` is not defined:

Graphics[{Orange, Point[pts]}]

