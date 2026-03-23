# Missing Features Investigation

**Date**: 2025-03-23
**Status**: 1,994 of 6,184 Wolfram Language functions implemented (~32%)

This document identifies the most impactful missing features for making Woxi
usable by end users, organized by priority.

---

## Priority 1: Critical for End-User Usability

These are features that users will immediately notice are missing and that block
common workflows.

### 1.1 Error Messages & Diagnostics

Currently only ~80 `emit_message()` calls across the codebase. Mathematica has
thousands of structured messages. When something goes wrong, users often get
unevaluated expressions back with no explanation.

**What's needed:**
- Structured `Message[symbol::tag, args...]` system
- Usage messages for all implemented functions (`f::usage = "..."`)
- `?FunctionName` should display function documentation, not just definitions
- Better error messages for type mismatches and wrong argument counts
- Warnings when functions are called with unsupported option combinations

### 1.2 Trace & Debugging

- `Trace[expr]` - Show evaluation steps (currently unimplemented, rank 2)
- `TracePrint[expr]` - Print trace during evaluation
- `Stack[]` - Show evaluation stack
- `AbortProtect[expr]` - Protect critical sections

These are essential for users debugging their code.

### 1.3 Interactive Features (Dynamic/Manipulate)

Completely missing. These are flagship Mathematica features for exploration:

- `Manipulate[expr, {x, min, max}]` - Interactive parameter exploration
- `Dynamic[expr]` - Auto-updating expressions
- `Slider[]`, `Button[]`, `Checkbox[]` - UI widgets
- `DynamicModule[{vars}, body]` - Scoped dynamic content

**Recommendation:** Even a basic `Manipulate` with sliders in the Playground
would be a huge differentiator.

### 1.4 Missing Common Functions (High-Rank, Unimplemented)

These are frequently-used functions (rank <= 3) that are still missing:

| Function | Description | Impact |
|----------|-------------|--------|
| `Trace` | Evaluation tracing | Debugging |
| `DeleteDirectory` | Remove directories | File I/O |
| `FileByteCount` | File size | File I/O |
| `FileDate` | File timestamps | File I/O |
| `ParentDirectory` | Parent dir path | File I/O |
| `TimeUsed` | CPU time measurement | Profiling |
| `TimeZone` | Timezone support | DateTime |
| `AbortProtect` | Protect from abort | Robustness |
| `GroebnerBasis` | Polynomial ideal basis | Algebra |
| `ParametricPlot3D` | 3D parametric curves | Plotting |
| `PaddedForm` | Number padding | Formatting |
| `NumberFormat` | Number format control | Formatting |
| `SetAccuracy` | Set numeric accuracy | Numerics |
| `Shallow` | Truncated output display | Display |
| `DumpSave` | Save definitions to file | Persistence |

---

## Priority 2: Important for Practical Use

### 2.1 Package System

Currently `BeginPackage`, `EndPackage`, `Begin`, `End` are all no-ops.
Users cannot:
- Create reusable packages with proper namespacing
- Use `Needs["pkg`"]` to load actual packages
- Have private vs public symbols in packages

### 2.2 Data Import/Export Gaps

Import/Export exists but key formats are incomplete:

| Format | Import | Export | Notes |
|--------|--------|--------|-------|
| CSV | Partial | Missing | `ImportString` works, `Import["file.csv"]` needs work |
| JSON | Missing | Missing | Critical for web/API workflows |
| TSV | Missing | Missing | Common data format |
| XLSX | Missing | Missing | Spreadsheet data |
| HDF5 | Missing | Missing | Scientific data |

### 2.3 Missing Chart Types

Only basic `Plot`, `ListPlot`, `ListLinePlot` exist. Missing:

- `Histogram[data]` - Frequency distribution
- `BarChart[data]` - Categorical data
- `PieChart[data]` - Proportional data
- `BoxWhiskerChart[data]` - Statistical distribution
- `MatrixPlot[matrix]` - Heatmap visualization
- `ListDensityPlot` - 2D density from data
- `DateListPlot` - Time series

### 2.4 Missing Statistics Functions

Some gaps in statistics that data analysts need:

- `Covariance[x, y]` - Listed in todos as needed
- `Skewness[data]` - Distribution shape
- `Kurtosis[data]` - Distribution tails
- `LinearModelFit` - Regression with diagnostics
- `BinCounts` / `BinLists` - Binning operations
- `MovingMedian` - Sliding window median
- `SurvivalFunction` - Survival analysis

### 2.5 3D Graphics

`Plot3D` is dispatched but many 3D features are limited:

- `ParametricPlot3D` - Unimplemented (rank 2)
- `Graphics3D` primitives need more options
- `ContourPlot3D` - 3D isosurfaces
- `RegionPlot3D` - 3D region visualization
- Lighting, viewpoint, and material options are sparse

### 2.6 TraditionalForm & Display

- `TraditionalForm` - Math notation display (explicitly noted in todos.md)
- `MakeBoxes` / `RawBoxes` rendering architecture (noted in todos.md)
- `CForm`, `FortranForm` - Code generation forms
- Better 2D typesetting (fractions, superscripts rendered properly)

**Recommendation from todos.md:** Adopt a Boxes-based rendering pipeline
(`MakeBoxes` -> `RawBoxes` -> SVG) for consistent, comparable output.

---

## Priority 3: Nice to Have for Power Users

### 3.1 Advanced Algebra

- `GroebnerBasis` - Polynomial ideal computations (rank 2)
- `Reduce` improvements - More complete equation reduction
- `Assuming` improvements - Persistent `$Assumptions` context
- `FunctionExpand` - Expand special functions
- `ComplexExpand` - Expand assuming real variables

### 3.2 Advanced Calculus

- Complex contour integration
- Multivariable integral improvements
- Better symbolic integration (possibly via RuleBasedIntegration, noted in todos.md)
- `AsymptoticExpand` / `AsymptoticSolve` improvements
- `LaplaceTransform` / `InverseLaplaceTransform` robustness

### 3.3 Numerical Methods

- `LinearProgramming` - Linear optimization
- `FindMinimum` improvements with more methods
- `NDSolve` for systems of ODEs
- `NIntegrate` with more integration strategies
- `SetPrecision` / `SetAccuracy` for precision control
- `WorkingPrecision` option support

### 3.4 String & Text Processing

Most string functions are implemented, but missing:
- `TextWords[text]` - Word tokenization
- `WordCounts[text]` - Word frequency
- `StringTemplate` - Template strings
- `Interpreter["type"]` - Semantic string interpretation

### 3.5 Developer/IDE Features

From todos.md:
- **LSP Server** - Language Server Protocol for editor integration
- **Tab Completion** - In Playground and Jupyter
- **MCP Server** - Model Context Protocol integration
- **Stop Button** - For runaway computations in Playground

### 3.6 Clebsch-Gordan & Physics

- `ClebschGordan` - Angular momentum coupling (rank 2)
- `SixJSymbol` - Wigner 6j symbols (rank 2)
- `ThreeJSymbol` - Wigner 3j symbols

### 3.7 Entity Data

- `ElementData` - Chemical element properties (noted in todos.md)
- Entity/EntityStore framework for structured data access

---

## Priority 4: Out of Scope (Browser/WASM Constraints)

These are explicitly not practical for Woxi's browser-based WASM target:

- Machine Learning (`NetTrain`, `Classify`, `Predict`)
- Cloud/Web Integration (`CloudDeploy`, `APIFunction`)
- Audio processing (`Play`, `ListPlay`, `Sound`)
- External program linking (`Install`, `Uninstall`, `WSTP`)
- Notebook manipulation (`CellObject`, `NotebookWrite`)
- System administration (`SystemInstall`, kernel management)
- Database connectivity (`DatabaseLink`)
- Geographic computing (`GeoGraphics`, `GeoDistance`)

---

## Summary: Top 20 Most Impactful Items

| # | Feature | Category | Effort |
|---|---------|----------|--------|
| 1 | Structured error messages & usage strings | UX | Large |
| 2 | `Manipulate[]` with sliders in Playground | Interactive | Large |
| 3 | `Trace[]` / debugging tools | Debugging | Medium |
| 4 | JSON import/export | Data I/O | Small |
| 5 | `Histogram` / `BarChart` / `PieChart` | Plotting | Medium |
| 6 | `TraditionalForm` / math typesetting | Display | Medium |
| 7 | Tab completion in Playground & Jupyter | UX | Medium |
| 8 | CSV import improvements (`Import["file.csv"]`) | Data I/O | Small |
| 9 | `ParametricPlot3D` | Plotting | Medium |
| 10 | Package system (`BeginPackage`/`EndPackage`) | Language | Large |
| 11 | `GroebnerBasis` | Algebra | Large |
| 12 | `LinearModelFit` / regression diagnostics | Statistics | Medium |
| 13 | `?FunctionName` documentation display | UX | Medium |
| 14 | File system functions (`DeleteDirectory`, `FileDate`, etc.) | File I/O | Small |
| 15 | Stop button for Playground | UX | Small |
| 16 | `Shallow[]` for truncated output | Display | Small |
| 17 | LSP server | Developer | Large |
| 18 | `TimeZone` / `TimeUsed` | Utilities | Small |
| 19 | MakeBoxes rendering pipeline | Architecture | Large |
| 20 | Precision control (`SetAccuracy`, `WorkingPrecision`) | Numerics | Medium |

---

## Methodology

This investigation examined:
- `functions.csv` (6,184 tracked functions, 1,994 implemented, 2,194 explicitly unimplemented)
- All source files in `src/evaluator/dispatch/` and `src/functions/`
- Test coverage across `tests/interpreter_tests/`, `tests/scripts/`, `tests/cli/`
- Existing `todos.md` roadmap items
- Parser and syntax support in `src/parser/`
- End-user interfaces (Playground, JupyterLite kernel)
