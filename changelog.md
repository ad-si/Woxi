# Changelog

# Unreleased

# 2026-11-08 - 0.4.0

Version 0.4.0 improves package and notebook compatibility: contexts keep
package definitions separate, Rubi loads unmodified, and the WLJS Notebook
backend gets past module loading. Symbolic algebra gains complete
multivariate factorization and more reliable equation solving. Woxi Studio
handles more typeset input and interactive controls, while a new language
server brings diagnostics, navigation, formatting and code actions to editors.

## Woxi Studio & Wolfram Demonstrations

- `Manipulate` supports colour controls with an initial colour but no range
    or choice list. Scalar and `Slider2D` bounds can depend on other controls,
    including through `Dynamic` wrappers.
- Explicit `RadioButtonBar` controls keep their button layout. A
    `PopupMenu[Dynamic[var], choices]` inside a body becomes an interactive
    control, with choices resolved through surrounding scopes and refreshed
    when dependent controls change.
- Control labels and choices typeset `Dynamic`, `Grid`, `TableForm`,
    `TextGrid`, `Framed`, `Labeled`, radicals and arithmetic over layout
    expressions. Intentionally blank labels stay blank.
- Notebook input reads typeset definite and indefinite integrals,
    `DifferentialD`, `Piecewise` grids, inline box escapes and quoted strings.
    Pasted `InputForm` results inside `InterpretationBox` are interpreted as
    their values rather than opaque display wrappers.
- Prose cells preserve grouping in fractions, omit empty subscript bases,
    and render overscripts, accents and prefix scripts. Invisible script
    bases and non-printing named characters no longer appear as text.
- Derivatives display as prime marks in StandardForm and TraditionalForm,
    with explicit orders for higher and multivariate derivatives.
    TraditionalForm renders standalone negative powers as fractions.
- All previously unreadable or unrunnable cells in Wolfram's 41-notebook
    *Introduction to Calculus* ebook now read and run. Some computations
    still remain less fully evaluated than in Wolfram.

## Plotting & graphics

- `ListPlot` accepts per-series `Joined` flags. `ParametricPlot` handles
    lists of conditional curves whose hidden entries evaluate to `{}`.
- `ContourPlot` and `DensityPlot` honour `RegionFunction`, `FrameLabel`
    and `Epilog`. `ContourStyle` reaches both function and equation forms
    and survives composition with `Show`.
- A flat `PlotRange -> {min, max}` applies to every axis. Plot options such
    as `ImageSize -> 400 {1, 1}` are evaluated before use, and 3D camera
    options accept `Dynamic` wrappers around `ViewPoint` and `ViewAngle`.
- Visual hosts display the graphic returned by an expression, rather than
    the last graphic drawn during evaluation. A `Tooltip` around a graphic
    inside a layout draws its content, and styling a layout no longer
    replaces embedded graphics with placeholders.
- Plot and control labels preserve typesetting wrappers, nested
    `NumberForm`, font families and explicit font sizes. `Style` sizes and
    `Spacer` widths use printer's points; explicit sizes override named
    styles. Negative real coefficients print as subtraction.
- `PieChart` supports `LabelingFunction`, structured labels and radial
    placement, including outside callouts with leader lines. `Inset` anchors
    can mix numeric and symbolic coordinates, such as `{0.8, Center}`.
- `Disk[c, Scaled[r]]` scales its radius by the plot width. `Graphics3D`
    draws `InfiniteLine`, `HalfLine`, `InfinitePlane` and `HalfPlane`, clipped
    to the existing picture range without enlarging it. Lists of sphere
    centres and `Ball` primitives render correctly.
- 3D `Scale` acts about the object's bounding-box centre and supports
    unequal scale factors for curved primitives. `EdgeForm` colours face
    outlines and curved primitives' end circles, retaining opaque outlines
    around transparent faces.
- `RegionPlot3D` retains a symbolic `GraphicsComplex`, so its surfaces can
    be extracted, transformed and combined with other 3D primitives.
- `PolyhedronData` adds `RhombicHexecontahedron` and
    `GreatStellatedDodecahedron`, with `MathematicaSpikey` as an alias for
    the former. The rhombic hexecontahedron includes exact measures and its
    62 vertices, 120 edges and 60 faces. Polyhedron truncation follows
    Wolfram's edge order.
- Graphs honour callable, named, per-edge and `None` forms of
    `EdgeShapeFunction`, passing the edge object to the callback and keeping
    directives scoped to that edge. `GraphPlot` accepts
    `Method -> "CircularEmbedding"`; an explicit `GraphLayout` takes priority.

## Calculus, algebra & equation solving

- `Factor` completely factors multivariate polynomials, including repeated
    factors and non-polynomial atoms treated as variables. For example,
    `Factor[Expand[(a + b + c + d)^3]]` returns `(a + b + c + d)^3`.
    Large coefficients no longer cause unbounded trial division or premature
    failure of factor lifting.
- Rubi, the rule-based integrator, loads unmodified and integrates. The
    [Rubi guide](tests/cli/rubi.md) describes setup and remaining limitations.
- `Solve` isolates exponential, logarithmic and trigonometric terms even
    when they appear inside quotients or sums. Polynomial systems use
    elimination and resultants, preserving distinct and repeated roots.
    Radical equations are checked against the original equation so that
    extraneous roots, such as the candidate for `Sqrt[x] == -1`, are rejected.
- `Minimize` and `Maximize` retain real critical points obtained from
    periodic solution families. `FindMinimum` and `FindMaximum` accept
    bounded variable specifications in both single- and multivariable forms.
- `FindRoot` evaluates held equations before inserting trial values,
    accepts singleton-list residuals, and tracks the best multivariate
    iterate rather than exhausting iterations after residual improvement
    stops.
- `DSolve` handles separable first-order initial-value problems, selecting
    the branch fixed by the initial condition. The corresponding form
    without an initial condition still remains unevaluated.
- Integer `Reduce` canonicalizes linear Diophantine results, parameter
    membership and variable order. Integer `FindInstance` searches systems
    with three or more variables.
- `RootReduce` and `MinimalPolynomial` keep rational polynomials in one
    algebraic root inside that root's field, avoiding repeated resultant
    degree growth. Remaining resultants use big-integer determinants and
    interpolation, preventing intermediate `i128` overflow.
- `Resultant` expands its output. `Coefficient` recognizes structural
    quotients, numeric ordering compares values across number types, and
    negative unit rational factors distribute their sign into sums where
    Wolfram does.
- `DifferenceDelta` and `DifferenceQuotient` of cosine use the correct
    phase. Parentheses preserve nested binary `Cross` products instead of
    flattening them into a higher-arity call.
- `Interpolation` accepts flat `{x, y, z}` data triples, reconstructing a
    2D grid with interpolation orders 1–3 and per-axis order settings.

## Special functions, statistics & number theory

- `FunctionExpand` reduces incomplete Gamma functions of positive integer
    order to elementary expressions. Orders requiring `ExpIntegralEi` or
    `Erf` remain unchanged.
- Inverse trigonometric functions applied to their forward functions at
    exact real arguments reduce into the principal range; for example,
    `ArcSin[Sin[2]]` is `Pi - 2`.
- `LinearModelFit` exposes `ParameterTableEntries`, `ParameterErrors`,
    `ParameterTStatistics` and `ParameterPValues`, using the OLS covariance
    and Student-t distribution.
- Constants stored in variables, including `Pi`, `E` and `Degree`, reach
    arbitrary-precision evaluation and `RealDigits` just as literal
    constants do.
- `GeometricTest` treats concurrency projectively, including parallel
    lines, and checks congruence and similarity with the supplied vertex
    correspondence for arbitrary polygons.

## Lists, arrays & structured data

- `Table` and `Array` splice `Sequence` and `Splice` results and remove
    `Nothing`, sharing the same list-construction rules as ordinary lists.
- `ParallelSelect` and `ParallelCases` support the forms of `Select` and
    `Cases`, including count and level specifications. As with Woxi's other
    parallel combinators, evaluation delegates to the sequential operations.
- `Length` handles operator forms through their canonical structure: rules
    have length 2, while complex numbers remain atoms of length 0. `FreeQ`
    looks inside rules, comparisons and pure functions.

## Strings, dates, import & export

- Named characters cover Wolfram's full table of 1,100 names, including
    names containing digits, long arrows, pictographs, musical accidentals
    and astronomical symbols. Six-digit `\|HHHHHH` string escapes decode
    characters outside the Basic Multilingual Plane in source and notebook
    input.
- `DeleteStopwords` preserves surrounding whitespace and punctuation and
    reports invalid argument types. `DateObject[date, granularity]` retags an
    existing date; `MidDate` returns exact integers for whole-second midpoints.
- `FileNameJoin` normalizes a single path string, and `FileNameSplit`
    preserves interior empty segments. `FileNames` accepts nested lists of
    string patterns, alternatives and regular expressions, using Wolfram's
    string-pattern wildcard semantics.
- Recursive `FileNames` searches support integer depths, `Infinity`,
    exact levels `{n}` and ranges `{n1, n2}`. Searches below a relative
    directory retain relative paths instead of returning only base names.
- File operations consistently resolve relative paths against
    `SetDirectory`, including imports, exports, streams, searches, metadata
    queries and file or directory mutations.
- File names beginning with `!` open command pipes. Reading streams advance
    as the command produces lines; writing streams feed the command's stdin,
    and `Close` ends input or stops a still-running reader. File-taking
    functions, including `Import`, `Get`, `Put` and `PutAppend`, accept these
    command forms too.
- `WriteLine` writes a string plus a newline to the same channels as
    `WriteString`, including files, pipes and standard output or error.
    `InputString` and `Input` read stdin when the CLI owns it and return
    `EndOfFile` when exhausted; embedders without stdin retain that result.
- `Get` returns the evaluated expression itself, preserving strings and
    `Null` instead of reparsing display text. `$InputFileName` and
    ``System`Private`$InputFileName`` identify the file currently being read,
    with the previous value restored after nested reads.
- Explicit import format names select the format rather than an element:
    `Import["data.csv", "CSV"]` behaves like `Import["data.csv"]`, while
    `{"CSV", "ColumnLabels"}` selects the column-label element.
- `PacletDirectoryLoad` and `PacletDirectoryUnload` manage paclet search
    directories. `Needs`, `Get` and `FindFile` resolve context names through
    paclet Kernel extensions and `$Path`; `Needs` skips loaded contexts and
    validates its arguments.
- Message templates fill unnumbered argument slots as well as numbered
    ones. Held trailing semicolons retain their empty statement without
    printing an extra `Null`, and pattern tests print with the parentheses
    needed to read them back correctly.
- The REPL uses OutputForm precision and notation while preserving full
    values in `%` and `Out[]`. `woxi eval` retains round-trip InputForm.

## Images & audio

- `Image` validates channel counts against its colour space rather than
    silently reinterpreting the data. Gaussian noise in `ImageEffect` is no
    longer clipped to `[0, 1]`, avoiding bias at the endpoints.
- `Play` accepts options and synthesizes at the requested `SampleRate`.
    Sounds inside `Grid`, `Column` and `Row` display a play button and
    waveform in visual hosts instead of their source expression.

## Robustness & conformance

- Symbols resolve against `$Context` and `$ContextPath` when input is read
    and retain their full names. Package-private helpers no longer collide,
    explicitly qualified global symbols stay global, and exported names
    become visible through `EndPackage`.
- Context-aware operations include `Context`, `Contexts`, `Names`,
    `Remove`, `Symbol`, `ToExpression`, `Definition`, `DownValues` and
    `Information`. Symbols print with enough qualification to round-trip;
    shadowing and invalid symbol queries report the appropriate messages.
    `Information[sym, "Property"]` returns individual properties.
- `Block` localizes definitions, options, messages and attributes as well
    as own values, then restores them and reevaluates its result. `ClearAll`
    uses the same removal logic and also clears messages.
- Tagged definitions recognize tags through pattern wrappers. Assignments
    consult upvalues on both sides, and arithmetic upvalues apply to
    `Minus`, `Subtract`, `Divide` and their operator spellings.
- Conditional definitions retain guards inside `With`, `Module` and
    `Block`, coexist with fallback rules, and no longer gain incorrect
    literal-rule precedence. Compound patterns retain their guards and
    bind variables inside `Rule` and `RuleDelayed` nodes.
- Orderless matching retries alternative argument bindings when a guard
    rejects the first match. Flat matching prefers bindings consistent with
    other arguments, and package-local head patterns substitute correctly.
- Anonymous head-restricted blanks no longer panic, distinct anonymous
    blanks remain distinct, and optional parameters are retained when
    definitions are stored. Defaults and pattern tests accept general
    self-delimiting expressions, not only named blanks.
- Structural replacement preserves grouping, held arguments, pattern names
    and head restrictions, and descends through application operators.
    `ReplacePart` reconstructs rule nodes correctly. `DownValues` accepts
    either rule spelling, and `DownValues[f] = {}` clears definitions.
- `MemberQ` supports `Heads` with level specifications, matches string
    elements correctly, and recognizes `Except` patterns. Held subtraction
    and division match their canonical `Plus` and `Times` forms.
- `Span` has the correct precedence, permits omitted operands after an
    operator, and supports part extraction from a parenthesized span.
    Trailing spans no longer absorb the next statement. Implicit products
    yield correctly to tighter application operators, and comments or
    whitespace between semicolons preserve empty statements.
- `Composition` respects a holding outer function, and `ToExpression`
    accepts compound wrapper heads and preserves held trailing `Null`.
    Sequence-bound pattern variables splice during substitution, including
    into local-variable lists.
- Loading `CodeParser` installs its context and symbols. Concrete syntax
    trees group bracketed tokens so internal newlines are not mistaken for
    top-level statement boundaries.
- `HoldCompleteForm` preserves fully held display expressions. `TracePrint`
    emits indented, optionally filtered evaluation steps, and tracing respects
    hold attributes. `TraceScan` prints rebuilt lists normally. `NCache`
    remains inert, retaining both its exact and approximate values.
- Storing definitions no longer keeps the definition table borrowed during
    comparisons that may read it again, avoiding a reentrant-access panic.
    `$VersionNumber` is the real-valued Wolfram compatibility version (`15.`);
    `$Version` continues to identify the Woxi build.

## Platform, performance & tooling

- `woxi lsp` provides syntax and unsupported-symbol diagnostics, hover,
    completion, definition and reference navigation, occurrence highlighting
    and document outlines. Its forgiving tokenizer handles incomplete input,
    comments, named slots, output references and leading-decimal numbers.
- The language server adds semantic highlighting, formatting that preserves
    tokens and line breaks, range formatting, spelling diagnostics and code
    actions, including fixing all occurrences of a misspelled builtin.
    User-defined names and pattern bindings are excluded from spelling hints.
- The JupyterLite kernel extension targets JupyterLite 0.8 instead of the
    removed 0.5 server plugin API.
- Python `evaluate_expr` no longer segfaults on deeply nested trees in
    unoptimized builds; it converts them or raises `RecursionError` beyond
    the 5,000-level limit.
- Python and WebAssembly builds explicitly declare the bundled `CityData`
    loader's Serde dependency and derive support. CI checks native and
    WebAssembly library builds independently of CLI features.
- Symbolic `Dot` avoids repeatedly flattening a growing sum. Rational-root
    searches cache divisor lists and use 64-bit trial division when possible.
    Dense scenes of thin cylinders use a scene-aware tessellation budget
    instead of generating hundreds of thousands of barely visible faces.
- Dependencies were refreshed across Rust, npm, Nix and GitHub Actions,
    including rand 0.10, jupyter-protocol 2, runtimelib 3, zeromq 0.6,
    rustyline 19, pyo3 0.30, criterion 0.8 and CodeMirror 6.x. resvg remains
    on 0.45 and num-bigint on 0.4 for compatibility; pest remains on 2.8
    because 2.9's per-rule stack probes slowed evaluation.
- Release checks keep package, kernel and macOS bundle versions aligned.
    The [release checklist](docs/releasing.md) covers dependency-crate
    publication, separate npm publication and verification of release assets.
    Windows graphics tests use explicit RGB byte types, and Clippy findings
    in evaluator helpers and regression tests are resolved.

# 2026-08-06 - 0.3.0

Between 0.2.0 and 0.3.0 the focus moved from growing the function library
to running real notebooks: Woxi Studio executes complete Wolfram
Demonstrations end to end, the plotters draw what wolframscript draws, and
hundreds of conformance and robustness fixes — many found by differential
fuzzing against wolframscript — closed gaps between the two engines.
Windows became a supported platform with prebuilt binaries. The list only
includes the most prominent changes; the Demonstration-driven fixes are
described in detail in the last section.

## Woxi Studio & Wolfram Demonstrations

- Woxi Studio opens and runs complete Wolfram Demonstration notebooks end
    to end — including definition notebooks and notebooks with embedded
    raster photos — and re-instantiates stored `Manipulate` widgets when a
    notebook is reopened. Demonstrations such as Kepler's Second Law,
    Parabolic Mirror, Doyle Spirals, Dedekind Cut, Damped Forced Pendulum
    and Merging Schools of Fish serve as end-to-end tests.
- `Manipulate` covers the Demonstrations control vocabulary: interactive
    `Locator`s (with `LocatorAutoCreate`), `Trigger`, `Button`,
    `ButtonBar`, `SetterBar`, `PopupMenu`, `Checkbox`/`CheckboxBar`,
    `Toggler`, `Opener`/`OpenerBar`, controls nested in `TabView`,
    `PaneSelector` or `Grid` layouts, per-control `ControlType` lists,
    dynamic control bounds, controls depending on other controls,
    `TrackedSymbols`, `Animate` bodies and `AnimationRunning`.
- Typeset notebook input is read back as code: inline box syntax, typeset
    `Sum`/`Product` boxes, the partial-derivative operator, `Part` written
    as a bracketed subscript, the radical prefix operators, and the `×`
    and `÷` characters.

## Plotting & graphics

- List plots grew `Around` error bars, `IntervalMarkers -> "Bands"`,
    `Callout` and `Labeled` wrappers, `PlotMarkers`, `Epilog`, `Filling`
    (including between datasets), `FillingStyle`, `DataRange`,
    `InterpolationOrder`, `Mesh -> Full`, `PlotLayout`, association keys
    as point labels, and `TimeSeries` input; `ComplexListPlot` and
    `ListLinePlot3D` are new.
- `Plot` supports `Background`, `Evaluated`, `EvaluationMonitor` and
    `Infinity` endpoints; automatic ticks match Wolfram's algorithm, tick
    labels switch to scientific notation outside `[10^-5, 10^6)`, frames
    take `FrameLabel` captions and `AxesLabel` sits at the end of its
    axis. Contour and density plots fix the marching-squares case table,
    draw their mesh, and keep their shading through `Show`.
- Graphics primitives and options: polygons with holes (and their
    measures), `Arrowheads`, `Tube`, `CapForm`, `Text` offsets,
    `Translate` and `Scale` in SVG export, `Framed`/`Highlighted`
    rendering, and `AspectRatio` applied to the plotting area rather than
    the whole image.
- Graphics3D draws curved surfaces without facet edges, triangulates
    concave polygons, honours `EdgeForm[]`, `SphericalRegion` and
    `ViewAngle`, and outlines flat faces the way Wolfram does.
    `PolyhedronData` gained face lists with Wolfram's vertex order and
    the icosahedral Archimedean solids.
- Exported SVG embeds the fonts it uses, renders `TableForm`/`MatrixForm`
    and the display-form wrappers as grids, and shows machine reals at
    six significant figures.

## Calculus, algebra & equation solving

- `Integrate` learned the antiderivatives of the error, Fresnel, inverse
    hyperbolic and exponential-integral families, `Sin`/`Cos[x^2]` to
    Fresnel integrals, `Exp[a x^2]` to `Erfi`, `1/Log[x]` to
    `LogIntegral`, trigonometric integrals over a linear term to
    `Si`/`Ci`, `u = x^n` substitution, Euler's log-trig definite
    integrals and `Abs` of a linear argument. `NIntegrate` handles
    endpoint singularities, multi-segment ranges and iterated
    multi-dimensional integrals (`NSum`/`NProduct` likewise).
- Differentiation covers `Piecewise`, `HeavisideTheta` (to `DiracDelta`,
    with its scaling law), `KroneckerDelta`, `Floor`/`Ceiling`, argument
    derivatives of the Bessel and Hankel functions, `PolyLog`,
    `ExpIntegralE`, the Airy primes and the incomplete elliptic
    integrals, plus the fractional `CaputoD`.
- `Series` expands `Gamma` at its pole, `Zeta` at 1 (Laurent) and
    algebraic expressions at `Infinity`; limits are decided from the
    leading exponent at infinity, and `MaxLimit`/`MinLimit` handle
    bounded trigonometric oscillations.
- Solving: `x^n == c` in radicals, polynomials with machine-real
    coefficients, systems over a modulus, `Modulus` and `MaxRoots`
    options, `NSolve` domain argument, `NSolveValues`, `SolveValues`
    result shaping, and `Reduce` over higher-degree polynomial
    inequalities. `DSolve` handles non-constant forcing and orders its
    fundamental pairs; `NDSolve` solves systems and constrained ODEs;
    `RSolve` solves the logistic map at `r = 4` and golden-ratio
    recurrences in the Fibonacci/Lucas basis.
- Optimization: `LinearProgramming` (exact simplex solver with variable
    bounds), constrained `FindMinimum`/`FindMaximum`/`NArgMin`/`NArgMax`,
    the domain argument, `NonlinearModelFit`, constrained and weighted
    `Fit`, and the minimum-norm fit for rank-deficient designs.
- Structural algebra: `RootReduce` and `RootApproximant`, `Root` objects
    treated as the numbers they are, polynomial reduction modulo
    polynomials, hyperbolic support in `TrigReduce`/`TrigFactor`, and
    many canonical-ordering fixes so sums, products and radicals print
    exactly as wolframscript prints them.

## Special functions, statistics & number theory

- Symbolic reductions for the hypergeometric families (`0F1`, `1F1`,
    `HypergeometricU`, `HypergeometricPFQ` Bessel forms), `PolyLog`,
    `LerchPhi`, the incomplete `Beta` and `Gamma` functions,
    `GegenbauerC[n, 1/2, x]`, parity of `Gudermannian`, `Haversine`,
    `Sinc`, `Erf` and the Fresnel functions, hyperbolic
    imaginary-period shifts, and exact values of the Carlson elliptic
    integrals. Factorial Taylor sums fold to `Sin`/`Cos`/`Sinh`/`Cosh`,
    `Sum[1/n^s]` gives `Zeta[s]`, and sums of `HarmonicNumber` give
    hyperharmonic closed forms.
- Numeric evaluation of the Coulomb wavefunctions at nonzero `eta`,
    `CarlsonRF` at complex arguments, two-argument `Erf`, `PolyLog` with
    real order, and inverse trigonometric functions at complex floats.
- Astronomy: `SunPosition`, `MoonPosition`, `MoonPhase`, eclipses and
    related functions compute from platform-independent ephemerides, as
    an observer on the ground sees them.
- Distributions: `SkewNormalDistribution`; `CDF`/`Quantile`/`Median` for
    the discrete families (`NegativeBinomial`, `Pascal`, `BetaBinomial`,
    `Zipf`, `Benford`, …); skewness, kurtosis, raw moments,
    characteristic functions and MGFs for many more; moments of
    `TruncatedDistribution` and `CensoredDistribution`; `RandomVariate`
    sampling for `BinomialDistribution` and hand-written distributions;
    and parameter-range validation instead of silent nonsense.

## Lists, arrays & structured data

- `SparseArray` behaves like the array it stores: arithmetic stays
    sparse, `Part` reaches into higher ranks, the structural list
    operations densify transparently, and the array predicates and grid
    queries see through it.
- `Dataset` answers queries through the `Query` engine instead of a few
    fixed shapes, and the statistics functions see through it.
    `TimeSeries` can be transformed, combined, resampled, windowed and
    rescaled; `MovingMap` supports padded and time-windowed forms.
- List operations rounded out: `Periodic`/`Reflected`/`Cyclic` padding
    (with nested padding arrays), two-dimensional `ListConvolve` and
    `ListCorrelate` with generalized operations, `MapAt` with `All` and
    `Span`, ordering functions in `SortBy`/`KeySort`, nested `GroupBy`
    classifiers, rule-based `StringSplit`/`SequenceSplit`, and the
    structured matrices (`CauchyMatrix`, `BlockDiagonalMatrix`,
    `CompanionMatrix`) plus `Symmetrize`/`SymmetrizedArray` and
    `TensorExpand`.

## Strings, dates, import & export

- XML imports to symbolic XML (namespaces included) and writes back out
    the way wolframscript does; CSV fixes cell quoting, empty fields and
    header inference; JSON export fails cleanly on unrepresentable
    values; `URLBuild`, `URLQueryEncode`/`URLQueryDecode`, `TextCases`
    for words and sentences, `CharacterName` and `CharacterNormalize`
    are new.
- Dates: `MaxDate`/`MinDate`, ISO and week-based date elements, exact
    rational `DateDifference` sub-day units, month rollover in
    `AbsoluteTime`, and `DateObject` granularity truncation.
- Output forms: `TeXForm` sets matrices as LaTeX arrays and matches
    wolframscript, `CForm`/`FortranForm` fix their operators and
    numbers, and the `NumberForm` display family shares one
    option-aware renderer (`NumberSigns`, `ExponentFunction`,
    `NumberPadding`, decimal-point alignment, `BaseForm` inner values).
- Patterns: `Verbatim` heads, `Longest`/`Shortest`/
    `OrderlessPatternSequence`, `Overlaps -> All` reporting every match
    at every start, string-pattern back-references, and implicitly named
    blanks in `StringSplit`.

## Images & audio

- Image processing: `ImagePad`, `ImageCrop`, `ImageFilter`,
    `MeanFilter`, `ImageDifference`, `ImageMeasurements`,
    `MorphologicalComponents`, `ComponentMeasurements`,
    `DeleteSmallComponents` and `MaxDetect`/`MinDetect` on matrices;
    `Blur` is the `GaussianFilter` it claims to be, `Sharpen` sharpens
    by Wolfram's amount, and `EdgeDetect` runs on the gradient Woxi
    already had.
- Audio: the accessors, `AudioNormalize`/`AudioReverse`/`AudioPad`, and
    `ListPlay` with playback in the Playground and Woxi Studio.

## Robustness & conformance

- Several rounds of differential fuzzing against wolframscript fixed
    dozens of silent divergences in arithmetic, ordering, `Simplify`,
    `Together` and friends.
- Edge cases that used to abort the evaluator or panic now return the
    message Wolfram reports: `FactorInteger` near `2^64`, degenerate
    random specifications, empty-list arguments, invalid permutation
    lists, out-of-range arguments and bad string positions across many
    heads.
- Evaluation-semantics fixes: the builtin attribute table, `Switch`
    pattern evaluation, `Return` no longer escaping `Table`/`Map`/
    `Select`, `Sequence` through postfix application, `Unevaluated`
    wrappers, chained `ReplaceAll`, `Condition` precedence, postfix `++`
    binding, and a parser call-limit fix for deeply nested brackets.
- `Enclose` and the `Confirm` family, `Success` and the `Failure`
    object family, and `$MessageList` (preserved around `Quiet`) are
    implemented.

## Platform, performance & tooling

- Windows is a supported platform: the unit tests pass and run in CI,
    and the nightly builds produce Windows binaries of both `woxi` and
    Woxi Studio.
- Prebuilt binaries for Linux, macOS and Windows are attached to every
    GitHub release, with a checksum file.
- The browser (WASM) build supports `Export`, including PNG/JPEG plot
    export via host rasterization.
- CI checks formatting and clippy (the codebase is clippy-clean outside
    three allowed lints), debug builds use `opt-level = 1`, and the
    crate metadata was fixed for crates.io publishing.

## Demonstration-driven fixes in detail

- Fixes driven by a Wolfram Demonstration that approximates derivatives on
    Chebyshev–Gauss–Lobatto points:
    - A pattern variable repeated on the left of a definition constrains the
        arguments to be equal: `f[i_, i_] := …` no longer matches `f[1, 2]`.
        The repeat used to be ignored, so a definition pair spelling out a
        matrix's diagonal and off-diagonal entries collapsed onto whichever
        rule came last — and a diagonal formula applied off the diagonal
        divided by zero. The constraint holds for a repeat inside a list
        pattern (`f[{a_, a_}]`) too, and alongside head constraints and
        `?test`s on the repeated slots.
    - `DownValues` and `Definition` print a nested argument pattern as it was
        written: `f[g[x_]] := x` used to read back as `f[__sp0_]`, exposing
        the placeholder such a slot is lowered to.
- Fixes driven by a Wolfram Demonstration that draws a rational cyclic
    polygon inside its circumcircle:
    - `Sphere` and `Ball` draw in a two-dimensional `Graphics`: in the plane
        a sphere is the circle bounding it and a ball the filled disk. Both
        used to be dropped, and since `Circumsphere` returns a `Sphere`
        whatever the dimension, a picture built around a circumcircle came
        out with no circle at all. A list of centres draws one per point,
        and `Ball[2]` is the unit disk at the origin.
    - A `Style` around a label paints its `Background` behind the text —
        `Style[Text[…], 12, Background -> White]` is what keeps a distance
        label readable over the line it sits on. Only a `Background` written
        on the `Text` itself was drawn before.
    - A square factor too large to reach by trial division comes out of a
        radical: `Sqrt[100003^2 * 115]` is `100003 Sqrt[115]`. `Sqrt` and the
        `c Sqrt[r]` merge used to look for square factors to different
        depths, so each rewrote what the other produced and halving such a
        coefficient — the `Mean` of two points carrying a radical — never
        returned.
- Fixes driven by the plane-geometry Wolfram Demonstrations that stack a
    numeric readout over a drawing:
    - A `Pane[…]` grid cell shows what it holds. `Pane` only reserves an
        area for its content, so a body laid out as
        `Grid[{{Pane[…]}, {Graphics[…]}}]` used to print the whole
        `Pane[…]` call as source text in place of the readout — and
        stretch the grid to the width of that source. `Item[…]` and
        `Text[…]` cells are peeled the same way, through the one helper
        every display pass already shares.
    - `Dividers -> All` rules every position of a grid, the outer edges
        included, so the grid is drawn as a closed box. Only the
        boundaries *between* cells were ruled before, which is what
        `Dividers -> Center` means; the two now differ, and either can be
        set per direction as `Dividers -> {colspec, rowspec}`.
    - `Spacings -> {{i -> s, …}, …}` sets the gap at individual column
        positions — position `i` is the gap to the left of column `i`, and
        position `ncols + 1` the margin after the last column. A
        list-valued horizontal spec used to be dropped, so a readout that
        groups its columns with wide and tight gaps came out evenly and
        far too widely spaced. A plain `{s1, s2, …}` names the positions
        in order, and positions left out keep the default gap.
- Manipulate improvements (driven by the "Recursive Exercises" Wolfram
    Demonstrations, which nest circles inside circles):
    - The leading assignments a body makes before anything else are
        evaluated with the control variables at their initial values, the
        way Wolfram evaluates the body before laying the controls out.
        A body opening with `u = {…, ss[-1., 1., n, dc]}`, where `ss`
        recurses down to a literal-`1` base case, used to recurse on a
        symbolic depth that never reached the base case, so extraction
        never returned and no widget appeared at all.
    - A choice list built from another control's variable
        (`Range[1, If[flat, 3, 6], 1]`) is re-resolved against the live
        bindings, so a level setter offering six levels flat narrows to
        three in 3D. A selected value the narrowed list no longer offers
        falls back to the last one it does, and the body is rendered again
        for it. Previously the list was fixed at build time.
    - `ControlType -> Slider` over a choice list renders a slider that
        steps through the choices, matching Wolfram; a twenty-entry
        colour-scheme list used to become a dropdown.
- Fixes driven by a Wolfram Demonstration that runs a cellular automaton
    over a dendrite and compares it with the linear case:
    - A layout that holds pictures — `Grid[{{plot1, plot2}, …}]`, a
        `Column` of them, a `Pane` around either — is composed into the one
        picture a notebook shows. The visual hosts (Playground, Woxi Studio)
        only ever reported the *last* graphic drawn while the layout was
        evaluated, so a Manipulate body that arranges several plots in a
        grid displayed a single one of them and dropped the rest.
    - A `Grid`'s columns line up when its rows hold different things: the
        caption under a picture is centred on that picture instead of
        being packed against the left edge of the grid.
    - `LayeredGraphPlot` and `TreePlot` draw the layered embedding they
        name, rather than the circular one every graph got. Vertices are
        placed by distance from a root, and the second argument (`Left`,
        `Right`, `Top`, `Bottom`) says which edge the roots go on.
    - `DirectedEdges -> False` draws a graph's edges as plain lines
        instead of arrows, and `ImageSize` sizes a graph plot — a
        Demonstration asking for a wide, short strip used to get the
        360-point default square.
    - `ArrayPlot` draws its `FrameLabel`, on any of the four edges, with
        the left and right labels rotated onto theirs.
- Fixes driven by the "Selectivity in a Semibatch Reactor" Wolfram
    Demonstration:
    - `Text` inside a `Graphics3D` scene is drawn — a labelled 3D
        schematic used to arrive with no lettering at all. It honours the
        `Style` size and colour, typesets `Subscript`/`Superscript`, and
        respects the alignment offset of `Text[expr, pos, offset]`.
    - `Scaled[{sx, sy}]` places a `Text` or `Inset` by fraction of the
        plot range, in `Graphics` and in a plot's `Epilog` alike. The
        fractions used to be unreadable, dropping every such label onto
        the origin.
    - `Inset[Graphics3D[…], pos]` embeds a three-dimensional picture in a
        flat one at its own size, instead of printing `-Graphics3D-`.
    - `Framed[…]` around a label draws its box — a border, and a panel
        when a `Background` is given — rather than printing its own
        source over the picture. `FrameStyle -> None` keeps the panel and
        drops the border.
    - `Subscript`/`Superscript` in a `Text` label typeset as scripts
        instead of falling through to the two-line `OutputForm` box.
    - `Graphics` accepts `Prolog` and `Epilog`, drawn under and over its
        content and taking no part in the plot range.
    - A `Manipulate` control panel written as a `Grid` drops its
        `SpanFromLeft` / `SpanFromAbove` cell markers; they used to
        survive as display elements and appear as literal text under the
        widget, one row per marker.
    - A control label computed with `Row[Flatten[{…, Riffle[…], …}]]` is
        evaluated before it is typeset, so the button shows the caption
        rather than the source of the computation.
- Fixes driven by the "Illustrative Performance Characteristics of Modern
    Motorcycles" Wolfram Demonstration:
    - `Piecewise` holds its pieces, so the value of a piece whose condition
        is `False` is never evaluated. This is what makes the construct a
        guard: `Piecewise[{{f[w], 0 <= w <= wmax}}, 0]` no longer calls `f`
        out of range, and the messages that call emitted are gone.
    - `FindRoot` falls back to a difference quotient when the symbolic
        derivative does not reduce to a number at the current iterate —
        differentiating a non-smooth function leaves `Derivative[1, 0][Max][…]`
        standing. It used to give up with `FindRoot::nlnum` and return
        unevaluated. Messages emitted by the discarded attempt (`D::ivar`
        and friends) no longer reach the user.
    - A `ControlType -> …` given to the `Manipulate` itself types every
        control that does not pick one, and a list value assigns one type per
        control in spec order. The Demonstrations idiom — controls laid out in
        a `Grid[…]`, typed once for the whole panel — now renders the dropdowns
        the author asked for instead of over-wide SetterBars.
- Fixes driven by a Wolfram Demonstration that draws a complex function as
    a colored vector field:
    - `ColorData[name, "ColorFunction"]` gives the gradient's color
        function, the same object `ColorData[name]` gives on its own. It
        used to be reported as unimplemented, so every primitive colored
        through it fell back to black. `ColorData[name, "Range"]` reads
        the parameter interval alongside it.
    - `Graphics[…, ImagePadding -> …]` honours the padding (it was only
        read by the plot functions), so the drawing area sits where the
        notebook puts it instead of at the automatic frame margins.
    - A named `Arrowheads` size is a fraction of the plot width and
        nothing else. It used to be capped at 45% of the arrow carrying
        it, which shrank the heads of a dense vector field — where each
        arrow is barely longer than its head — to unreadable specks.
- Fixes driven by the "Mass with a Spring and a Rubber Band" Wolfram
    Demonstration:
    - Substituting an `NDSolve` solution into a derivative
        (`y'[t] /. sol`) evaluates numerically instead of echoing a
        `Derivative[…]` expression, so a phase portrait of a numeric
        solution can be plotted.
    - A plot nested in a `Row` inside a `Column` keeps its own coordinate
        space and is drawn (it used to render blank).
    - `Frame -> True` labels its frame ticks even with `Axes -> False`.
    - `AspectRatio` together with `ImagePadding` sizes the image so the
        plot area spans the full width, instead of shrinking it into a
        canvas sized by the default ratio.
    - Labels written as a function of the plot variable typeset the way
        Wolfram draws them — `AxesLabel -> {t, y[t]}` reads `y(t)` and a
        derivative reads `y′(t)`, where both used to be dropped — and a
        `Manipulate` control labelled `Style["y", Italic]'` shows `y′`.
- Symbol names may contain `$` anywhere (`a$b`, `signal$1`), matching Wolfram.
- `PlotRange` bounds given in reversed order (e.g. `{3, -3}`) normalize to
    the same plot as the sorted form, matching Wolfram.
- Manipulate improvements (driven by the "Oscilloscope with Two Signal
    Inputs" Wolfram Demonstration):
    - `Style[…]`, bare-string, and `Delimiter` arguments are treated as
        static annotation rows between controls (no more spurious
        `Manipulate::vsform` messages) and render as headings/separators
        in Woxi Studio and the Playground.
    - Compound control variables such as `Subscript[signal, 1]` work: they
        are bound through synthesized symbols and keep a typeset label
        (`signal₁`).
    - Control specs may carry trailing options (`ControlType -> PopupMenu`,
        `ImageSize -> Tiny`, …); `ControlType -> PopupMenu` always renders
        a dropdown.
    - A `Manipulate` whose body is an `Animate[…]` renders as one combined
        widget, `AnimationRunning -> False` builds the widget paused, and
        an `Infinity` animation bound gets a finite looping window.
- Woxi Studio re-instantiates stored `Manipulate` widgets when opening a
    notebook (instead of showing the saved `DynamicModuleBox[…]` text dump).
- Manipulate improvements (driven by the "Center of Mass of a Polygon"
    Wolfram Demonstration):
    - `Locator` controls are interactive: a single point binds as a 2D
        slider, a point list becomes a per-point X/Y control (with
        add/remove when `LocatorAutoCreate -> True`, and a multi-handle
        drag pad in the Playground), instead of freezing the variable at
        its initial value.
    - Discrete choices whose rule label is a graphic (`"+" -> myIcon[2]`)
        render the icon in the SetterBar instead of dumping the label's
        InputForm.
    - A bare control-type shorthand in the range position
        (`{{p, init}, Locator}`) is no longer misread as a `Dynamic[…]`
        range.
- InputForm keeps required parentheses around loose operands of `@@`, `/@`,
    `@@@`, and `.` (`Plus @@ (x*y)/2`, `a . (b - c)`); previously the
    printed form re-parsed to a different expression, silently corrupting
    re-evaluated Manipulate bodies.
- `PlotLabel` works on plain `Graphics[…]` (rendered as a centered title),
    and labels may be arbitrary expressions such as
    `Row[{"center of mass: ", CM}]` on all plot types.
- An operator that binds tighter than `Times` now reaches into an adjacent
    implicit product: `2 Times @@ {3, 4}` is `2 (Times @@ {3, 4})` and
    `a b . c` is `a (b . c)`, matching Wolfram. Previously the whole product
    became one operand, so `lcm^n Times @@ Table[…]` applied the wrong head.
- Manipulate improvements (driven by the "Descartes's Rule of Signs"
    Wolfram Demonstration):
    - `Setter`, `Toggler`, `CheckboxBar`, `Opener` and `OpenerBar` are
        recognised as control types. A `Setter` spec used to be read as a
        slider bound, and since one unparsable spec aborts the whole
        extraction, the entire Manipulate fell back to a text echo.
    - `{v, domain, ControlType -> None}` starts `v` at the first choice of
        its domain rather than binding the choice list itself.
    - `TrackedSymbols :> {…}` limits which controls re-run the body; the
        others move without re-rendering, as in Wolfram.
- Display of `Text[content]` in the visual hosts (Woxi Studio, the
    Playground) shows the content, so a `Text@Pane[Column[{…}]]` body
    renders the whole column instead of only the picture inside it.
- `TraditionalForm` inside a layout is typeset in conventional notation
    rather than being stripped to StandardForm markup: `Equal` reads `=`
    (`≤`, `≥`, `≠` likewise), a `Row` of strings and `Style[…]`s is set as
    the text it displays, and a term with a negative coefficient carries
    its sign on the operator (`a - 130 x³`, not `a + -130 x³`).

# 2026-07-16 - 0.2.0

Between 0.1.0 and 0.2.0 Woxi grew from a minimal interpreter into a broad
computer algebra system covering a large subset of the Wolfram Language.
The list only includes the most prominent additions rather than every function.

## Calculus, algebra & equation solving

- Differentiation and integration: `D`, `Dt`, `Derivative` (including `f'[x]`
    prime notation, multi-index and pure-function derivatives), `Integrate`
    (trigonometric, Gaussian, u-substitution, multivariate/iterated, definite
    integrals), `NIntegrate` (including infinite bounds), `Grad`, `Div`, `Curl`,
    `Laplacian`, `Wronskian`, `ArcLength`, `FrenetSerretSystem`.
- Limits and series: `Limit` (directional, at infinity, finite points),
    `DiscreteLimit`, `MaxLimit`, `MinLimit`, `Series` (Taylor, Laurent, Puiseux,
    fractional powers, expansion at infinity), `SeriesData`, `Normal`, `O`,
    `Residue`, `PadeApproximant`, `ComposeSeries`, `InverseSeries`,
    `SeriesCoefficient`, `Asymptotic*` and `AsymptoticIntegrate`.
- Sums and products: symbolic `Sum` and `Product`, including geometric,
    exponential, alternating and p-series, harmonic-number and zeta closed forms,
    telescoping rational sums, multi-dimensional ranges, `NSum`, `NProduct`,
    `SumConvergence`, infinite rational products, and `GeneratingFunction` /
    `ExponentialGeneratingFunction`.
- Equation solving and optimization: `Solve`, `Reduce`, `FindInstance`,
    `FindRoot`, `Eliminate`, `SolveAlways`, `Roots`, `Root`, `RootSum`,
    `RSolve`, `RSolveValue`, `RecurrenceTable`, `FindLinearRecurrence`,
    `DSolve`, `DSolveValue`, `NDSolve`, `NDSolveValue`, `Minimize`, `Maximize`,
    `NMinimize`, `NMaximize`, `FindMinimum`, `FindMaximum`, `ArgMin`, `ArgMax`,
    `FunctionDomain`, `FunctionRange`.
- Simplification and manipulation: `Simplify`, `FullSimplify`, `Refine`,
    `Assuming`, `ConditionalExpression`, `Factor`, `Expand`, `ExpandAll`,
    `Together`, `Apart` (over irreducible quadratics and repeated factors),
    `ApartSquareFree`, `Cancel`, `Collect`, `PowerExpand`, `FunctionExpand`,
    `ComplexExpand`, `PiecewiseExpand`, `TrigExpand`, `TrigReduce`, `TrigFactor`,
    `TrigToExp`, `ExpToTrig`, `Variables`.

## Special functions

- Bessel family: `BesselJ`, `BesselY`, `BesselI`, `BesselK`,
    `SphericalBesselJ`/`SphericalBesselY`, `SphericalHankelH1`/`SphericalHankelH2`,
    `HankelH1`/`HankelH2`, `BesselJZero`, `KelvinBer`/`KelvinBei`,
    `StruveH`/`StruveL`, `AngerJ`, `WeberE`.
- Elliptic and Jacobi functions: `EllipticK`, `EllipticE`, `EllipticF`,
    `EllipticPi`, `EllipticTheta`, `EllipticNomeQ`, `JacobiSN`/`JacobiCN`/`JacobiDN`,
    `JacobiAmplitude`, `JacobiEpsilon`, `JacobiZeta`, all twelve inverse Jacobi
    functions (`InverseJacobiSN`, `InverseJacobiCN`, …), `WeierstrassP`,
    `WeierstrassInvariants`, `WeierstrassHalfPeriods`, the Neville theta
    functions, `ModularLambda`, `KleinInvariantJ`, `ArithmeticGeometricMean`.
- Gamma, zeta and related: `Gamma` (incomplete and regularized), `LogGamma`,
    `PolyGamma`, `Beta`/`BetaRegularized`, `Pochhammer`, `FactorialPower`,
    `BarnesG`, `LogBarnesG`, `Hyperfactorial`, `Zeta`, `HurwitzZeta`, `PrimeZetaP`,
    `RiemannR`, `RiemannSiegelZ`/`RiemannSiegelTheta`, `StieltjesGamma`, `LerchPhi`,
    `PolyLog`, `DirichletEta`/`DirichletBeta`/`DirichletLambda`/`DirichletL`.
- Hypergeometric functions: `Hypergeometric0F1`, `Hypergeometric1F1`,
    `Hypergeometric2F1`, `HypergeometricPFQ`, `HypergeometricU`, `MeijerG`,
    the Appell functions `AppellF1`–`F4`, and their regularized variants.
- Error, exponential-integral and Airy functions: `Erf`, `Erfc`, `Erfi`,
    `InverseErf`/`InverseErfc`, `FresnelS`/`FresnelC`/`FresnelF`/`FresnelG`,
    `ExpIntegralE`, `LogIntegral`, `SinIntegral`/`CosIntegral`,
    `ExpIntegralEi`, `SinhIntegral`/`CoshIntegral`, `DawsonF`, `OwenT`,
    `AiryAi`/`AiryBi` (and their derivatives `AiryAiPrime`/`AiryBiPrime`).
- Orthogonal polynomials and misc: `LegendreP`/`LegendreQ`, `HermiteH`,
    `LaguerreL`, `GegenbauerC`, `ChebyshevT`/`ChebyshevU`, `JacobiP`,
    `SphericalHarmonicY`, `ZernikeR`, `ClebschGordan`, `ThreeJSymbol`,
    `SixJSymbol`, `WignerD`, `MittagLefflerE`, `ChampernowneNumber`, `ThueMorse`,
    `RudinShapiro`, and the mathematical constants (`EulerGamma`, `Catalan`,
    `Glaisher`, `Khinchin`, `GoldenRatio`) at arbitrary precision.

## Statistics & probability

- Over 60 distributions with `PDF`, `CDF`, `Mean`, `Variance`,
    `StandardDeviation`, `Quantile`, `Moment`, `CharacteristicFunction`,
    `HazardFunction` and `SurvivalFunction` support, including
    `NormalDistribution`, `LogNormalDistribution`, `MultinormalDistribution`,
    `StudentTDistribution`, `ChiDistribution`/`ChiSquareDistribution`,
    `GammaDistribution`/`InverseGammaDistribution`,
    `BetaDistribution`/`BetaPrimeDistribution`/`BetaBinomialDistribution`,
    `CauchyDistribution`, `LaplaceDistribution`, `LogisticDistribution`,
    `WeibullDistribution`, `FrechetDistribution`,
    `ExtremeValueDistribution`/`GumbelDistribution`, `RayleighDistribution`,
    `MaxwellDistribution`, `ParetoDistribution`, `PoissonDistribution`,
    `BinomialDistribution`/`NegativeBinomialDistribution`, `GeometricDistribution`,
    `HypergeometricDistribution`, `ZipfDistribution`, `SkellamDistribution`,
    `PERTDistribution`, `DagumDistribution`, `RiceDistribution`,
    `InverseGaussianDistribution`, `MoyalDistribution`, `StableDistribution`,
    `VonMisesDistribution`, `HoytDistribution`, `NakagamiDistribution`,
    `LogLogisticDistribution`, `LogSeriesDistribution`, `MeixnerDistribution`,
    `TukeyLambdaDistribution`, `TsallisQGaussianDistribution`, `WakebyDistribution`,
    `SinghMaddalaDistribution`, `BenktanderWeibullDistribution`,
    `BenfordDistribution`, `PoissonConsulDistribution`,
    `CompoundPoissonDistribution`, `CoxianDistribution`,
    `HyperexponentialDistribution`, `HotellingTSquareDistribution`,
    `DirichletDistribution`, `WishartMatrixDistribution`,
    `NegativeMultinomialDistribution`, `HistogramDistribution` and
    many more, plus reliability distributions (`FailureDistribution`,
    `StandbyDistribution`, `FirstPassageTimeDistribution`) and meta-distributions
    (`TransformedDistribution`, `ProductDistribution`, `CensoredDistribution`,
    `EmpiricalDistribution`, `MixtureDistribution`, `SliceDistribution`,
    `QuantityDistribution`).
- Random processes: `WienerProcess`, `GeometricBrownianMotionProcess`,
    `OrnsteinUhlenbeckProcess`, `BrownianBridgeProcess`, `PoissonProcess`,
    `DiscreteMarkovProcess` and the Bernoulli/Binomial/WhiteNoise processes, with
    time slices, `CovarianceFunction`, `CorrelationFunction` and
    `AbsoluteCorrelationFunction`; plus `StateSpaceModel`, `ObservabilityMatrix`
    and `ControllabilityMatrix` for linear systems.
- Descriptive statistics: `Mean`, `Median`, `Commonest`, `Quantile`,
    `Quartiles`, `InterquartileRange`, `Variance`, `StandardDeviation`,
    `GeometricMean`, `HarmonicMean`, `ContraharmonicMean`, `RootMeanSquare`,
    `TrimmedMean`, `WinsorizedMean`, `Skewness`, `Kurtosis`, `CentralMoment`,
    `Cumulant`, `Covariance`, `Correlation`, `MeanDeviation`, `Standardize`,
    `MovingAverage`, `ExponentialMovingAverage`, `PrincipalComponents`.
- Fitting and inference: `Fit`, `LinearModelFit`, `FindFit`,
    `FindDistributionParameters`, `Expectation`, `Probability`, `LogLikelihood`,
    correlation and dissimilarity measures (`SpearmanRho`, `GoodmanKruskalGamma`,
    `HoeffdingD`, `BlomqvistBeta`, plus a family of distance functions), and
    random sampling via `RandomVariate`, `RandomChoice`, `RandomSample`,
    `RandomReal`, `RandomInteger`, `RandomComplex`, `RandomPrime`.

## Number theory

- Primes and factoring: `PrimeQ` (BigInteger Miller–Rabin), `NextPrime`,
    `PrimePi`, `PrimeOmega`, `PrimeNu`, `FactorInteger` (incl. Gaussian
    integers), `Divisors`, `DivisorSigma`, `DivisorSum`, `EulerPhi`,
    `CarmichaelLambda`, `MoebiusMu`, `PerfectNumber`/`PerfectNumberQ`,
    `MersennePrimeExponentQ`.
- Modular arithmetic and symbols: `Mod`, `PowerMod`/`PowerModList`,
    `ModularInverse`, `MultiplicativeOrder`, `PrimitiveRoot`, `ChineseRemainder`,
    `JacobiSymbol`, `KroneckerSymbol`, `CoprimeQ`.
- Integer sequences and digits: `Fibonacci`, `LucasL`, `CatalanNumber`,
    `BernoulliB`, `EulerE`, `StirlingS1`/`StirlingS2`, `BellB`,
    `PartitionsP`/`PartitionsQ`,
    `IntegerPartitions`, `Subfactorial`, `HarmonicNumber` (and hyper/multiple
    variants), `IntegerDigits`, `DigitCount`, `DigitSum`, `FromDigits`,
    `RealDigits` (arbitrary bases, repeating decimals), `ContinuedFraction`,
    `FareySequence`, `MinkowskiQuestionMark`, `RomanNumeral`, `IntegerName`.

## Polynomials

- `PolynomialGCD`/`PolynomialLCM`, `PolynomialQuotient`/`PolynomialRemainder`,
    `PolynomialExtendedGCD`, `PolynomialReduce`, `Resultant`, `Subresultants`,
    `Discriminant`, `GroebnerBasis`, `Cyclotomic`, `FactorList`,
    `FactorSquareFree`, `FactorTerms`, `Decompose`, `MonomialList`,
    `CoefficientList`/`CoefficientRules`/`FromCoefficientRules`,
    `InterpolatingPolynomial`, `CharacteristicPolynomial`, `HornerForm`,
    `SymmetricPolynomial`, `PowerSymmetricPolynomial`, `SymmetricReduction`,
    `SubresultantPolynomials`/`SubresultantPolynomialRemainders`, `ToRadicals`,
    `NumberFieldDiscriminant`, `AlgebraicNumber` norm/trace/`AlgebraicUnitQ`, and
    modular polynomial arithmetic over GF(p).

## Linear algebra & tensors

- Decompositions and solvers: `LinearSolve`, `LeastSquares`, `RowReduce`,
    `Inverse`, `PseudoInverse`, `Det`, `PfaffianDet`, `MatrixRank`, `NullSpace`,
    `Eigenvalues`, `Eigenvectors`, `Eigensystem`, `LUDecomposition`,
    `QRDecomposition`, `CholeskyDecomposition`, `LDLDecomposition`,
    `JordanDecomposition`, `SchurDecomposition`, `HermiteDecomposition`,
    `SmithDecomposition`, `FrobeniusReduce` (rational canonical form),
    `SingularValueList`, `Orthogonalize`, `LatticeReduce`, and `Modulus`-option
    solvers over GF(p).
- Matrix functions and constructors: `MatrixPower`, `MatrixExp`, `MatrixLog`,
    `MatrixFunction`, `DrazinInverse`, `Adjugate`, `RankDecomposition`,
    `LyapunovSolve`/`DiscreteLyapunovSolve`, `IdentityMatrix`, `DiagonalMatrix`,
    `HilbertMatrix`, `HankelMatrix`, `ToeplitzMatrix`, `HadamardMatrix`,
    `FourierMatrix`, `VandermondeMatrix`, `PauliMatrix`, `RotationMatrix`,
    various rotation/reflection/scaling/shearing matrices, and a full set of
    matrix predicates (`SymmetricMatrixQ`, `PositiveDefiniteMatrixQ`,
    `OrthogonalMatrixQ`, `UnitaryMatrixQ`, …).
- Vectors and tensors: `Dot`, `Cross`, `Norm`, `Normalize`, `Projection`,
    `VectorAngle`, `KroneckerProduct`, `Outer`, `Inner`, `TensorProduct`,
    `TensorWedge`, `TensorTranspose`, `ArrayDot`, `LeviCivitaTensor`,
    `SparseArray` and numerous distance functions.

## Integral transforms & signal processing

- `FourierTransform`/`InverseFourierTransform` (incl. sine/cosine variants),
    `LaplaceTransform`/`InverseLaplaceTransform`, `ZTransform`/`InverseZTransform`,
    `MellinTransform`/`InverseMellinTransform`, discrete `Fourier`/`InverseFourier`
    (Cooley–Tukey FFT), `Convolve`, `DiscreteConvolve`, `ListConvolve`,
    `ListCorrelate`, Fourier series coefficients, `DiscreteHadamardTransform`,
    `DiscreteHilbertTransform`.
- Filters and resampling: `LowpassFilter`, `HighpassFilter`, `BandpassFilter`,
    `BandstopFilter`, `WienerFilter`, `TotalVariationFilter`, `MeanFilter`,
    `MedianFilter`, `MinFilter`/`MaxFilter`, `Upsample`/`Downsample`,
    `PeakDetect`/`FindPeaks`, `CrossingDetect`, `SavitzkyGolayMatrix`, waveform
    generators (`SawtoothWave`, `SquareWave`, `TriangleWave`) and a full set of
    window functions.
- Wavelet analysis: the wavelet families (`HaarWavelet`, `DaubechiesWavelet`,
    `SymletWavelet`, `CoifletWavelet`, `MeyerWavelet`, `MexicanHatWavelet`,
    `MorletWavelet`, …), the transforms (`DiscreteWaveletTransform`,
    `StationaryWaveletTransform`, `LiftingWaveletTransform`,
    `ContinuousWaveletTransform` and inverses), the data objects
    (`DiscreteWaveletData`, `ContinuousWaveletData`), coefficient manipulation
    (`WaveletThreshold`, `WaveletBestBasis`), and the wavelet plots
    (`WaveletListPlot`, `WaveletScalogram`, …).

## Graph theory & permutations

- Graph construction and rendering: `Graph`, `GraphPlot`, `LayeredGraphPlot`,
    named graphs (`CompleteGraph`, `PathGraph`, `CycleGraph`, `WheelGraph`,
    `StarGraph`, `HypercubeGraph`, `PetersenGraph`, `KaryTree`, `TuranGraph`,
    `DeBruijnGraph`, `CirculantGraph`, …), adjacency/incidence conversions,
    `Subgraph`, `LineGraph`, `NeighborhoodGraph`, `DirectedGraph`,
    `TransitiveReductionGraph`, and edge/vertex editing.
- Metrics and algorithms: `DegreeCentrality`, `BetweennessCentrality`,
    `ClosenessCentrality`, `EigenvectorCentrality`, `KatzCentrality`,
    `PageRankCentrality`, `RadialityCentrality`, `GraphLinkEfficiency`,
    `GraphDistance`, `FindShortestPath`,
    `FindSpanningTree`, `FindCycle`, `FindMaximumFlow`, `FindMinimumCostFlow`,
    `FindClique`, `FindVertexCover`, `ConnectedComponents`,
    `WeaklyConnectedComponents`, `TuttePolynomial`, `ChromaticPolynomial`,
    `GraphDiameter`/`GraphRadius`/`GraphCenter`/`GraphPeriphery`, and a family of
    graph predicates.
- Group theory: `SymmetricGroup`, `AlternatingGroup`, `DihedralGroup`,
    `CyclicGroup`, the Mathieu groups (`M11`, `M12`, `M22`, `M23`, `M24`),
    `CycleIndexPolynomial`, `GroupMultiplicationTable`, `GroupStabilizer`,
    `GroupOrbits`, `GroupElementPosition`, and permutation operations
    (`PermutationProduct`, `PermutationPower`, `InversePermutation`,
    `FindPermutation`, `Cycles`).

## Lists, associations & functional programming

- Core list operations: `Table`, `Map`, `MapThread`, `MapIndexed`, `MapAt`,
    `Apply`, `Thread`, `Through`, `Fold`/`FoldList`/`FoldPair`, `Nest`/`NestList`,
    `NestWhile`/`NestWhileList` (with cycle detection), `Tuples`, `Subsets`,
    `Permutations`, `Flatten`, `Partition`, `Take`/`Drop`, `Part` (with `All`,
    `Span` and multi-index specs), `Cases`, `Count`, `Select`, `Pick`,
    `Sow`/`Reap`, `Gather`/`GatherBy`, `SortBy`, `Ordering`, `Subdivide`.
- Associations: the `<|…|>` constructor with key access, `AssociationThread`,
    `AssociationMap`, `Merge`, `KeyMap`, `KeySelect`, `KeyTake`/`KeyDrop`,
    `Keys`/`Values`, `KeySort`, `KeyValueMap`, `Lookup`, `GroupBy`, `Counts`,
    `Query` and `Dataset`.
- Higher-order helpers: `OperatorApplied`, `Comap`, `ReverseApplied`, `Curry`,
    `SequenceFold`, `ArrayReduce`, `SubsetMap`, `ReplaceAt`, `NearestTo`,
    `PositionLargest`/`PositionSmallest`, and the `AddSides`/`SubtractSides`/…
    equation-manipulation operators.

## Strings & text

- `StringJoin`, `StringSplit`, `StringReplace`, `StringCases`, `StringPosition`,
    `StringMatchQ`, `StringContainsQ`/`StringStartsQ`/`StringEndsQ`,
    `StringTake`/`StringDrop`, `StringInsert`/`StringDelete`, `StringPartition`,
    `StringRiffle`, `StringTemplate`,
    `StringForm`, `Capitalize`/`Decapitalize`, `ToUpperCase`/`ToLowerCase`,
    `Characters`, `CharacterRange`, `ToCharacterCode`/`FromCharacterCode`
    (with encodings), `IntegerString`, `Alphabet`, `Transliterate`, and full
    string-pattern support (`RegularExpression`, `Except`, `Repeated`, captures
    and backreferences) threaded over lists.
- Sequence alignment and similarity: `LongestCommonSubsequence`,
    `SequenceAlignment`, `NeedlemanWunschSimilarity`, `SmithWatermanSimilarity`,
    `DamerauLevenshteinDistance`, `WordCounts`, `TextSentences`.

## Dates, times, units & quantities

- Date/time: `DateObject`, `DateList`, `DateString`, `DateValue`, `DateRange`,
    `DatePlus`, `DayName`, `DayCount`, `DayRange`, `DateWithinQ`, `DateOverlapsQ`,
    `DateSelect`, `DatePattern`, `Duration`, `CalendarConvert`, `Now`, `Today`,
    `TimeObject`, `TimeSeries`/`TemporalData`, `AbsoluteTime`/`FromAbsoluteTime`,
    `UnixTime`, `JulianDate`, `TimeZoneConvert`/`TimeZoneOffset` (DST-aware named
    IANA zones), `$TimeZone`, and `DateObject` + `Quantity` arithmetic.
- Units: `Quantity`, `UnitConvert`, `UnitDimensions`, `QuantityUnit`,
    `KnownUnitQ`, compound-unit parsing and dimensional analysis, and affine
    temperature handling.

## Geometry & regions

- Regions and measures: `RegionMeasure`, `Area`, `Volume`, `Perimeter`,
    `SurfaceArea`, `ArcLength`, `RegionCentroid`, `RegionMoment`,
    `MomentOfInertia`, `RegionNearest`, `RegionDistance`, `RegionMember`,
    `RegionDisjoint`, `BoundingRegion`, `MeshRegion`, `VoronoiMesh`, `ArrayMesh`,
    `CantorMesh` and morphological operations.
- Constructors and transforms: `Triangle` (AAS/ASA/SAS/SSS, including symbolic
    angles), `TriangleCenter`/`TriangleMeasurement`, `Simplex`, `Ball`,
    `Ellipsoid`, `RegularPolygon`, the Platonic-solid primitives, `SphericalShell`,
    `CapsuleShape`, `StadiumShape`, `DiskSegment`, `HalfSpace`, `Insphere`,
    `AngleBisector`/`PerpendicularBisector`, the geometric predicates
    (`CollinearPoints`, `CoplanarPoints`, `ConvexPolygonQ`, `SimplePolygonQ`),
    coordinate-bounding utilities, and the affine
    transformation family (`TranslationTransform`, `RotationTransform`,
    `ScalingTransform`, `ShearingTransform`, `ReflectionTransform`,
    `AffineTransform`, `EulerMatrix`, `RollPitchYawMatrix`).
- Space-filling and fractal curves: `HilbertCurve`, `PeanoCurve`,
    `SierpinskiCurve`, `KochCurve`, `CantorStaircase`, `AnglePath`/`AnglePath3D`,
    `MandelbrotSetMemberQ` and `MandelbrotSetIterationCount`.

## Graphics & plotting

- Function plots: `Plot`, `Plot3D`, `ParametricPlot`/`ParametricPlot3D`,
    `PolarPlot`, `PolarCurve`/`FilledPolarCurve`, `ContourPlot`, `DensityPlot`,
    `RegionPlot`/`RegionPlot3D`, `RevolutionPlot3D`, `SphericalPlot3D`,
    `ComplexPlot`/`ComplexPlot3D`/`ComplexRegionPlot`, `LogPlot`/`LogLogPlot`/
    `LogLinearPlot`, `DiscretePlot`/`DiscretePlot3D`.
- List and chart visualizations: `ListPlot`, `ListLinePlot`, `ListPointPlot3D`,
    `ListContourPlot`, `ListDensityPlot`, `DateListPlot`, `NumberLinePlot`,
    `BarChart`/`BarChart3D`, `PieChart`/`SectorChart`, `BubbleChart`,
    `Histogram`, `BoxWhiskerChart`, `ArrayPlot`, `MatrixPlot`, `WordCloud`,
    `TimelinePlot`, `AngularGauge`, `PeriodicTablePlot`, `GeoGraphics`/
    `GeoHistogram`.
- Primitives, styling and output: `Graphics`/`Graphics3D`, the box-language
    pipeline, `GraphicsComplex`, `BezierCurve` and `BSplineCurve`, `Raster`, gradient
    fills, plot options (`PlotStyle`, `PlotLegends`, `PlotTheme`, `GridLines`,
    `Filling`, `Frame`, `Callout`, …), `Show`,
    `GraphicsRow`/`GraphicsColumn`/`GraphicsGrid`,
    light/dark-mode SVG, and rendering via `ExportString[expr, "SVG"]`.

## Images, audio & music

- Images: an `Image`/`Image3D` type with data access, arithmetic
    (`ImageAdd`/`ImageSubtract`/`ImageMultiply`, `Blend`), filters (`GaussianFilter`,
    `MedianFilter`, `ImageConvolve`, …), geometry (`ImageResize`, `ImageRotate`,
    `ImageReflect`, `ImageTrim`, `ImagePartition`, `Thumbnail`), color operations
    (`ColorConvert`, `ColorSeparate`, `ColorCombine`, `ColorNegate`,
    `ColorDistance`), analysis (`ImageValue`, `DistanceTransform`,
    `FillingTransform`, `MorphologicalBinarize`),
    `ImageCollage`/`ImageAssemble`, `Rasterize`, and image import/SVG export.
- Audio: the Audio Processing guide — editing (`AudioAmplify`, `AudioTrim`,
    `AudioJoin`, `AudioPitchShift`), analysis (`AudioMeasurements`,
    `AudioLocalMeasurements`, `AudioIntervals`), the short-time Fourier transform,
    spectral plots (`Spectrogram`, `Cepstrogram`, `Periodogram`), noise-removal
    filters, WAV import/export, and audible `Play`/`Sound`/`Audio` playback in
    the Playground and Studio.
- Music: computational-music objects (`MusicNote`, `MusicChord`, `MusicPitch`,
    `MusicDuration`) with canonicalization, pitch arithmetic, MIDI export and
    SMuFL staff rendering.

## Data, knowledge & I/O

- Knowledge and entities: `EntityStore`/`EntityRegister`/`EntityValue`,
    `ElementData` for all 118 elements, a country/planet knowledge base,
    `GeoPosition`/`Latitude`/`Longitude` and the geodesy functions
    (`GeoDistance`, `GeoPath`, `GeoDestination`, `GeoAntipode`), plus
    `Molecule`/`MoleculeValue` and `WikidataData`.
- Import/Export: `Import`/`ImportString` and `Export`/`ExportString` for CSV,
    TSV, Table, Text, JSON, XLSX, XML, image/SVG and CERN ROOT formats, `Dataset`,
    `BinarySerialize`/`BinaryDeserialize` (WXF), `$ImportFormats`/`$ExportFormats`,
    and `Hash` (MD5, SHA, CRC32, … with multiple output encodings),
    `BaseEncode`/`BaseDecode`.
- Files, streams and system: file-path utilities
    (`FileNameJoin`/`FileNameSplit`/`FileNameTake`, `DirectoryName`,
    `FileExistsQ`, `FileNames`, `SetDirectory`, `CreateFile`, `CopyFile`,
    `RenameFile`/`RenameDirectory`, `FileSize`), stream I/O
    (`OpenRead`/`OpenWrite`, `Read`/`Write`/`ReadList`, `BinaryRead`/`BinaryWrite`,
    `Put`/`Get`), and system variables (`$Version`, `$VersionNumber`,
    `$OperatingSystem`, `$SystemID`, and the memory/timing variables).
- Web: `URLRead`, `HTTPRequest`, `URLParse`, `URLBuild`, `URLEncode`/`URLDecode`.

## Language, patterns & evaluation

- Pattern matching: `Pattern`, `Blank`/`BlankSequence`/`BlankNullSequence`,
    `Optional` and defaults, `Alternatives`, `Except`, `Condition` (`/;`),
    `Verbatim`, `HoldPattern`, `KeyValuePattern`, `Repeated`/`RepeatedNull`,
    with `Flat`/`Orderless`/`OneIdentity` matching.
- Rules, definitions and attributes: `Set`/`SetDelayed`, `TagSet`/`TagSetDelayed`,
    `UpSet`/`UpSetDelayed`, `Unset`, `DownValues`/`UpValues`/`SubValues`,
    `Attributes` with the full attribute set, `Protect`/`Unprotect`,
    `Clear`/`ClearAll`/`Remove`, `Options`/`SetOptions`/`OptionValue`.
- Scoping and control flow: `Module`, `Block`, `With`, `If`, `Which`, `Switch`,
    `For`, `While`, `Do`, `Break`, `Continue`, `Return`, `Goto`/`Label`,
    `CompoundExpression`, `ApplyTo` (`//=`), `PrintTemporary`,
    `Hold`/`HoldForm`/`ReleaseHold`, `Evaluate`,
    `Sequence` flattening, `Catch`/`Throw`, `Quiet`, `Check`, `TimeConstrained`,
    `MemoryConstrained`, and `Message`/`MessageName` diagnostics matching
    `wolframscript`.
- Booleans and logic: `And`/`Or`/`Not`/`Nand`/`Nor`/`Xor`/`Xnor`/`Implies`/
    `Equivalent`, `Boole`, `BooleanConvert` (DNF/CNF), `BooleanMinimize`
    (Quine–McCluskey), `BooleanTable`, `SatisfiableQ`, `TautologyQ`, `Exists`,
    `ForAll`.

## Parser & syntax

- Operator support: implicit multiplication, `n!`/`n!!`, `*^` scientific
    literals, `..`/`...` repeated patterns, the `@`/`@@`/`@@@`/`//`/`/@`/`~f~`
    application operators with correct precedence, `|->` arrow functions,
    `;;` spans, `>>`/`>>>` `Put`/`PutAppend`, `^:=` `UpSetDelayed`, `::`
    message names, and many named infix operators (`CircleDot`, `CircleTimes`,
    `Wedge`, `CenterDot`, `Element`, `Distributed`, …).
- Unicode and box syntax: Unicode operators (`≤`, `≥`, `≠`, `→`, `∈`, `∑`, …),
    named characters (`\[Psi]`, `\[Element]`, …) as symbols and function heads,
    character escapes (`\.HH`, `\:HHHH`, `\OOO`), box-syntax escapes and
    multi-line continuation.

## Output & formatting

- Form functions: `InputForm`, `FullForm`, `OutputForm`, `TraditionalForm`
    (conventional TeX-like typesetting of sums, integrals, derivatives, matrices,
    radicals and special functions), `TeXForm`, `MathMLForm`, `CForm`,
    `FortranForm`, `TableForm`, `MatrixForm`, `TreeForm`, `Column`, `Grid`,
    `Row`, `Framed`, `Definition`/`FullDefinition`.
- Number formatting: `NumberForm`, `ScientificForm`, `EngineeringForm`,
    `AccountingForm`, `PaddedForm`, `PercentForm`, `BaseForm`, with correct
    scientific-notation thresholds, digit blocking and banker's rounding, and
    consistent 6-significant-figure machine-real rendering.

## Interactive manipulation (Playground & Studio)

- `Manipulate` renders as an interactive widget driving live graphics, with
    sliders, popup menus, `SetterBar`/`CheckboxBar`/`RadioButtonBar`, `Locator`
    controls, 2D sliders, interval sliders, discrete pick-lists and the
    standalone `Control[…]` expression. The remaining interactive-manipulation
    heads (`Animator`, `Trigger`, `ProgressIndicator`, `PopupView`,
    `PaneSelector`, `Slider2D`, …) stay symbolic in script mode, matching
    `wolframscript`.
- `Animate` and `ListAnimate` render as auto-playing widgets with a play/pause
    button; `LocatorPane` and `ClickPane` render as draggable/clickable pads that
    feed pointer positions to their handlers. `ControlActive` now evaluates to
    its inactive form outside an actively manipulated control.

## Woxi Studio

- New native `.nb` notebook editor (`woxi-studio` crate) built with `iced`:
    per-cell evaluation (Shift+Enter), cell-type dropdown, drag-and-drop cell
    reordering, undo/redo, preview mode, dark-mode styling, selectable output
    text, keyboard shortcuts and navigation, 3D-graphics and image modals, an
    interactive `Manipulate` pipeline, external-player audio play/pause, and
    export to Mathematica / Jupyter / Markdown / LaTeX / Typst.

## Woxi Playground & JupyterLite

- Playground: WASM interpreter with a CodeMirror editor, per-expression output
    boxes, SVG/graphics and `Dataset`/`TableForm` rendering, `?symbol`
    information lookup, an auto/light/dark theme toggle, and a share button that
    encodes the session into the URL.
- JupyterLite: an integrated Woxi kernel with graphical `Plot`/SVG output and
    `?symbol` support, embedded in the docs.

## Language bindings

- Woxi for Python: a PyO3/maturin package (published to PyPI as `woxi`) that
    wraps the interpreter, evaluating Wolfram Language expressions from Python.
- Node.js: an npm package with WebAssembly bindings for running Woxi in
    JavaScript/Node.js environments.

# 2025-05-08 - 0.1.0

- Render top-level `PolarCurve[…]` and `FilledPolarCurve[…]` as graphics in
    the playground and Woxi Studio (the CLI keeps the symbolic echo), and
    support the `FilledPolarCurve[r, θ]` bare-variable form
- Render `Region[Style[reg, directives…]]` with the style directives applied
- Render `DateObject[…]` results (e.g. from `RandomDate` or `Now`) as a
    framed date panel in the playground and Woxi Studio
- Implement `WikidataData` and `ExternalIdentifier`,
    including `Import` of SVG files and `URL[…]` sources
- Render `Audio[…]` objects (file-backed via `File[…]`/path strings, or from
    sample data) as a graphical audio player in the playground and Woxi Studio
- Add support for `HTTPRequest` objects including property extraction
- Add support for `QuestionObject`, `AssessmentFunction`, and `AssessmentResultObject`
- Implement `DateString` and `Now`
- Implement `StringStartsQ` and `StringEndsQ`
- Support executing Woxi as a shebang script
- Implement `RandomInteger` function
- Implement `AllTrue` function
- Add support for anonymous functions
- Implement `MemberQ` function
- Implement `NumberQ` function
- Add support for all comparison operators
- Add support for function declarations
- Add support for semicolon separated expressions, implement `Set`
- Add support for comments, implement `Sin`, `@`, and `//`
- Add support for associations
- Implement `Floor`, `Ceiling`, and `Round` functions
- Implement `Divide` function
- Implement several boolean functions
- Implement `Times` function
- Implement `Minus` function
- Implement `Plus` function
- Implement `Sqrt` function
- Implement several string functions
- Add support for `#^2& /@ …`
- Implement `Abs` function
- Implement `/@` (Map operator)
- Implement `Total` function
- Implement `Select` and `Flatten`
- Implement `Drop`, `Append`, and `Prepend`
- Implement `Rest`, `Most`, `Take`, and `Part`
- Implement `Map` function
- Implement `Sign` function
- Implement `Length` function
- Implement `Print` function
- Implement `EvenQ` and `OddQ` functions
- Implement `Prime` function
- Add CLI subcommands `run`
- Add subcommand `eval` for evaluating Wolfram Language expressions
