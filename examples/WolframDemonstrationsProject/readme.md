# Wolfram Demonstrations Project

Notebooks from the [Wolfram Demonstrations Project][wdp] that Woxi Studio
opens and runs.

Each one is covered by an end-to-end test in `woxi-studio/src/main.rs`:
the notebook is parsed, its `Manipulate` is instantiated, and its body is
rendered at several control settings. The notebooks themselves are not
checked into this repository — the tests inline the cell source they need.

A random Demonstration can be picked with:

```sh
curl -sSL -o /dev/null -w '%{url_effective}\n' \
  'https://www.wolframcloud.com/obj/resourcesystem/api/1.0/RandomResourcePage?ResourceTypes=Demonstration'
```

[wdp]: https://resources.wolframcloud.com/DemonstrationRepository/web


## Supported

| Demonstration | Notable for |
| --- | --- |
| A Converging Geometric Series | `Grid` with a `NumberForm` caption over two pictures |
| A Procedure to Compute the Digit Sequence of a Square Root | Typeset radical and binary `BaseForm` over two `ArrayPlot`s |
| A Solution of Euler's Type for an Exact Differential Equation | Meshed `ContourPlot` under gradient arrows, with a `Locator` |
| An Expanding Structure Based on the Diamond Lattice | `Graphics3D` from `PolyhedronData` faces, `RadioButton` control |
| Balanced Ternary Notation | Balance beam labelled with underscored ternary digits |
| Binomial Probability Distribution | Stem plot with `PlotLabel` and both `AxesLabel`s |
| Constant Price Elasticity of Demand | `Grid` of two `Show[Plot[…], Graphics[…]]` panels |
| Deciding Rain-Affected Cricket Matches: The Duckworth-Lewis Method | Controls in a `TabView`, `StyleForm` grid with `SpanFromLeft` |
| Dedekind Cut | A circle at every rational `p/q < 1`, coloured by the cut |
| Dynamics of a Spring-Pendulum System | `NDSolveValue` with an algebraic constraint; `First[Plot[…]]` as a shape |
| Filling Cone, Hemisphere and Cylinder: Easy as 1:2:3 | `Tube` walls, a `RevolutionPlot3D` bowl lifted with `First`, `SetterBar` views |
| Force to Overcome Vacuum Pull | `Column` of a diagram over two captioned plot panels |
| Goldbach Conjecture | `Column` of decompositions over a `ListPlot` with explicit ticks |
| Gravestone from Transformation of Bilinski Dodecahedron 2 | `RasterBox[CompressedData[…]]` texture |
| Inscribed Angles That Intercept the Same Arc | `DynamicModule` ending in a `Grid`, driven by `Locator`s |
| Merging Schools of Fish | Assignment to a list of downvalue patterns; 120 translucent polygons |
| Plot a Quadratic Inequality | `RegionPlot` over an indirectly named region |
| Sampling a Digital Signal | Two `ListPlot`s with `Filling -> Axis` and `ImagePadding` |
| Some Irreptiles of Order Greater than 20 | Data table with omitted (`Null`) elements, hard-wrapped boxes |
| Stochastic Model of Microbial Injury and Mortality | In-place part assignment inside a compound `If` |
| The Mayan Calendar | `Disk` sectors with `Inset` teeth, `Row` separated by a `Spacer` |
| The Price of a Call Option on Electrical Power | Formula written in Unicode notation across four initialization cells |
| Trigonometric Sums as Parametric Curves | Typeset `∑` boxes read back as `Sum[…]` |
| Two Circular Windows | Bracketed subscript boxes as part specifications |


## Partially supported

| Demonstration | What is missing |
| --- | --- |
| Miscible Displacement of Oil in Heterogenous Porous Media | The cell parses and its controls are found, but the body needs a finite-element PDE solver — `NDSolve` handles ODEs only |
| Nets for Regular Spherical Models | The nets match; the solid view draws about 1.3× smaller, because it asks for `ViewAngle` and Woxi projects orthographically |
| Non Placet Net of a Dodecahedron | Folds correctly through the four-argument `Rotate`; scaled like the above, for the same `ViewAngle` reason |
