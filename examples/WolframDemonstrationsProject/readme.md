# Wolfram Demonstrations Project

The [Wolfram Demonstrations Project][wdp] publishes thousands of interactive
notebooks, each built around a single `Manipulate`. They are a good measure of
how much of the Wolfram Language Woxi covers in practice: a Demonstration is
real code written by somebody else, and it either opens and behaves like its
published snapshots or it does not.

The notebooks themselves are not part of this repository. Download one from
its page (the *Source Notebook* button) and open it with Woxi Studio:

```sh
cargo run --bin woxi-studio -- ThermodynamicConsistencyTestBasedOnDifferentialResiduals.nb
```

[wdp]: https://demonstrations.wolfram.com

## Supported

Every notebook below has an end-to-end test in `woxi-studio/src/main.rs` that
parses it, instantiates its widget and checks what it draws, so support for it
cannot regress unnoticed.

| Demonstration | Exercises |
| --- | --- |
| [A Converging Geometric Series](https://demonstrations.wolfram.com/AConvergingGeometricSeries/) | `Grid` with a `NumberForm` caption over pictures assembled from `Sow`n rectangles |
| [A Procedure to Compute the Digit Sequence of a Square Root](https://demonstrations.wolfram.com/AProcedureToComputeTheDigitSequenceOfASquareRoot/) | |
| [A Solution of Euler's Type for an Exact Differential Equation](https://demonstrations.wolfram.com/ASolutionOfEulersTypeForAnExactDifferentialEquation/) | |
| [An Expanding Structure Based on the Diamond Lattice](https://demonstrations.wolfram.com/AnExpandingStructureBasedOnTheDiamondLattice/) | `Graphics3D` assembled from `PolyhedronData` face lists |
| [Balanced Ternary Notation](https://demonstrations.wolfram.com/BalancedTernaryNotation/) | balance beam labelled with a `Section`-styled `Row` |
| [Binomial Probability Distribution](https://demonstrations.wolfram.com/BinomialProbabilityDistribution/) | plot labels with a typeset subscript |
| [Constant Price Elasticity of Demand](https://demonstrations.wolfram.com/ConstantPriceElasticityOfDemand/) | `Grid` of `Show[Plot[…], Graphics[…]]` panels |
| [Deciding Rain-Affected Cricket Matches: The Duckworth-Lewis Method](https://demonstrations.wolfram.com/DecidingRainAffectedCricketMatchesTheDuckworthLewisMethod/) | |
| [Dedekind Cut](https://demonstrations.wolfram.com/DedekindCut/) | circles at every distinct rational below the cut |
| [Dynamics of a Spring-Pendulum System](https://demonstrations.wolfram.com/DynamicsOfASpringPendulumSystem/) | `NDSolveValue` with an algebraic constraint |
| [Force to Overcome Vacuum Pull](https://demonstrations.wolfram.com/ForceToOvercomeVacuumPull/) | `Column` of `Show[Plot[…], Graphics[…]]` panels with styled `FrameLabel`s |
| [Goldbach Conjecture](https://demonstrations.wolfram.com/GoldbachConjecture/) | `ListPlot` with explicit ticks |
| [Gravestone from Transformation of Bilinski Dodecahedron 2](https://demonstrations.wolfram.com/GravestoneFromTransformationOfBilinskiDodecahedron2/) | compressed texture data |
| [Inscribed Angles That Intercept the Same Arc](https://demonstrations.wolfram.com/InscribedAnglesThatInterceptTheSameArc/) | |
| [Merging Schools of Fish](https://demonstrations.wolfram.com/MergingSchoolsOfFish/) | |
| [Miscible Displacement of Oil in Heterogenous Porous Media](https://demonstrations.wolfram.com/MiscibleDisplacementOfOilInHeterogenousPorousMedia/) | typeset partial-derivative operators |
| [Non Placet Net of a Dodecahedron](https://demonstrations.wolfram.com/NonPlacetNetOfADodecahedron/) | |
| [Plot a Quadratic Inequality](https://demonstrations.wolfram.com/PlotAQuadraticInequality/) | shaded inequality region |
| [Sampling a Digital Signal](https://demonstrations.wolfram.com/SamplingADigitalSignal/) | `ListPlot`s with `Filling -> Axis` driven by an `Initialization` block |
| [Some Irreptiles of Order Greater than 20](https://demonstrations.wolfram.com/SomeIrreptilesOfOrderGreaterThan20/) | data table with omitted (`Null`) elements |
| [Stochastic Model of Microbial Injury and Mortality](https://demonstrations.wolfram.com/StochasticModelOfMicrobialInjuryAndMortality/) | in-place part assignment inside `Manipulate` |
| [The Mayan Calendar](https://demonstrations.wolfram.com/TheMayanCalendar/) | wheel of `Disk` sectors carrying `Inset` pictures |
| [The Price of a Call Option on Electrical Power](https://demonstrations.wolfram.com/ThePriceOfACallOptionOnElectricalPower/) | |
| [Trigonometric Sums as Parametric Curves](https://demonstrations.wolfram.com/TrigonometricSumsAsParametricCurves/) | typeset `∑` boxes in the initialization cells |
| [Two Circular Windows](https://demonstrations.wolfram.com/TwoCircularWindows/) | |

## Partially supported

| Demonstration | What differs |
| --- | --- |
| [Nets for Regular Spherical Models](https://demonstrations.wolfram.com/NetsForRegularSphericalModels/) | its net templates match, but the solid view asks for `ViewAngle -> 20 Degree` and Woxi projects orthographically, so that view draws about 1.3x smaller |
| [Thermodynamic Consistency Test Based on Differential Residuals](https://demonstrations.wolfram.com/ThermodynamicConsistencyTestBasedOnDifferentialResiduals/) | every point, curve, marker glyph and the ±0.1 acceptance band match the published snapshots. Two things still differ: the pressure axis is labelled every 10 kPa where the Wolfram Language labels it every 5 (its automatic step falls between two "nice" values, and Woxi rounds to the coarser one), and the consistency test's left frame label keeps only the plain text of its `Row`, dropping the two `TraditionalForm` expressions around it |

## Widget covered

Only the `Manipulate` of these is tested, not the notebook around it: the
controls it builds and the picture it draws.

| Demonstration | Exercises |
| --- | --- |
| [A Word Problem about Boats](https://demonstrations.wolfram.com/AWordProblemAboutBoats/) | a slider the body tracks |
| [Center of Mass of a Polygon](https://demonstrations.wolfram.com/CenterOfMassOfAPolygon/) | `Locator` control |
| [Evolution in a Cellular-Automaton Model of Gray-Scott Reaction-Diffusion System](https://demonstrations.wolfram.com/EvolutionInACellularAutomatonModelOfGrayScottReactionDiffusi/) | `Trigger`, a reset `Button`, and a body that mutates simulation state |
| [Kepler's Second Law](https://demonstrations.wolfram.com/KeplersSecondLaw/) | `Trigger` and sliders bounded by another control |
| [Lorentz Oscillator Model for Optical Constants](https://demonstrations.wolfram.com/LorentzOscillatorModelForOpticalConstants/) | ten sliders, and tick labels in scientific notation |
| [Mandelbrot Set Print](https://demonstrations.wolfram.com/MandelbrotSetPrint/) | |
| [Nets for Polyhedral Approximations of the Sphericon](https://demonstrations.wolfram.com/NetsForPolyhedralApproximationsOfTheSphericon/) | controls grouped in a `Row`; its solid view has the same `ViewAngle` difference as "Nets for Regular Spherical Models" |
| [Oscilloscope with Two Signal Inputs](https://demonstrations.wolfram.com/OscilloscopeWithTwoSignalInputs/) | in-body `Locator` and `TogglerBar` |
| [Power of a Test about a Binomial Parameter](https://demonstrations.wolfram.com/PowerOfATestAboutABinomialParameter/) | labels carrying inline linear-syntax boxes |
| [Quicksort versus Selection Sort](https://demonstrations.wolfram.com/QuicksortVersusSelectionSort/) | |
| [Sliding the Roots of Cubics](https://demonstrations.wolfram.com/SlidingTheRootsOfCubics/) | |
| [Triangle Calculator](https://demonstrations.wolfram.com/TriangleCalculator/) | every variable held by a `Toggler` |
