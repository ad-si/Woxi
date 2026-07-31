# Wolfram Demonstrations Project

Notebooks from the
[Wolfram Demonstrations Project](https://demonstrations.wolfram.com)
that Woxi Studio opens and runs end to end:
the initialization cells evaluate, the `Manipulate` builds every control,
and the body draws at each control setting.

The notebooks themselves are not part of this repo — they are published under
the [Wolfram Demonstrations Project license][license].
Each one is pinned by a Studio test that inlines the cell's box source, so the
coverage below is checked by `make test` rather than by fetching the notebook.

[license]: https://demonstrations.wolfram.com/licensing.html

## Supported notebooks

| Demonstration | Test |
| --- | --- |
| [A Converging Geometric Series](https://demonstrations.wolfram.com/AConvergingGeometricSeries/) | `geometric_series_notebook_opens_with_its_widget` |
| [An Expanding Structure Based on the Diamond Lattice](https://demonstrations.wolfram.com/AnExpandingStructureBasedOnTheDiamondLattice/) | `diamond_lattice_notebook_opens_with_its_widget` |
| [Balanced Ternary Notation](https://demonstrations.wolfram.com/BalancedTernaryNotation/) | `balanced_ternary_notebook_opens_with_its_widget` |
| [Constant Price Elasticity of Demand](https://demonstrations.wolfram.com/ConstantPriceElasticityOfDemand/) | `price_elasticity_notebook_opens_with_its_widget` |
| [Dedekind Cut](https://demonstrations.wolfram.com/DedekindCut/) | `dedekind_cut_notebook_draws_its_circles` |
| [Force to Overcome Vacuum Pull](https://demonstrations.wolfram.com/ForceToOvercomeVacuumPull/) | `vacuum_pull_notebook_opens_with_its_widget` |
| [Goldbach Conjecture](https://demonstrations.wolfram.com/GoldbachConjecture/) | `goldbach_notebook_opens_with_its_widget` |
| [Gravestone from Transformation of Bilinski Dodecahedron 2](https://demonstrations.wolfram.com/GravestoneFromTransformationOfBilinskiDodecahedron2/) | `gravestone_notebook_loads_its_compressed_texture` |
| [Regular Polygon Rolling on a Catenary](https://demonstrations.wolfram.com/RegularPolygonRollingOnACatenary/) | `rolling_polygon_on_catenary_notebook_rolls_its_polygon` |
| [Sampling a Digital Signal](https://demonstrations.wolfram.com/SamplingADigitalSignal/) | `sampling_a_digital_signal_notebook_builds_its_widget` |
| [Some Irreptiles of Order Greater than 20](https://demonstrations.wolfram.com/SomeIrreptilesOfOrderGreaterThan20/) | `irreptiles_notebook_opens_with_its_widget` |
| [Stochastic Model of Microbial Injury and Mortality](https://demonstrations.wolfram.com/StochasticModelOfMicrobialInjuryAndMortality/) | `microbial_injury_notebook_builds_its_widget` |
| [The Mayan Calendar](https://demonstrations.wolfram.com/TheMayanCalendar/) | `mayan_calendar_notebook_opens_with_its_widget` |
| [Trigonometric Sums as Parametric Curves](https://demonstrations.wolfram.com/TrigonometricSumsAsParametricCurves/) | `trigonometric_sums_notebook_opens_with_its_widget` |
| [Two Circular Windows](https://demonstrations.wolfram.com/TwoCircularWindows/) | `two_circular_windows_notebook_opens_with_its_widget` |

Further Demonstrations are covered by their `Manipulate` alone rather than by
the whole notebook — see the `woxi-studio` tests naming them.

## Adding a notebook

1. Download the notebook from its Demonstration page
   ("Download Source Notebook").
2. Open it in Woxi Studio and compare it against the Demonstration's own
   snapshots at several control settings.
3. Fix whatever diverges, in the interpreter or in Studio — never for the one
   notebook, always for the construct it exercises.
4. Add a Studio test that inlines the cell's box source, then list the
   Demonstration above.
