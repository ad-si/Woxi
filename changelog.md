# Changelog

# Unreleased

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
