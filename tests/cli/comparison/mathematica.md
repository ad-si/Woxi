---
icon: lucide/scale
---

# Comparison with Mathematica

[WolframScript] is the official command line interface for the Wolfram Language
provided by [Wolfram Research], the company behind the Wolfram Language.

[Mathematica] is the official frontend with a notebook interface.
It is implemented as a cross-platform desktop application
and is available for macOS, Linux, and Windows.

[WolframScript]: https://www.wolfram.com/wolframscript/
[Wolfram Research]: https://www.wolfram.com
[Mathematica]: https://www.wolfram.com/mathematica/

<dl>
  <dt>Implementation Language</dt>
  <dd>C++</dd>
  <dt>First Release</dt>
  <dd>1988</dd>
  <dt>License</dt>
  <dd>Proprietary</dd>
</dl>

Woxi is our alternative to WolframScript and
Woxi Studio is our alternative to Mathematica.
They try to be as compatible as possible, but there are a few features,
they intentionally deviate from to provide a better user experience.


## WolframScript vs Woxi

- **Woxi supports Unicode characters** \
    For example to calculate the circumference of a circle with radius 4:
    ```sh
    woxi eval 'N[2π * 4]'
    ```


## Mathematica vs Woxi Studio

- **Woxi Studio does not support out of order evaluation of cells** \
    When running a cell, it automatically also runs all cells before it.
    This is to avoid confusion about the state of the kernel
    and ensures consistent results when working with notebooks.

- **Woxi Studio does not support `%`** \
    This is too brittle as it refers to the last calculation that was evaluated,
    which could be any notebook cell and therefore leads to confusion
    about the state of the kernel.
    If you want to reuse results, assign them to a variable.

- **Mostly not implemented yet**
    - [Wolfram Knowledgebase](https://www.wolfram.com/language/core-areas/knowledgebase/) \
        This includes functions like:
        - `WolframAlpha[]`
        - Built-in `Entity[]` objects
        - Natural language input with `ctrl =`
        - Most functions listed on
            http://reference.wolfram.com/language/guide/KnowledgeRepresentationAndAccess.html
    - [Machine Learning and Neural Networks](https://www.wolfram.com/language/core-areas/machine-learning/)
    - [Optimization](https://www.wolfram.com/language/core-areas/optimization/)
    - [FEM](https://www.wolfram.com/language/core-areas/fem/)
    - [Chemistry](https://www.wolfram.com/language/core-areas/chemistry/)
    - [Audio Computation](https://www.wolfram.com/language/core-areas/audio/)
    - [Video Computation](https://www.wolfram.com/language/core-areas/video/)
    - [Geography](https://www.wolfram.com/language/core-areas/geography/)
    - [Astronomy](https://www.wolfram.com/language/core-areas/astronomy/)
    - [Control Systems](https://www.wolfram.com/language/core-areas/controls/)
    - [Signal Processing](https://www.wolfram.com/language/core-areas/signal/)
    - [Tools for AIs](https://www.wolfram.com/artificial-intelligence/)


## Missing features by Mathematica release

Woxi targets a *subset* of the Wolfram Language, so some of the
several-hundred functions added in each
[Mathematica release](https://writings.stephenwolfram.com/version-release/)
are not (yet) implemented.
The lists below highlight the marquee feature areas of each version that
Woxi does **not** support.

### Version 15.0 (2026)

- Built-in AI assistant and `Wolfram Agent Tools` framework
- Rebuilt time-series engine: `EventSeries`, `TimeSeriesEvents`,
    `EventSeriesAccumulate`
- `ModelFit` superfunction with `ExponentialModel`, `PowerModel`,
    `LogModel`, `PolynomialModel`, `PeriodicModel`, `DecisionTreeModel`
- Big-data `Tabular` enhancements, `TabularSummary`
- `Around` data type
- Exception handling: `ThrowException`, `CatchExceptions`,
    `RegisterExceptionType`
- Lazy sequences via `IncrementalObject` and incremental
    `Permutations` / `Subsets` / `Tuples`
- Visualization: `PlotGrid`, `BubbleHistogram`, `PeriodicTablePlot`
- WebSocket connectivity
- GPU/CUDA kernels

### Version 14.3 (2025)

- Dark mode: `LightDarkSwitched`, `SystemColor`, `ThemeColor`
- Data fitting: `ListFitPlot`, `ListFitPlot3D`, `LocalModelFit` (LOESS),
    `KernelModelFit`
- Surface and mesh processing: `SurfaceDensityPlot3D`, `SmoothMesh`,
    `SimplifyMesh`, `Remesh`, `SubdivisionRegion`
- Geodesics: `FindShortestCurve`, `ShortestCurveDistance`
- Non-commutative algebra: `NonCommutativeExpand`, `Commutator`,
    `AntiCommutator`
- `HilbertTransform`
- Lommel functions `LommelS1`/`LommelS2`/`LommelT1`/`LommelT2`
- Linear algebra: `EigenvalueDecomposition`, `FrobeniusDecomposition`,
    `MatrixMinimalPolynomial`
- `LLMGraph` for agentic workflows
- Many new database connectors

### Version 14.2 (2025)

- `Tabular` big-data subsystem: `ToTabular`, `AggregateRows`,
    `PivotTable`, `TransformColumns`
- Chat cells in notebooks
- Symbolic arrays: `ArrayExpand`, `ArraySimplify`, `ComponentExpand`
- Game theory: `MatrixGame`, `FindMatrixGameStrategies`, `GameTheoryData`
- Video object tracking: `VideoObjectTracking`, `ImageBoundingBoxes`
- `GPUArray` GPU-native arrays
- Astronomy: `FindAstroEvent`
- `ParallelSelect`, `ParallelCases`
- `Failsafe`
- `MidDate`

### Version 14.1 (2024)

- LLM integration: `LLMPromptGenerator`, `LLMConfiguration`
- Vector search: `SemanticSearch`, `CreateSemanticSearchIndex`,
    `VectorDatabaseSearch`
- Symbolic arrays: `MatrixSymbol`, `ArraySymbol`, `ArrayDot`
- `PascalBinomial`
- `DStabilityConditions`
- Biomolecules: `BioMolecule`, `BioMoleculePlot3D`
- `AstroGraphics`
- `PolarCurve`, `FilledPolarCurve`
- Video generation: `ManipulateVideo`, `VideoTranscribe`, `SpeechRecognize`

### Version 14.0 (2024)

- Chat Notebooks and LLM tooling
- Calculus: `ImplicitD`, `LineIntegrate`, `SurfaceIntegrate`,
    `ContourIntegrate`, fractional differentiation
- Video as a first-class object: `VideoJoin`, `OverlayVideo`
- Astronomy: `AstroPosition`, `AstroGraphics`
- Chemistry: `Molecule`, `ChemicalFormula`, `ReactionBalance`
- Symbolic finite-field arithmetic and factoring
- Solid-mechanics / fluid-dynamics PDEs
- Computable species data
- Synthetic geometry constraint solving
- Graphics: texture mapping, haloing, `DropShadowing`

### Version 13.3 (2023)

- LLM suite: `LLMFunction`, `LLMSynthesize`, `LLMTool`,
    `LLMExampleFunction`, Chat Notebooks
- Appell functions: `AppellF2`, `AppellF3`, `AppellF4`
- `FiniteField` arithmetic
- Region metrics: `RegionDistance`, `RegionHausdorffDistance`,
    `InscribedBall`, `CircumscribedBall`
- `PlotHighlighting`, `Haloing`, `Highlighted`
- `ImageSynthesize` (text-to-image)
- `FindImageShapes`
- Test framework: `TestCreate`, `TestObject`, `TestReport`
- Foreign function interface: `ForeignFunctionLoad`, `RawMemoryAllocate`
- `ARPublish`

### Version 13.2 (2022)

- Astronomy: `AstroPosition`, `AstroDistance`, `AstroAngularSeparation`
- Multivariate polynomial factoring over finite fields
- Temperature-difference units
- `RandomTime`
- `ClusteringMeasurements`
- `NetExternalObject` (ONNX)
- `TerminatedEvaluation`
- `TypeHint`
- Chess format support (FEN/PGN)

### Version 13.1 (2022)

- `Threaded` construct
- Compiler enhancements
- Full 32-bit Unicode and emoji support
- Fractional calculus: `FractionalD`, `CaputoD`, `ImplicitD`,
    `IntegrateChangeVariables`, `DSolveChangeVariables`
- Data ops: `UniqueElements`, `ReplaceAt`
- Chemistry: `PatternReaction`, `ApplyReaction`, `ChemicalConvert`
- Geometry: `ReconstructionMesh`, 3D `VoronoiMesh`, `GeodesicPolyhedron`
- `TernaryListPlot`
- `VideoCapture`, `VideoScreenCapture`

### Version 13.0 (2021)

- Solid mechanics: `SolidMechanicsPDEComponent`, `SolidMechanicsStress`
- Matrix ops: `DrazinInverse`, `FunctionPoles`, `CenteredInterval`
- Geometry: `RegionFit`, `ConcaveHullMesh`, `CSGRegion`,
    `FindRegionTransform`
- Graph theory: `FindVertexColoring`, `VertexChromaticNumber`,
    `FindSubgraphIsomorphism`, `PlanarFaceList`, `DominatorTreeGraph`
- Chemistry: `ChemicalReaction`, `ReactionBalance`, `FindIsomers`
- Spatial estimation: `SpatialEstimate`, `VariogramModel`
- Video composition: `TourVideo`, `GridVideo`
- Symbolic lighting (`PointLight`, `SpotLight`)
- Quizzes: `QuestionObject`, `AssessmentFunction`

### Version 12.3 (2021)

- Multivariate transcendental equation solving
- Symbolic PDE solutions
- Data structures: `ByteTrie`, `KDTree`, `ImmutableVector`
- `GeometricTest`
- Region dilation/erosion
- `StreamPlot3D`, `ListStreamPlot3D`
- Video editing: `VideoRecord`, `VideoInsert`, `VideoReplace`
- Carlson elliptic integrals, Fox H-function

### Version 12.2 (2020)

- Biomolecular sequences: `BioSequence`, `BioSequenceTranslate`,
    `BioSequenceComplement`
- Spatial statistics: `SpatialPointData`, `MeanPointDensity`,
    `SpatialRandomnessTest`
- PDE term framework: `LaplacianPDETerm`, `DiffusionPDETerm`,
    `HelmholtzPDEComponent`
- Interactive Euclidean geometry
- 37 new calendar systems

### Version 12.1 (2020)

- `Video` for frame extraction and analysis
- HiDPI / Metal / Direct3D rendering
- `DataStructure` (linked lists, binary trees, hash tables, stacks)
- Heun functions
- `CategoricalDistribution`
- `GeometricOptimization` (convex problems)
- Neural net types BERT and GPT-2
- `NetGANOperator`
- `MoleculeRecognize`
- `WikidataData`

### Version 12.0 (2019)

- Complex plotting: `ComplexPlot`, `ComplexPlot3D`, `ReImPlot`
- Euclidean geometry automation: `GeometricScene`, `RandomInstance`,
    `FindGeometricConjectures`, `TriangleCenter`
- Theorem proving: `AxiomaticTheory`, `FindEquationalProof`
- Machine learning: `LearnDistribution`, `FindAnomalies`, `AttentionLayer`
- Recognition: `ImageCases`, `ImageContents`, `AudioIdentify`,
    `PitchRecognize`
- Chemistry: `Molecule`, `MoleculePlot3D`, `FindMoleculeSubstructure`
- NLP: `TextCases`, `TextContents`, `Synonyms`, `Antonyms`
- `Iconize`

### Version 11.3 (2018)

- Blockchain: `BlockchainData`, `BlockchainTransactionData`
- `AsymptoticDSolveValue`
- `FindTextualAnswer`
- `FindFaces`, `FacialFeatures`
- Presentation environment
- `SideNotes`, `SideCode`
- Mail: `SendMessage`, `MailServerConnect`, `MailSearch`
- Neural net surgery: `NetTake`, `NetJoin`, `NetFlatten`

### Version 11.2 (2017)

- `ImageRestyle` (style transfer)
- Improved `Classify` / `Predict`
- `GeoImage` (satellite imagery)
- `TideData`
- `StackedListPlot`, `StackedDateListPlot`
- `AnatomyPlot3D`
- `RegionIntersection` (constructive solid geometry)
- `SpeechSynthesize`
- `AudioStream`
- `ExternalEvaluate` (Python, NodeJS)
- `TaskObject`

### Version 11.1 (2017)

- 30 new neural net layer types
- `NetModel`
- `FeatureSpacePlot`
- `AudioCapture`
- `Cepstrogram`
- `ImageGraphics` (bitmap to vector)
- `GeoBubbleChart`
- `WebSearch`, `WebImageSearch`, `TextTranslation`
- `HilbertCurve`, `SierpinskiMesh`, `SpherePoints`, `AnglePath3D`
- `DateObject` granularity
- `PersistentValue`

### Version 11.0 (2016)

- `Printout3D` (3D printing) with automatic mesh repair
- Neural networks: `ImageIdentify`, `NetChain`, `ConvolutionLayer`
- `Audio` as a first-class type
- Routing: `TravelDirections`, `TravelTime`
- Differential operators: `DEigenvalues`, `GreenFunction`, `MellinTransform`
- `Channel` publish-subscribe framework
- Cryptography
- `HTTPRequest`
- Visualization: `TimelinePlot`, `Dendrogram`, `GeoHistogram`, `AudioPlot`
- Text: `TextCases`, `Transliterate`

### Version 10.0 (2014)

- Machine learning: `Classify`, `Predict`
- Finite-element method for PDEs (`NDSolve` `"FiniteElement"`)

### Version 9.0 (2012)

- Predictive interface and Suggestions Bar (notebook UI)
- `SocialMediaData`
- Random / stochastic processes: `MarkovProcess`, `QueueingProcess`,
    `ARProcess`
- Survival analysis
- Symbolic tensors (`TensorReduce`, `TensorExpand`)
- Image recognition: `FindFaces`, `ImageFeatureTrack`
- Control systems design

### Version 8.0 (2010)

- Free-form linguistic input via Wolfram|Alpha
- GPU computing: `CUDAFunction` (CUDALink / OpenCLLink)
- C code generation: `CCodeGenerate`, `CompileToC`
- Financial engineering: `FinancialDerivative`, `FinancialData`
- Wavelet analysis: `WaveletTransform`, `DiscreteWaveletTransform`

### Version 7.0 (2008)

- Built-in curated data (genomic, weather, astronomical, chemical)
- Delay differential equations in `NDSolve`
- Automatic charting superfunctions and computational typesetting

### Version 6.0 (2007)

- Dynamic UI panes: `Animate`, `Animator`, `LocatorPane`, `ClickPane`
- Live mouse-driven manipulation of 2D/3D graphics in notebooks

### Version 5.0 (2003)

- Arbitrary-precision numerics engine and packed-array performance optimizations

### Version 4.0 (1999)

- Packed arrays — an internal machine-number storage optimization
    (Woxi computes the same results without this representation)

### Version 3.0 (1996)

- Interactive 2D mathematical typesetting / notation input (front-end)

### Version 2.0 (1991)

- `MathLink` protocol for external C / Fortran programs

### Version 1.0 (1988)

The original 554 built-in functions and the symbolic computation core are
fully supported.
The interactive notebook front-end is provided separately by Woxi Studio.
