---
icon: lucide/history
---

# Missing features by Mathematica release

Woxi targets a *subset* of the Wolfram Language, so some of the
several-hundred functions added in each
[Mathematica release](https://writings.stephenwolfram.com/version-release/)
are not (yet) implemented.
The lists below highlight the marquee feature areas of each version that
Woxi does **not** support.

## Version 15.0 (2026)

- Built-in AI assistant and `Wolfram Agent Tools` framework
- `TimeSeriesEvents` from the rebuilt time-series engine
    (`EventSeries` and `EventSeriesAccumulate` are supported)
- `ModelFit` superfunction with `ExponentialModel`, `PowerModel`,
    `LogModel`, `PolynomialModel`, `PeriodicModel`, `DecisionTreeModel`
- Big-data `Tabular` enhancements, `TabularSummary`
- Exception handling: `ThrowException`, `CatchExceptions`,
    `RegisterExceptionType`
- Lazy sequences via `IncrementalObject` and incremental
    `Permutations` / `Subsets` / `Tuples`
- WebSocket connectivity
- GPU/CUDA kernels
- Multiple polylogarithms and zeta values: `MultiplePolyLog`,
    `MultipleZeta`, `GeneralizedPolyLog`, `HarmonicPolyLog`
- Matrix decompositions: `PolarDecomposition`,
    `BunchKaufmanDecomposition`, `OrderedSchurDecomposition`,
    `PopovDecomposition`
- Symbolic non-commutative algebras: `CliffordAlgebra`,
    `GrassmannAlgebra`, `WeylAlgebra`, `NonCommutativePolynomialReduction`
- Structured partial fractions: `PartialFractions`,
    `PartialFractionElements`
- Music computation: `MusicMeasurements`, `MusicTransform`
- Astronomy: `FindSolarEclipse`, `OrbitalElements`
- Dark-mode notebook theming: `NotebookTheme`, `DarkModePane`,
    `LightModePane`
- `SystemModelSurrogate` model-order reduction

## Version 14.3 (2025)

- Data fitting: `ListFitPlot`, `ListFitPlot3D`, `LocalModelFit` (LOESS),
    `KernelModelFit`
- Surface and mesh processing: `SurfaceDensityPlot3D`, `SmoothMesh`,
    `SimplifyMesh`, `Remesh`, `SubdivisionRegion`
- Non-commutative algebra: `NonCommutativeExpand`, `Commutator`,
    `AntiCommutator`
- `HilbertTransform`
- Lommel functions `LommelS1`/`LommelS2`/`LommelT1`/`LommelT2`
- Linear algebra: `EigenvalueDecomposition`, `FrobeniusDecomposition`
- `LLMGraph` for agentic workflows
- Many new database connectors

## Version 14.2 (2025)

- `Tabular` big-data operations: `AggregateRows`, `PivotTable`,
    `TransformColumns` (the `Tabular` object itself is supported)
- Chat cells in notebooks
- Symbolic arrays: `ArrayExpand`, `ArraySimplify`, `ComponentExpand`
- Game theory: `MatrixGame`, `FindMatrixGameStrategies`, `GameTheoryData`
- Video object tracking: `VideoObjectTracking`, `ImageBoundingBoxes`
- `GPUArray` GPU-native arrays
- Astronomy: `FindAstroEvent`
- `Failsafe`

## Version 14.1 (2024)

- LLM integration: `LLMPromptGenerator`, `LLMConfiguration`
- Vector search: `SemanticSearch`, `CreateSemanticSearchIndex`,
    `VectorDatabaseSearch`
- Symbolic arrays: `MatrixSymbol`, `ArraySymbol`
- `DStabilityConditions`
- Biomolecules: `BioMolecule`, `BioMoleculePlot3D`
- `AstroGraphics`
- Video generation: `ManipulateVideo`, `VideoTranscribe`, `SpeechRecognize`
- Diff framework: `DiffObject`, `Diff3`, `DiffApply`, `DiffGranularity`
- Fixed points and stability of difference/differential equations:
    `DFixedPoints`, `RFixedPoints`, `RStabilityConditions`
- `AstroRiseSet`
- `FreeformEvaluate` (natural-language evaluation)
- Video authoring: `ConstantVideo`, `SowVideo`, `ReapVideo`,
    `VideoFrameFold`

## Version 14.0 (2024)

- Chat Notebooks and LLM tooling
- Calculus: `ImplicitD`, `LineIntegrate`, `SurfaceIntegrate`,
    `ContourIntegrate`, fractional differentiation
- Video as a first-class object: `VideoJoin`, `OverlayVideo`
- Astronomy: `AstroPosition`, `AstroGraphics`
- Chemistry: `ChemicalFormula`, `ReactionBalance`
- Symbolic finite-field arithmetic and factoring
- Solid-mechanics / fluid-dynamics PDEs
- Computable species data
- Synthetic geometry constraint solving
- Graphics: texture mapping
- Numeric integral superfunctions: `NContourIntegrate`, `NLineIntegrate`,
    `NSurfaceIntegrate`
- Pairwise statistical visualization: `PairwiseListPlot`,
    `PairwiseDensityHistogram`, `PairwiseQuantilePlot`
- Structured-matrix constructors: `HermitianMatrix`, `OrthogonalMatrix`,
    `SymmetricMatrix`, `UnitaryMatrix`
- Multiphysics PDE components: `ElectrostaticPDEComponent`,
    `FluidFlowPDEComponent`, `SchrodingerPDEComponent`
- `TextSummarize`
- Lunar-phase arithmetic: `LunationNumber`, `FromLunationNumber`

## Version 13.3 (2023)

- LLM suite: `LLMFunction`, `LLMSynthesize`, `LLMTool`,
    `LLMExampleFunction`, Chat Notebooks
- `FiniteField` arithmetic
- Region metrics: `RegionHausdorffDistance`, `InscribedBall`
    (`CircumscribedBall` is supported)
- `PlotHighlighting`
- `ImageSynthesize` (text-to-image)
- `FindImageShapes`
- Test framework: `TestCreate`, `TestObject`, `TestReport`
- Foreign function interface: `ForeignFunctionLoad`, `RawMemoryAllocate`
- `ARPublish`
- Subkernel evaluation: `KernelEvaluate`, `KernelConfigurationEdit`
- Sphere geometry: `SphericalDistance`, `RegionFarthestDistance`
- `SystemModelCalibrate`

## Version 13.2 (2022)

- Astronomy: `AstroPosition`, `AstroDistance`, `AstroAngularSeparation`
- Multivariate polynomial factoring over finite fields
- Temperature-difference units
- `ClusteringMeasurements`
- `NetExternalObject` (ONNX)
- `TerminatedEvaluation`
- `TypeHint`
- Chess format support (FEN/PGN)
- `AstroGraphics` customization: `AstroBackground`, `AstroProjection`,
    `AstroGridLines`, `AstroReferenceFrame`
- Numeric fractional calculus: `NFractionalD`, `NCaputoD`
- Standalone compiled components: `CompiledComponent`,
    `BuildCompiledComponent`
- `FileSystemTree`

## Version 13.1 (2022)

- Compiler enhancements
- Full 32-bit Unicode and emoji support
- Fractional calculus: `FractionalD`, `ImplicitD`,
    `IntegrateChangeVariables`, `DSolveChangeVariables`
    (`CaputoD` is supported)
- Chemistry: `PatternReaction`, `ApplyReaction`, `ChemicalConvert`
- Geometry: `ReconstructionMesh`, 3D `VoronoiMesh`, `GeodesicPolyhedron`
- `VideoCapture`, `VideoScreenCapture`
- `Until` loop construct
- Graph operations: `GraphJoin`, `GraphProduct`, `GraphSum`,
    `BuckyballGraph`
- Compiled type system and raw pointers: `TypeDeclaration`,
    `CreateTypeInstance`, `Cast`, `ToRawPointer`,
    `LibraryFunctionDeclaration`
- Polygon shading: `PhongShading`, `GouraudShading`, `FlatShading`
- Feature-impact plots for ML models: `FeatureImpactPlot`,
    `CumulativeFeatureImpactPlot`
- `ModelPredictiveController`
- Inert expressions: `InertExpression`, `InertEvaluate`
- `ResidueSum`

## Version 13.0 (2021)

- Solid mechanics: `SolidMechanicsPDEComponent`, `SolidMechanicsStress`
- Matrix ops: `FunctionPoles`
- Geometry: `RegionFit`, `ConcaveHullMesh`, `CSGRegion`,
    `FindRegionTransform`
- Graph theory: `FindVertexColoring`, `VertexChromaticNumber`,
    `FindSubgraphIsomorphism`, `PlanarFaceList`, `DominatorTreeGraph`
- Chemistry: `ChemicalReaction`, `ReactionBalance`, `FindIsomers`
- Spatial estimation: `SpatialEstimate`, `VariogramModel`
- Video composition: `TourVideo`, `GridVideo`
- Symbolic lighting (`PointLight`, `SpotLight`)
- Edge coloring and subgraph isomorphism: `FindEdgeColoring`,
    `EdgeChromaticNumber`, `FindIsomorphicSubgraph`
- Vector displacement plots: `VectorDisplacementPlot`,
    `VectorDisplacementPlot3D`
- `ImageStitch` panorama stitching
- `BilateralZTransform`
- `CompleteIntegral` for first-order PDEs
- `BioSequencePlot`
- Trainable content detectors: `TrainImageContentDetector`,
    `TrainTextContentDetector`

## Version 12.3 (2021)

- Multivariate transcendental equation solving
- Symbolic PDE solutions
- Data structures: `ByteTrie`, `KDTree`, `ImmutableVector`
- Region dilation/erosion
- `StreamPlot3D`, `ListStreamPlot3D`
- Video editing: `VideoRecord`, `VideoInsert`, `VideoReplace`
- Carlson elliptic integrals, Fox H-function
- `Tree` data structure: `Tree`, `NestTree`, `RandomTree`, `RulesTree`,
    tree styling and layout options
- `BilateralLaplaceTransform`
- Compiler environments: `CreateCompilerEnvironment`,
    `FunctionDeclaration`
- Graph geo/3D layouts: `GeoGraphPlot`, `LayeredGraphPlot3D`
- Molecule comparison: `MoleculeAlign`, `MoleculeName`,
    `MoleculeMaximumCommonSubstructure`
- `SolarTime`, `GeoOrientationData`
- `DatasetTheme`
- `PersistentSymbol`
- Tick styling options: `TickLabels`, `TickPositions`, `TickDirection`

## Version 12.2 (2020)

- Biomolecular sequences: `BioSequence`, `BioSequenceTranslate`,
    `BioSequenceComplement`
- Spatial statistics: `SpatialPointData`, `MeanPointDensity`,
    `SpatialRandomnessTest`
- PDE term framework: `LaplacianPDETerm`, `HelmholtzPDEComponent`
- Interactive Euclidean geometry
- 37 new calendar systems
- Convex optimization umbrella: `ConvexOptimization`,
    `ParametricConvexOptimization`, `RobustConvexOptimization`
- Real-function properties: `FunctionConvexity`, `FunctionMonotonicity`,
    `FunctionInjective`, `FunctionSingularities`,
    `FunctionDiscontinuities`
- Point process models and statistics: `PoissonPointProcess`,
    `MaternPointProcess`, `StraussPointProcess`, `RipleyK`,
    `NearestNeighborG`
- Combinator calculus: `CombinatorS`, `CombinatorK`, `CombinatorY`
- Lamé functions: `LameC`, `LameS`, `LameEigenvalueA`
- New visualization types: `RadialAxisPlot`, `ParallelAxisPlot`,
    `PointValuePlot`, `ComplexArrayPlot`, `ArrayPlot3D`, `AnimatedImage`
- Video processing: `VideoMap`, `VideoCombine`, `VideoTranscode`,
    `VideoGenerator`
- Cloud batch computation: `RemoteBatchSubmit`, `RemoteEvaluate`
- Convex-hull regions: `ConvexHullRegion`, `ConvexRegionQ`
- `FaceRecognize`
- Asymptotic statistics: `AsymptoticExpectation`, `AsymptoticProbability`

## Version 12.1 (2020)

- `Video` for frame extraction and analysis
- HiDPI / Metal / Direct3D rendering
- `DataStructure` (linked lists, binary trees, hash tables, stacks)
- Heun functions
- `CategoricalDistribution`
- `GeometricOptimization` (convex problems)
- Neural net types BERT and GPT-2
- `NetGANOperator`
- `MoleculeRecognize`
- Paclet repository management: `PacletInstall`, `PacletFind`,
    `PacletObject`, `PacletSite` functions
- Complex-function plots: `ComplexContourPlot`, `ComplexStreamPlot`,
    `ComplexVectorPlot`
- Artistic shading: `HatchShading`, `HalftoneShading`, `StippleShading`,
    `GoochShading`, `ToonShading`
- Geo field plots: `GeoContourPlot`, `GeoDensityPlot`
- Speech analysis: `SpeechCases`, `SpeechInterpreter`, `SpeakerMatchQ`
- `FindImageText`
- `TableView`
- External storage services (S3, IPFS): `ExternalStorageUpload`,
    `ExternalStorageDownload`
- OS credential store: `SystemCredential`
- `WikidataSearch`

## Version 12.0 (2019)

- Euclidean geometry automation: `RandomInstance`,
    `FindGeometricConjectures` (`GeometricScene` itself is supported)
- Theorem proving: `AxiomaticTheory`, `FindEquationalProof`
- Machine learning: `LearnDistribution`, `FindAnomalies`, `AttentionLayer`
- Recognition: `ImageCases`, `ImageContents`, `AudioIdentify`,
    `PitchRecognize`
- Chemistry: `MoleculePlot3D`, `FindMoleculeSubstructure`
- NLP: `TextContents`, `Synonyms`, `Antonyms`, and most `TextCases`
    content types (basic classes like `"Word"` work)
- `Iconize`
- Convex optimization solvers: `LinearOptimization`,
    `QuadraticOptimization`, `SemidefiniteOptimization`,
    `ConicOptimization`, `SecondOrderConeOptimization`
- Compiler: `FunctionCompile`, `CompiledCodeFunction`,
    `FunctionCompileExport`
- Relational databases and entity-class algebra: `RelationalDatabase`,
    `DatabaseReference`, `FilteredEntityClass`, `AggregatedEntityClass`,
    `SortedEntityClass`
- Computational polygon/polyhedron operations: `RandomPolygon`,
    `RandomPolyhedron`, `DualPolyhedron`, `BeveledPolyhedron`,
    `PolygonDecomposition`, `WindingCount`
- Browser automation: `WebExecute`, `StartWebSession`
- Molecule editing: `MoleculeModify`, `MoleculePattern`, `MoleculeGraph`
- Asymptotic solvers: `AsymptoticSum`, `AsymptoticRSolveValue`
- Digital signatures and derived keys: `GenerateDigitalSignature`,
    `VerifyDigitalSignature`, `GenerateDerivedKey`
- Anomaly and missing-value ML: `AnomalyDetection`, `DeleteAnomalies`,
    `SynthesizeMissingValues`
- Geo vector fields: `GeoVectorPlot`, `GeoStreamPlot`
- `NBodySimulation`
- `AbsArgPlot`
- `InverseSpectrogram`
- Uncertainty containers beyond `Around`: `VectorAround`, `AroundReplace`

## Version 11.3 (2018)

- Blockchain: `BlockchainData`, `BlockchainTransactionData`
- `AsymptoticDSolveValue`
- `FindTextualAnswer`
- `FindFaces`, `FacialFeatures`
- Presentation environment
- `SideNotes`, `SideCode`
- Mail: `SendMessage`, `MailServerConnect`, `MailSearch`
- Neural net surgery: `NetTake`, `NetJoin`, `NetFlatten`
- SystemModeler integration: `SystemModel`, `SystemModelSimulate`,
    `SystemModelPlot`
- Remote execution: `RemoteConnect`, `RemoteRun`, `RemoteRunProcess`
- Network packet capture: `NetworkPacketCapture`,
    `NetworkPacketRecording`
- `ProofObject` proof inspection
- Audio ML: `AudioDistance`, `AudioRecord`
- `FeatureSpacePlot3D`, `GeoSmoothHistogram`
- `EntityPrefetch`
- Nondimensionalization: `NondimensionalizationTransform`,
    `IndependentPhysicalQuantity`

## Version 11.2 (2017)

- `ImageRestyle` (style transfer)
- Improved `Classify` / `Predict`
- `GeoImage` (satellite imagery)
- `TideData`
- `StackedDateListPlot`
- `AnatomyPlot3D`
- `SpeechSynthesize`
- `AudioStream`
- `ExternalEvaluate` (Python, NodeJS)
- `TaskObject` and task control: `SessionSubmit`, `LocalSubmit`,
    `TaskWait`, `TaskSuspend`, `TaskRemove`
- Socket servers: `SocketOpen`, `SocketListen`, `SocketReadMessage`
- `RadonTransform`, `InverseRadonTransform`
- Weierstrass utilities: `WeierstrassHalfPeriodW1`,
    `WeierstrassInvariantG2`, `WeierstrassE1`, `WeierstrassEta1`
    (and siblings)
- Discrete extremal limits: `DiscreteMaxLimit`, `DiscreteMinLimit`
- Byte-array import/export of any format: `ImportByteArray`,
    `ExportByteArray`
- Resource publishing: `ResourceRegister`, `ResourceSubmit`,
    `SecuredAuthenticationKey`
- Screen and notebook capture: `CurrentScreenImage`,
    `CurrentNotebookImage`
- `RegionImage` region rasterization
- `RegisterExternalEvaluator`

## Version 11.1 (2017)

- 30 new neural net layer types
- `NetModel`
- `FeatureSpacePlot`
- `ImageGraphics` (bitmap to vector)
- `GeoBubbleChart`
- `WebSearch`, `WebImageSearch`, `TextTranslation`
- `SierpinskiMesh`, `SpherePoints`
- `PersistentValue`
- Sequence learning: `SequencePredict`, `ActiveClassification`,
    `ActivePrediction`
- `HankelTransform`, `InverseHankelTransform`
- File encryption: `EncryptFile`, `DecryptFile`
- Robust statistics: `SpatialMedian`, `BiweightLocation`,
    `SnDispersion`, `QnDispersion`
- Audio analysis: `AudioLoudness`, `AudioSpectralMap`,
    `AudioSpectralTransformation`
- `MengerMesh` fractal meshes
- Data: `SpectralLineData`, `PsychrometricPropertyData`
- `CurrentDate`

## Version 11.0 (2016)

- `Printout3D` (3D printing) with automatic mesh repair
- Neural networks: `ImageIdentify`, `NetChain`, `ConvolutionLayer`
- Routing: `TravelDirections`, `TravelTime`
- Differential operators: `DEigenvalues`, `GreenFunction`
- `Channel` publish-subscribe framework
- Cryptography
- Audio processing: `AudioResample`, `AudioDelay`, `AudioFade`,
    `AudioReverb`, `AudioPartition`, `AudioChannelMix`
    (the `Audio` object itself is supported)
- Wolfram Data Repository: `ResourceData`, `ResourceObject`,
    `ResourceSearch`
- Neural net training plumbing: `NetTrain`, `NetInitialize`,
    `NetExtract`, and the layer zoo (`LinearLayer`, `PoolingLayer`,
    `DropoutLayer`, …)
- HTTP: `URLDownload`, `URLSubmit`, `Authentication`, cookie management
    (`SetCookies`, `FindCookies`)
- Mesh repair for 3D printing: `RepairMesh`, `FindMeshDefects`,
    `RegionResize`
- Bayesian optimization: `BayesianMinimization`, `BayesianMaximization`
- `KnapsackSolve`
- Feature learning: `FeatureExtract`, `FeatureExtraction`,
    `FeatureDistance`
- `MeijerGReduce`
- Dynamic time warping: `CanonicalWarpingDistance`,
    `CanonicalWarpingCorrespondence`

## Version 10.4 (2016)

- Interactive web forms: `Ask`, `AskFunction`, `AskConfirm`
- Cloud expressions: `CloudExpression`, `CreateCloudExpression`
- Clustering: `ClusterClassify`, `ClusteringTree`
- In-place zoomable images and maps: `DynamicImage`,
    `DynamicGeoGraphics`
- Data: `WordFrequencyData`, `WeatherForecastData`, `UniverseModelData`

## Version 10.3 (2015)

- Random matrix theory: `GaussianUnitaryMatrixDistribution`,
    `TracyWidomDistribution`, `MarchenkoPasturDistribution`,
    `MatrixNormalDistribution`
- Linguistic data: `WordList`, `RandomWord`, `WordDefinition`,
    `WordTranslation`
- Symbolic differential eigenproblems: `DEigensystem`
- Travel: `TravelDistance`, `TravelDirectionsData`
- Low-level networking: `SocketConnect`, `HostLookup`, `IPAddress`
- Data: `AnatomyData`, `MortalityData`, `StandardOceanData`
- `EntityInstance`, `GenerateHTTPResponse`

## Version 10.2 (2015)

- Volumetric slice visualization: `SliceContourPlot3D`,
    `SliceDensityPlot3D`, `SliceVectorPlot3D`, `ListDensityPlot3D`
- `RandomPoint` on regions
- Numeric PDE eigenproblems: `NDEigensystem`, `NDEigenvalues`
- Text search framework: `CreateSearchIndex`, `TextSearch`,
    `ContentObject`
- Cloud sharing: `CloudPublish`, `CloudShare`, `MailReceiverFunction`
- Persistent local storage: `LocalObject`, `LocalSymbol`
- `FindFormula`, `NestGraph`, `DateHistogram`

## Version 10.1 (2015)

- Cryptography: `Encrypt`, `Decrypt`, `GenerateSymmetricKey`,
    `GenerateAsymmetricKeyPair`
- Machine learning: `DimensionReduce`, `LanguageIdentify`,
    `ImageInstanceQ`, `WordStem`
- Data: `WikipediaData`, `GeomagneticModelData`, `GeogravityModelData`,
    `HumanGrowthData`, `StoppingPowerData`
- `QuantityArray`
- `InhomogeneousPoissonProcess`

## Version 10.0 (2014)

- Machine learning: `Classify`, `Predict`
- Finite-element method for PDEs (`NDSolve` `"FiniteElement"`)
- Entity-framework curated data: `CanonicalName`, `CommonName`,
    `ToEntity`, `FromEntity`, and dozens of domain functions
    (`StarData`, `PlanetData`, `AirportData`, `CompanyData`,
    `EarthquakeData`, …)
- Cloud deployment: `CloudDeploy`, `CloudObject`, `CloudGet`, `CloudPut`,
    `APIFunction`, `FormFunction`, `ScheduledTask`, `Permissions`
- Region discretization and meshes: `DiscretizeRegion`,
    `DiscretizeGraphics`, `TriangulateMesh`, `BoundaryMesh`,
    `ParametricRegion`
- Geo mapping: `GeoListPlot`, `GeoRange`, `GeoBackground`, `GeoStyling`,
    `GeoElevationData`
- Device framework: `DeviceOpen`, `DeviceRead`, `DeviceWrite`
- External processes: `RunProcess`, `StartProcess`, `KillProcess`
- Semantic import and interpretation: `SemanticImport`,
    `SemanticInterpretation`, `GrammarRules`, `GrammarApply`
- Templating and reports: `FileTemplateApply`, `DocumentGenerator`,
    `GenerateDocument`, `NotebookTemplate`
- Unit testing: `VerificationTest`
- Wolfram Data Drop: `Databin`, `DatabinAdd`
- URL/HTTP: `URLExecute`, `EmbedCode`, `HTTPRedirect`
- Time series superfunctions: `TimeSeriesModelFit`,
    `TimeSeriesAggregate` (the `TimeSeries` object itself is supported)
- Fractal sets: `MandelbrotSetPlot`, `JuliaSetPlot`
- Barcodes: `BarcodeImage`, `BarcodeRecognize`
- Financial conversions: `CurrencyConvert`, `InflationAdjust`
- `WikipediaSearch`
- Gradient image generators: `LinearGradientImage`,
    `RadialGradientImage`

## Version 9.0 (2012)

- Predictive interface and Suggestions Bar (notebook UI)
- `SocialMediaData`
- Random / stochastic processes: `MarkovProcess`, `QueueingProcess`,
    `ARProcess`
- Survival analysis
- Symbolic tensors (`TensorReduce`; `TensorExpand` is supported)
- Image recognition: `FindFaces`, `ImageFeatureTrack`
- Control systems design
- Gauges: `HorizontalGauge`, `VerticalGauge`, `ThermometerGauge`,
    `BulletGauge`
- Signal filters and analysis: `PIDTune`, `ButterworthFilterModel`,
    `EllipticFilterModel`, `KalmanFilter`, `PowerSpectralDensity`,
    `PeriodogramArray`
- Reliability analysis: `ReliabilityDistribution`, importance measures
    (`BirnbaumImportance`, `FussellVeselyImportance`, …)
- Volumetric images: `Image3D`, `Image3DSlices`
- Parametric differential equations: `ParametricNDSolve`,
    `ParametricNDSolveValue`
- Coordinate-system data: `CoordinateChartData`,
    `CoordinateTransformData`, `HodgeDual`
- Graph communities: `FindGraphCommunities`, `CommunityGraphPlot`
- Time-series processes: `ARIMAProcess`, `SARIMAProcess`,
    `FARIMAProcess`, `TimeSeriesForecast`, `RandomFunction`
- Queueing theory: `QueueingNetworkProcess`, `QueueProperties`
- More hypothesis tests: `LogRankTest`, `SpearmanRankTest`,
    `PearsonCorrelationTest`, `UnitRootTest`
- Unit utilities: `UnitSimplify`, `QuantityForm`
- Business calendars: `BusinessDayQ`, `HolidayCalendar`, `CalendarData`

## Version 8.0 (2010)

- Free-form linguistic input via Wolfram|Alpha
- GPU computing: `CUDAFunction` (CUDALink / OpenCLLink)
- C code generation: `CCodeGenerate`, `CompileToC`
- Financial engineering: `FinancialDerivative`, `FinancialData`
- Continuous `WaveletTransform`
- Control systems: `TransferFunctionModel`, `OutputResponse`,
    `StateResponse`, `NyquistPlot`, `NicholsPlot`, `RootLocusPlot`,
    `SingularValuePlot`, `RiccatiSolve`, `KalmanEstimator`,
    `StabilityMargins` (`StateSpaceModel` and `BodePlot` are supported)
- Distribution fitting and derived distributions:
    `EstimatedDistribution`, `DistributionFitTest`,
    `SmoothKernelDistribution`, `MarginalDistribution`,
    `CopulaDistribution`
- Hypothesis testing framework: `TTest`, `PairedTTest`,
    `KolmogorovSmirnovTest`, `MannWhitneyTest`, `SignTest`,
    `AndersonDarlingTest`, `ShapiroWilkTest`, `LeveneTest`
- Statistical visualization: `SmoothHistogram`, `DensityHistogram`,
    `DistributionChart`, `ProbabilityPlot`, `QuantilePlot`,
    `PairedHistogram`
- Graph analysis extras: `CayleyGraph`,
    `FindVertexCover`, `FindEdgeCover`, `HITSCentrality`,
    `BreadthFirstScan`, `DepthFirstScan`, and styling options
    (`VertexStyle`, `EdgeLabels`, `GraphLayout`)
- Permutation group theory: `GroupCentralizer`, `GroupStabilizerChain`,
    sporadic simple groups (`MonsterGroupM`, `ConwayGroupCo1`, …)
- Advanced image processing: `ImageKeypoints`, `ImageAlign`,
    `ImageCorrespondingPoints`, `Inpaint`, `ImageDeconvolve`, `Radon`,
    `InverseRadon`, many filters (`BilateralFilter`, `MeanShiftFilter`,
    `KuwaharaFilter`)
- Financial charts: `CandlestickChart`, `TradingChart`, `RenkoChart`,
    `KagiChart`, `PointFigureChart`, `FinancialIndicator`,
    `FinancialBond`
- Camera capture: `CurrentImage`, `ImageCapture`
- LibraryLink runtime: `LibraryFunction`, `LibraryLoad`,
    `LibraryFunctionUnload` (`LibraryFunctionLoad` is supported)
- Archives: `CreateArchive`, `ExtractArchive`

## Version 7.0 (2008)

- Built-in curated data (genomic, weather, astronomical, chemical)
- Delay differential equations in `NDSolve`
- Automatic charting superfunctions and computational typesetting
- Multi-kernel parallelism: `ParallelEvaluate`, `ParallelSum`,
    `ParallelCombine`, `DistributeDefinitions`, `WaitAll`,
    `CloseKernels` (`ParallelMap` / `ParallelTable` are supported but
    evaluate sequentially)
- Vector-field visualization variants: `ListVectorPlot`,
    `VectorDensityPlot`, `LineIntegralConvolutionPlot`,
    `ListStreamDensityPlot`
- Charting: `Histogram3D`, `RectangleChart`, `SectorChart3D`,
    `BubbleChart3D`
- Fourier series: `FourierSeries`, `FourierTrigSeries`,
    `FourierCosSeries`, `FourierSinSeries`, `FourierSequenceTransform`
- Morphological image processing: `HitMissTransform`, `TopHatTransform`,
    `BottomHatTransform`, `GeodesicDilation`, `GeodesicErosion`,
    `MorphologicalPerimeter`, filters (`LaplacianFilter`,
    `EntropyFilter`, `GeometricMeanFilter`)
- Generalized linear models: `GeneralizedLinearModelFit`,
    `ProbitModelFit`
- Holonomic sequences: `DifferenceRoot`, `DifferentialRoot`,
    `FindGeneratingFunction`, `Casoratian`
- q-analogs: `QHypergeometricPFQ`, `QPolyGamma`
- Geodesy: `GeodesyData`, `GeoProjectionData`, `GeoPositionXYZ`,
    `FindGeoLocation`
- Speech synthesis: `Speak`, `SpokenString`
- `SendMail`
- `SatisfiabilityInstances` (`SatisfiableQ` is supported)
- Curve reconstruction: `FindCurvePath`, `ListCurvePathPlot`

## Version 6.0 (2007)

- Live mouse-driven manipulation of 2D/3D graphics in notebooks
    (`Animate`, `Animator`, `ListAnimate`, `LocatorPane`, and `ClickPane` are
    supported as interactive widgets in the Playground and Studio, but
    dragging directly on a graphic — and 3D rotation — is not)
- Curated data collections: `CityData`, `ChemicalData`, `WordData`,
    `GraphData`, `KnotData`, `IsotopeData`, `LatticeData`,
    `ParticleData`, `DictionaryLookup`
- Dialogs, palettes, and windows: `CreateDialog`, `CreatePalette`,
    `CreateWindow`, `MessageDialog`, `ChoiceDialog`, `SystemDialogInput`,
    `Monitor`
- Sound output: `EmitSound`, `Beep`
- Spheroidal functions: `SpheroidalPS`, `SpheroidalS1`,
    `SpheroidalEigenvalue` (and primes), `SiegelTheta`
- Algebraic number fields: `AlgebraicNumber`, `ToNumberField`,
    `NumberFieldClassNumber`, `NumberFieldFundamentalUnits`
- 3D data visualization: `ListContourPlot3D`, `ListSurfacePlot3D`,
    `ReliefPlot`, `GraphPlot3D`
- `FindShortestTour`
- `BlockRandom`
- Cylindrical algebraic decomposition extras:
    `GenericCylindricalDecomposition`, `SemialgebraicComponentInstances`,
    `RootIntervals`

## Version 5.0–5.2 (2003–2005)

- Arbitrary-precision numerics engine and packed-array performance optimizations
- `CylindricalDecomposition`
- Factored linear solving: `LinearSolveFunction`
- `HessenbergDecomposition`

## Version 4.0–4.2 (1999–2002)

- Web/markup formats: MathML and XML `Import`/`Export`
- Add-on packages: `Combinatorica` (graph theory) and `ANOVA` (statistics)
- `AbsoluteOptions`
- The `Algebraics` domain for `Element` / `Simplify`
- Front-end box-option symbols (`ButtonBoxOptions`,
    `SuperscriptBoxOptions`, …) and notebook page options
    (`PageHeaders`, `PageFooters`)

## Version 3.0 (1996)

- Interactive 2D mathematical typesetting / notation input (front-end)
- Programmatic notebook control: `NotebookOpen`, `NotebookRead`,
    `NotebookWrite`, `NotebookFind`, `CellPrint`, `SelectionEvaluate`,
    `FrontEndExecute`
- In-language MathLink: `LinkCreate`, `LinkConnect`, `LinkLaunch`,
    `LinkRead`, `LinkWrite`, and the `*Packet` protocol symbols
- Mathieu even functions and characteristics: `MathieuC`,
    `MathieuCPrime`, `MathieuCharacteristicA`, `MathieuCharacteristicB`
    (`MathieuS` is supported)
- Weierstrass auxiliary functions: `WeierstrassSigma`, `WeierstrassZeta`
- `DumpSave` binary definition files
- `TrigFactorList`
- Hundreds of extensible operator symbols without built-in meanings
    (`LeftRightArrow`, `SquareSubset`, `Precedes`, `CupCap`, …)
- Notebook styling options: `CellMargins`, `WindowSize`,
    `StyleDefinitions`, `Magnification`, …

## Version 2.0–2.2 (1991–1993)

- `MathLink` protocol for external C / Fortran programs, including
    `Install` / `Uninstall` for external program functions
- Interactive debugging: `TraceDialog`, `TraceOn`, `TraceOff`,
    `StackBegin`, `StackInhibit`
- Dialog subsessions: `Dialog`, `DialogProlog`, `DialogSymbols`
- Sound primitives: `SampledSoundList`, `SampledSoundFunction`,
    `PlayRange`, `SampleDepth`
- Encoded packages: `Encode`, `DeclarePackage`
- The `Gradient` option for `FindMinimum`
- Various formatting and stream option symbols: `TableAlignments`,
    `TableSpacing`, `NumberFormat`, `RecordSeparators`,
    `WordSeparators`, …

## Version 1.0 (1988)

The original 554 built-in functions and the symbolic computation core are
fully supported.
The interactive notebook front-end is provided separately by Woxi Studio.
