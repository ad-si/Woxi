# Dispatched functions still without scrut-tested docs

Generated from `src/evaluator/dispatch/`. Total: **287**.

These are functions that are reachable in Woxi's evaluator but don't yet have a `## \`Fn\``
section with a working scrut example in `tests/cli/`. Most fall into categories that resist
single-line CLI examples: stream/IO setup, graphics primitives that only render inside `Graphics`,
syntactic forms (`UpSet`, `Span`), and special functions that stay symbolic for typical inputs.

## Associations (3)

- `Dataset` — Wraps data with type information for structured data handling
- `Tabular` — Represents a data frame or table of structured data
- `ToTabular` — Converts data into a Tabular object with specified orientation

## Attributes (1)

- `Clear` — Clears values and definitions of symbols but preserves attributes

## Boolean (1)

- `Equivalent` — Logical equivalence

## Calculus / Symbolic (14)

- `AsymptoticSolve` — Find asymptotic solutions to equations near a point
- `DifferenceDelta` — Compute the forward difference of an expression
- `DifferenceQuotient` — Compute the forward difference quotient of an expression
- `DiscreteConvolve` — Discrete convolution of two sequences
- `DiscreteLimit` — Discrete limit of sequence
- `FourierCosTransform` — Fourier cosine transform of a function
- `FourierSinTransform` — Fourier sine transform of a function
- `FrenetSerretSystem` — Compute the Frenet-Serret system (curvatures and orthonormal frame) for a parametric curve
- `InverseFunction` — Represents the inverse of a function
- `NDSolve` — Solves ordinary differential equations numerically
- `NDSolveValue` — Numerically solve a differential equation and return the solution
- `RSolve` — Solve recurrence equations
- `RecurrenceTable` — Generate a table of values from a recurrence relation
- `Series` — Computes a power series expansion

## Date/Time (4)

- `DateInterval` — Creates an interval between two dates with normalized date list representation
- `SessionTime` — Elapsed time in seconds since session start
- `TimeUsed` — Total CPU time (user+system) used by the current process in seconds
- `UnixTime` — Returns the current Unix timestamp (seconds since 1970-01-01)

## Evaluation control (8)

- `BinomialDistribution` — Binomial probability distribution
- `ByteArray` — Creates a byte array from a list of unsigned byte values or a Base64-encoded string
- `HoldComplete` — Holds arguments completely unevaluated
- `Out` — Gives the expression on the nth output line in an interactive session
- `RegularExpression` — Represents a regular expression pattern for string matching
- `Stack` — Return the current evaluation call stack as a list of function names
- `TimeRemaining` — Returns the time remaining in a TimeConstrained evaluation
- `Trace` — Returns a minimal evaluation trace as a two-element list

## I/O & Streams (34)

- `AbsoluteFileName`
- `AnimationRate`
- `BinaryRead` — Reads a single binary object from a file or stream
- `BinaryWrite` — Write binary data
- `Close` — Closes an open stream and returns its name
- `DirectoryStack`
- `Environment` — Symbol representing the system environment
- `Export` — Exports data to a file
- `FileByteCount` — Count bytes in file (file I/O)
- `FileDate` — Get file modification date
- `FileFormat`
- `FileHash`
- `Find` — Finds a file on the search path
- `FindFile` — Find file
- `Get` — Reads and evaluates a file returning the last result
- `GetEnvironment` — Returns rule(s) mapping env variable names to their values
- `ImageResolution` — Option for image resolution in DPI
- `Message` — Issues a message (no-op in Woxi)
- `OpenRead` — Opens a file for reading
- `PNG`
- `ParentDirectory` — Parent directory path
- `PutAppend` — Append expressions to a file
- `Read` — Read an object of a specified type from a stream
- `ReadLine` — Read a single line from a stream or file
- `ResetDirectory` — resets the current directory to the previous one set by SetDirectory
- `Save` — Saves definitions associated with symbols to a file
- `SetDirectory` — Sets the current working directory
- `SetEnvironment` — Sets or unsets an environment variable (rule {name -> value} or list of rules; None unsets)
- `SetStreamPosition` — Sets the current position in an open stream
- `Skip` — Skip n values from a stream while reading
- `StreamPosition` — Returns the current position in an open stream
- `URLFetch`
- `Write` — Write expressions to an output stream in OutputForm
- `WriteString` — Write a string to an output stream

## Images (25)

- `Binarize` — Converts an image to black and white using a threshold
- `Blur` — Applies Gaussian blur to an image
- `ColorConvert` — Convert between color spaces
- `ColorData` — List of color data categories (no-arg form only)
- `ColorDistance` — Distance between colors
- `DominantColors` — Finds dominant colors in an image using k-means clustering
- `EdgeDetect` — Detects edges in an image with Gaussian smoothing and Sobel operator
- `ImageAdd` — Adds pixel values of two images pointwise
- `ImageAdjust` — Auto-adjusts image contrast and brightness
- `ImageApply` — Applies a function to each pixel of an image
- `ImageAssemble` — Assembles a grid of images into a single image
- `ImageCollage` — Creates a collage from a list of images with optional weights and fitting
- `ImageColorSpace` — Returns the color space of an image
- `ImageCompose` — Overlays one image on another
- `ImageCrop` — Crops an image to a specified region
- `ImageMultiply` — Multiplies pixel values of two images pointwise
- `ImageReflect` — Reflects an image horizontally or vertically
- `ImageResize` — Resizes an image to specified dimensions
- `ImageRotate` — Rotates an image by a given angle
- `ImageSubtract` — Subtracts pixel values of two images pointwise
- `ImageTake` — Takes a rectangular region from an image
- `ImageTrim` — Trims an image to a specified coordinate region
- `RandomImage` — Generates a random image with given dimensions
- `Rasterize` — Converts expression to raster image
- `Sharpen` — Sharpens an image using unsharp mask

## Intervals (1)

- `IntervalIntersection` — Computes the intersection of intervals

## Linear algebra (15)

- `AntihermitianMatrixQ` — Test whether a matrix is anti-Hermitian
- `Cartesian`
- `Cylindrical`
- `DiceDissimilarity` — Dice dissimilarity between binary vectors
- `JaccardDissimilarity` — Jaccard dissimilarity between binary vectors
- `LinearModelFit` — Fits a linear model to data and returns a FittedModel object
- `LogitModelFit` — Logistic regression model fitting via IRLS
- `MatchingDissimilarity` — Matching dissimilarity between binary vectors
- `RogersTanimotoDissimilarity` — Rogers-Tanimoto dissimilarity between binary vectors
- `RussellRaoDissimilarity` — Russell-Rao dissimilarity between binary vectors
- `SingularValueDecomposition` — Computes the singular value decomposition of a matrix
- `SokalSneathDissimilarity` — Sokal-Sneath dissimilarity between binary vectors
- `Spherical`
- `TensorWedge` — Exterior (wedge) product of vectors/tensors
- `YuleDissimilarity` — Yule dissimilarity between binary vectors

## Lists (4)

- `CountBy` — Groups and counts elements by a function
- `List` — Represents a list of elements
- `Nearest` — Find the nearest elements in a list to a given value
- `SymmetricDifference` — Returns elements that appear in an odd number of the given lists

## Math (general) (42)

- `AiryAiZero` — n-th real zero of the Airy function Ai
- `AiryBiZero` — n-th real zero of the Airy function Bi
- `AngerJ` — Anger function AngerJ[nu z] which equals BesselJ for integer orders
- `ArcCsc` — Returns the arc cosecant
- `ArcCsch` — Returns the inverse hyperbolic cosecant
- `ArcSec` — Returns the arc secant
- `BesselYZero`
- `Expectation` — Compute expected value of a function of a random variable
- `Fourier` — Discrete Fourier transform of a list
- `HankelH2` — Hankel function of the second kind
- `Hypergeometric0F1` — Confluent hypergeometric limit function
- `Hypergeometric0F1Regularized` — Regularized confluent hypergeometric limit function
- `Hypergeometric1F1` — Kummer confluent hypergeometric function 1F1(a; b; z)
- `Hypergeometric2F1` — Gauss hypergeometric function 2F1
- `Hypergeometric2F1Regularized` — Regularized hypergeometric 2F1 function
- `InverseFourier` — Inverse discrete Fourier transform of a list
- `InverseWeierstrassP` — Inverse Weierstrass elliptic function
- `KelvinBei` — Kelvin bei function (imaginary part of BesselJ on a rotated argument)
- `KelvinBer` — Kelvin ber function (real part of BesselJ on a rotated argument)
- `KelvinKei` — Kelvin kei function (imaginary part of e^(-Pi I/2) BesselK on a rotated argument)
- `KelvinKer` — Kelvin ker function (real part of e^(-Pi I/2) BesselK on a rotated argument)
- `ListFourierSequenceTransform` — Discrete-time Fourier transform of a list sequence
- `LocationTest` — perform a one- or two-sample t-test for location
- `NormalizedSquaredEuclideanDistance` — Normalized squared Euclidean distance between vectors
- `NumberDigit` — Returns the digit at a given position of a real number
- `PearsonChiSquareTest` — perform a Pearson chi-square goodness-of-fit test
- `PolynomialLCM` — Least common multiple of polynomials
- `Probability` — Compute probability of an event given a distribution
- `RandomPrime` — Random prime number
- `SeedRandom` — Seeds the random number generator
- `SphericalBesselJ` — Spherical Bessel function of the first kind
- `SphericalBesselY` — Spherical Bessel function of second kind
- `SphericalHankelH1` — Spherical Hankel function of the first kind
- `SphericalHankelH2` — Spherical Hankel function of the second kind
- `StirlingS1` — Returns the Stirling number of the first kind
- `StirlingS2` — Returns the Stirling number of the second kind
- `ThreeJSymbol` — Wigner 3-j symbol
- `TrimmedVariance` — Variance after trimming extreme values
- `WeberE` — Weber function E_nu(z) via numerical quadrature
- `WignerD` — Wigner D-matrix elements for rotation group representations
- `WinsorizedMean` — Mean after winsorizing extreme values
- `WinsorizedVariance` — Variance after winsorizing extreme values

## Math (special functions) (24)

- `Default` — Default value for optional arguments
- `DiracDelta` — Dirac delta generalized function
- `DisplayForm` — Wrapper that causes box expressions to be rendered visually
- `Format` — Display wrapper that formats an expression in a given form
- `FractionBox` — Low-level box construct for fraction display
- `FrameBox` — Low-level box construct for framed content display
- `FullDefinition` — Shows the definition of a symbol and all its dependencies
- `GridBox` — Low-level box construct for grid/table display
- `HeavisideTheta` — Heaviside step function
- `In` — Returns a previous input expression
- `InterpretationBox` — Low-level box that displays one expression but interprets as another
- `MakeBoxes` — Construct box form of an expression
- `OverscriptBox` — Low-level box construct for overscript display
- `RadicalBox` — Low-level box construct for nth root display
- `RawBoxes` — Wrapper for raw box expressions in typesetting
- `SqrtBox` — Low-level box construct for square root display
- `StyleBox` — Low-level box construct for styled content display
- `SubscriptBox` — Represents a subscript box for typesetting
- `SubsuperscriptBox` — Represents a box with both subscript and superscript for typesetting
- `SuperscriptBox` — Low-level box construct for superscript display
- `TagBox` — Low-level box construct that associates a tag with box content
- `TimeConstrained` — Runs computation with time limit (no-op in Woxi)
- `UnderoverscriptBox` — typesetting box for under-overscripts
- `UnderscriptBox` — a low-level typesetting box representing an expression with an underscript

## Misc (31)

- `BMP`
- `Character` — Symbol used in string patterns to represent a single character
- `EntityClassList` — List entity classes for a type
- `EntityProperties` — Lists properties available for an entity type
- `EntityRegister` — Registers an entity store for use with Entity
- `EntityStore` — Represents an entity store for custom entity types
- `EntityUnregister` — Unregisters entities from a registered entity store
- `Expression` — Type specifier for reading expressions
- `FrameRate`
- `GIF`
- `HalfSpace`
- `Heads` — Option for including heads in operations
- `Hue` — Specifies a color by hue saturation and brightness
- `InfinitePlane` — Infinite plane geometric region
- `JPEG`
- `JPG`
- `ListInterpolation` — Creates interpolation from uniformly spaced data
- `NotEqual`
- `Number` — Head type for numbers
- `Pphi`
- `Rr`
- `TIF`
- `TIFF`
- `True` — Boolean constant representing logical truth
- `Ttheta`
- `Undefined` — Undefined mathematical result
- `UpSet` — Assign upvalues to symbols appearing in the LHS
- `UpSetDelayed` — Assign delayed upvalues to symbols appearing in the LHS
- `VertexEccentricity` — Maximum distance from a vertex
- `Word` — Symbol representing a word in string patterns
- `Zz`

## Plotting (37)

- `ArrayPlot` — Plots a matrix as a grid of colored cells
- `BarChart3D` — 3D bar chart
- `BoxWhiskerChart` — Creates a box-and-whisker chart from data
- `ComplexPlot` — Domain coloring visualization of a complex function
- `DateListPlot` — Plots data with date-valued x-axis
- `DiscretePlot` — Plots discrete values of a function
- `DiscretePlot3D` — 3D surface plot at discrete integer grid points
- `GraphicsColumn` — Arranges graphics in a vertical column
- `GraphicsGrid` — Arranges graphics in a 2D grid
- `GraphicsRow` — Arranges graphics in a horizontal row
- `KochCurve` — Generates a Koch fractal curve as a Line primitive
- `ListContourPlot` — Creates a contour plot from a list of values or {x y z} triples
- `ListDensityPlot` — Creates a density plot from a list of values or {x y z} triples
- `ListLogLinearPlot` — Plots data with logarithmic x-axis
- `ListLogLogPlot` — Plots data with logarithmic x and y axes
- `ListLogPlot` — Plots data with logarithmic y-axis
- `ListPlot3D` — Creates a 3D plot from data
- `ListPolarPlot` — Plots data in polar coordinates
- `ListStepPlot` — Plots data as a step function
- `LogLinearPlot` — Plots with logarithmic x-axis
- `LogLogPlot` — Plots with logarithmic x and y axes
- `LogPlot` — Plots with logarithmic y-axis
- `MatrixPlot` — Plots a matrix as a colored grid
- `NumberLinePlot` — Plots values or intervals on a number line
- `ParametricPlot3D` — 3D parametric plot
- `PieChart3D` — 3D pie chart
- `RegionPlot3D` — plots the 3D region where a condition is true
- `RevolutionPlot3D` — creates a surface of revolution by rotating a curve around the z-axis
- `SectorChart` — Creates a sector chart from data
- `SphericalPlot3D` — 3D plot in spherical coordinates
- `StreamDensityPlot` — Creates a stream density plot of a vector field
- `StreamPlot` — Creates a streamline plot of a vector field
- `TreeForm` — Display an expression as a tree structure
- `TreeGraph` — Tree graph from directed edges with tree layout visualization
- `VectorPlot` — Creates a vector field plot
- `VectorPlot3D` — Creates a 3D vector field plot
- `WordCloud` — Creates a word cloud from a list of words

## Polynomials (9)

- `CoefficientArrays` — Returns sparse arrays of polynomial coefficients indexed by total degree
- `Eliminate` — Eliminates variables from a system of equations
- `ExpandAll` — Expands all products and powers in an expression
- `FindMaximum` — Finds a local maximum of a function numerically
- `FindMinimum` — Finds a local minimum of a function numerically
- `FromCoefficientRules` — Builds a polynomial from coefficient rules
- `HornerForm` — Converts a polynomial to Horner (nested) form
- `NMaximize` — Numerical global maximization with constraints using sampling and gradient refinement
- `NMinimize` — Numerical global minimization with constraints using sampling and gradient refinement

## Predicates (30)

- `Alternatives` — Pattern object representing any of several alternatives
- `Backslash`
- `CenterDot` — Symbolic centered dot operator
- `CircleTimes` — Symbolic tensor product operator with Unicode display
- `Contexts` — Lists known contexts; optional string pattern with * wildcards filters the result
- `DefaultValues` — Returns the default values of a symbol
- `Diamond`
- `DownValues` — Returns the downvalues of a symbol
- `FormatValues`
- `Function` — Represents a pure function
- `Inequality`
- `MaxMemoryUsed` — Peak memory usage in the current session
- `MemoryAvailable` — Returns the estimated free system memory in bytes
- `MemoryInUse` — Current memory usage of the process
- `Messages` — Returns the messages associated with a symbol
- `NValues`
- `Optional` — Represents a pattern with an optional default value
- `Options` — Returns or sets the options associated with a symbol
- `OwnValues` — Returns the own values of a symbol
- `Pattern` — Names a pattern object for use in transformation rules
- `PlusMinus` — Symbolic plus-or-minus operator with Unicode display
- `Postfix` — Symbol for postfix display formatting
- `Precedence` — Returns the parsing precedence of a symbol or expression
- `Prefix` — Prefix output form symbol
- `Put` — Write expressions to a file
- `SmallCircle` — Composition operator (formatting only)
- `Span` — Extract a span of elements using ;; notation in Part
- `StringExpression` — Represents a string pattern formed by concatenating string pattern objects
- `SubValues` — Returns the sub values of a symbol
- `Wedge` — Symbolic wedge product operator with Unicode display

## Quantities (2)

- `KnownUnitQ` — Tests whether an expression is a recognised unit specification
- `UnitConvert` — Converts a quantity to a different unit

## Strings (1)

- `DamerauLevenshteinDistance` — Levenshtein distance allowing one swap of adjacent chars as unit cost

## Structural / Forms (1)

- `Module` — Scopes local variables with unique names
