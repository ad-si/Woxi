#[allow(unused_imports)]
use super::*;

/// Returns the built-in attributes for a given symbol name.
/// Attributes are returned in alphabetical order, matching wolframscript output.
pub fn get_builtin_attributes(name: &str) -> Vec<&'static str> {
  match name {
    // Arithmetic operators
    "Plus" | "Times" => vec![
      "Flat",
      "Listable",
      "NumericFunction",
      "OneIdentity",
      "Orderless",
      "Protected",
    ],
    "GCD" | "LCM" => {
      vec!["Flat", "Listable", "OneIdentity", "Orderless", "Protected"]
    }
    "Composition" => vec!["Flat", "OneIdentity", "Protected"],
    "Power" => vec!["Listable", "NumericFunction", "OneIdentity", "Protected"],
    "Max" | "Min" => vec![
      "Flat",
      "NumericFunction",
      "OneIdentity",
      "Orderless",
      "Protected",
    ],

    // Trigonometric and math functions (Listable + NumericFunction + Protected)
    "Sin"
    | "Cos"
    | "Tan"
    | "Cot"
    | "Sec"
    | "Csc"
    | "ArcSin"
    | "ArcCos"
    | "ArcTan"
    | "ArcCot"
    | "ArcSec"
    | "ArcCsc"
    | "Sinh"
    | "Cosh"
    | "Tanh"
    | "Coth"
    | "Sech"
    | "Csch"
    | "ArcSinh"
    | "ArcCosh"
    | "ArcTanh"
    | "ArcCoth"
    | "ArcSech"
    | "ArcCsch"
    | "Log"
    | "Sqrt"
    | "Abs"
    | "AbsArg"
    | "Sign"
    | "Floor"
    | "Ceiling"
    | "Round"
    | "IntegerPart"
    | "FractionalPart"
    | "Gamma"
    | "Factorial"
    | "Factorial2"
    | "Subfactorial"
    | "Pochhammer"
    | "Erf"
    | "Erfc"
    | "Erfi"
    | "DawsonF"
    | "InverseErf"
    | "Beta"
    | "Zeta"
    | "PolyGamma"
    | "Hypergeometric0F1"
    | "Hypergeometric0F1Regularized"
    | "Hypergeometric1F1"
    | "Hypergeometric2F1"
    | "HypergeometricU"
    | "MittagLefflerE"
    | "WhittakerM"
    | "WhittakerW"
    | "BesselJ"
    | "BesselY"
    | "BesselI"
    | "BesselK"
    | "EllipticK"
    | "EllipticE"
    | "EllipticF"
    | "LegendreP"
    | "LegendreQ"
    | "PolyLog"
    | "LerchPhi"
    | "ExpIntegralEi"
    | "ExpIntegralE"
    | "EllipticTheta"
    | "WeierstrassP"
    | "WeierstrassPPrime"
    | "JacobiDN"
    | "JacobiSN"
    | "JacobiCN"
    | "JacobiSC"
    | "JacobiDC"
    | "JacobiCD"
    | "JacobiSD"
    | "JacobiCS"
    | "JacobiDS"
    | "JacobiNS"
    | "JacobiND"
    | "JacobiNC"
    | "ChebyshevT"
    | "ChebyshevU"
    | "GegenbauerC"
    | "LaguerreL"
    | "LogIntegral"
    | "HermiteH"
    | "Conjugate"
    | "Re"
    | "Im"
    | "ReIm"
    | "Arg"
    | "Gudermannian"
    | "InverseGudermannian"
    | "Sinc"
    | "Haversine"
    | "InverseHaversine"
    | "FresnelC"
    | "FresnelS"
    | "ProductLog"
    | "DigitCount"
    | "BitLength"
    | "BitAnd"
    | "BitOr"
    | "BitXor"
    | "BitNot"
    | "BitShiftRight"
    | "BitShiftLeft"
    | "SinhIntegral"
    | "CoshIntegral"
    | "BetaRegularized"
    | "GammaRegularized"
    | "Hypergeometric1F1Regularized"
    | "RealSign"
    | "RealAbs" => {
      vec!["Listable", "NumericFunction", "Protected"]
    }

    // Listable + Protected (no NumericFunction)
    "Discriminant" => vec!["Listable", "Protected"],

    // Listable + Protected + ReadProtected (no NumericFunction)
    "Divisible" => vec!["Listable", "Protected", "ReadProtected"],

    // These have ReadProtected too
    "Exp"
    | "AiryAi"
    | "AiryBi"
    | "InverseJacobiCN"
    | "InverseJacobiSN"
    | "InverseJacobiDN"
    | "InverseJacobiCD"
    | "InverseJacobiSC"
    | "InverseJacobiCS"
    | "InverseJacobiSD"
    | "InverseJacobiDS"
    | "InverseJacobiNS"
    | "InverseJacobiNC"
    | "InverseJacobiND"
    | "InverseJacobiDC"
    | "StruveH"
    | "StruveL"
    | "ParabolicCylinderD"
    | "AngerJ"
    | "WeberE"
    | "WignerD"
    | "InverseWeierstrassP" => {
      vec!["Listable", "NumericFunction", "Protected", "ReadProtected"]
    }
    "ArithmeticGeometricMean" => vec![
      "Listable",
      "NumericFunction",
      "Orderless",
      "Protected",
      "ReadProtected",
    ],
    "Multinomial" => {
      vec!["Listable", "NumericFunction", "Orderless", "Protected"]
    }

    // Listable + Protected (non-numeric)
    "Range" | "IntegerDigits" | "RealDigits"
    | "IntegerString" | "ToCharacterCode" | "FromCharacterCode"
    | "StringLength" | "Characters" | "ToUpperCase" | "ToLowerCase"
    | "Boole" | "Positive" | "Negative" | "NonPositive" | "NonNegative"
    | "EvenQ" | "OddQ" | "PrimeQ" | "IntegerQ" | "NumberQ" | "NumericQ"
    | "AtomQ" | "Clip" | "Cyclotomic" | "PartitionsP" | "PartitionsQ"
    | "Rescale"
    | "Resultant" | "Unitize" | "UnitStep" | "N" | "FactorSquareFree"
    | "PrimePi" | "BitGet" | "BitSet" | "BitClear" | "PowerMod"
    | "JacobiSymbol" | "IntegerExponent" => {
      vec!["Listable", "Protected"]
    }

    // HoldAllComplete + Protected
    "HoldComplete" | "Unevaluated" => {
      vec!["HoldAllComplete", "Protected"]
    }
    // MakeBoxes: HoldAllComplete only (matches wolframscript)
    "MakeBoxes" => vec!["HoldAllComplete"],

    // HoldAll + Protected
    "Hold" | "HoldForm" | "HoldPattern" | "Table" | "Do" | "While" | "For"
    | "Module" | "Block" | "With" | "Assuming" | "Trace" | "TraceScan"
    | "Defer" | "Compile" | "CompoundExpression" | "Switch" | "Which"
    | "Catch" | "Throw" | "Clear" | "ClearAll" | "Condition" | "Off" | "On"
    | "TimeConstrained" | "MemoryConstrained" | "TagUnset" | "NProduct"
    | "Definition" | "FullDefinition" | "Attributes" | "Quiet" | "Assert"
    | "OwnValues" | "DownValues" | "SubValues" | "UpValues"
    | "DefaultValues" | "FormatValues" | "NValues" | "Messages"
    // FindRoot holds its iterator `{var, x0}` so the variable name doesn't
    // get substituted by an OwnValue before the iteration starts.
    | "FindRoot" => {
      vec!["HoldAll", "Protected"]
    }
    // Manipulate: Protected + ReadProtected (matches wolframscript).
    // Wolfram does NOT expose HoldAll on Manipulate even though it
    // holds its body and variable specs in practice — the hold
    // behavior is implemented by the kernel internally (and in Woxi
    // by the explicit name-match in core_eval.rs), not via the
    // attribute. Adding HoldAll here would diverge from `Attributes[
    // Manipulate]` in wolframscript without changing semantics.
    "Manipulate" => vec!["Protected", "ReadProtected"],
    // Control: Protected (matches wolframscript). Like Manipulate it holds
    // its argument via the explicit name-match in core_eval.rs rather than a
    // HoldAll attribute.
    "Control" => vec!["Protected"],
    // PfaffianDet: Protected + ReadProtected (matches wolframscript).
    "PfaffianDet" => vec!["Protected", "ReadProtected"],
    // Parallel* combinators: Protected + ReadProtected. This matches a COLD
    // wolframscript kernel — these functions autoload lazily, so a fresh query
    // returns the {Protected, ReadProtected} stub. Once the Parallel subsystem
    // initializes (e.g. after any ParallelDo runs) wolframscript swaps in the
    // real {HoldAll, Protected} definition. There is no single stable reference;
    // do not "fix" this to {HoldAll, Protected} (it has been flip-flopped
    // twice). Like Manipulate, they hold their body via the explicit name-match
    // in core_eval.rs rather than a HoldAll attribute.
    "ParallelDo"
    | "ParallelTable" | "ParallelSum" | "ParallelProduct"
    | "ParallelMap" | "ParallelArray" | "ParallelCombine"
    | "ParallelSubmit" => vec!["Protected", "ReadProtected"],
    "Remove" => vec!["HoldAll", "Locked", "Protected"],
    "True" | "False" => vec!["Locked", "Protected"],

    // Function is HoldAll + Protected
    "Function" => vec!["HoldAll", "Protected"],

    // HoldFirst + Protected
    "MessageName" | "Increment" | "Decrement" | "PreIncrement"
    | "PreDecrement" | "Unset" => {
      vec!["HoldFirst", "Protected", "ReadProtected"]
    }
    // Dynamic holds its displayed expression (Attributes[Dynamic] =
    // {HoldFirst, Protected, ReadProtected}). Without this, `Dynamic[
    // data[[i, j]]]` collapses to the cell's value and loses the reference
    // an interactive control (e.g. a Checkbox) needs to write back to.
    "Dynamic" => vec!["HoldFirst", "Protected", "ReadProtected"],
    "Message" | "AddTo" | "SubtractFrom" | "TimesBy" | "DivideBy"
    | "ClearAttributes" | "AssociateTo" | "KeyDropFrom" | "Inactivate" => {
      vec!["HoldFirst", "Protected"]
    }
    "ApplyTo" => vec!["HoldFirst", "Protected"],
    "Set" => vec!["HoldFirst", "Protected", "SequenceHold"],
    "SetDelayed" | "TagSetDelayed" | "UpSetDelayed" => {
      vec!["HoldAll", "Protected", "SequenceHold"]
    }
    "TagSet" => vec!["HoldAll", "Protected", "SequenceHold"],
    "UpSet" => vec!["HoldFirst", "Protected", "SequenceHold"],

    // HoldRest + Protected
    "If" | "PatternTest" | "Save" => vec!["HoldRest", "Protected"],
    "Rule" => vec!["Protected", "SequenceHold"],
    "RuleDelayed" => vec!["HoldRest", "Protected", "SequenceHold"],

    // And / Or: Flat + HoldAll + OneIdentity + Protected
    "And" | "Or" => vec!["Flat", "HoldAll", "OneIdentity", "Protected"],

    // Flat + OneIdentity + Protected
    "NonCommutativeMultiply" => {
      vec!["Flat", "OneIdentity", "Protected"]
    }

    // Constants
    "Pi" | "E" | "EulerGamma" | "GoldenRatio" | "Catalan" | "Degree"
    | "Khinchin" | "Glaisher" => {
      vec!["Constant", "Protected", "ReadProtected"]
    }
    "I" => vec!["Locked", "Protected", "ReadProtected"],
    "Locked" => vec!["Locked", "Protected"],
    "EllipticExp"
    | "EllipticLog"
    | "Infinity"
    | "InputString"
    | "InverseSeries"
    | "PlotRange"
    | "MatrixForm"
    | "Show"
    | "ListPlot3D"
    | "Input"
    | "SeriesData"
    | "RunThrough"
    | "AbsolutePointSize"
    | "Entity"
    | "SquareWave"
    | "TriangleWave"
    | "SawtoothWave"
    | "GeneratingFunction"
    | "ExponentialGeneratingFunction"
    | "ScalingTransform"
    | "ReflectionTransform"
    | "ShearingTransform"
    | "AffineTransform"
    | "NetGraph"
    | "FunctionInterpolation"
    | "CMYKColor" => {
      vec!["Protected", "ReadProtected"]
    }
    "Plot3D" => {
      vec!["HoldAll", "Protected", "ReadProtected"]
    }

    // NHoldRest
    "Subscript" => vec!["NHoldRest"],
    "Superscript" => vec!["NHoldRest", "ReadProtected"],
    "EngineeringForm" | "NumberForm" | "AccountingForm" | "PercentForm" => {
      vec!["NHoldRest", "Protected"]
    }

    // NHoldAll + Protected
    "SlotSequence" => vec!["NHoldAll", "Protected"],

    // Listable + NHoldFirst + Protected
    "In" | "Out" => vec!["Listable", "NHoldFirst", "Protected"],

    // Protected only
    "Map"
    | "Apply"
    | "Select"
    | "Sort"
    | "SortBy"
    | "Reverse"
    | "Flatten"
    | "Join"
    | "Append"
    | "Prepend"
    | "Take"
    | "Drop"
    | "Part"
    | "First"
    | "Last"
    | "Rest"
    | "Most"
    | "Length"
    | "Level"
    | "Depth"
    | "Head"
    | "ReplaceAt"
    | "Nest"
    | "NestList"
    | "NestWhile"
    | "NestWhileList"
    | "Fold"
    | "FoldList"
    | "FoldWhile"
    | "FixedPoint"
    | "FixedPointList"
    | "MemberQ"
    | "FreeQ"
    | "Count"
    | "Position"
    | "Cases"
    | "DeleteCases"
    | "Replace"
    | "ReplaceAll"
    | "ReplaceRepeated"
    | "Thread"
    | "MapThread"
    | "MapIndexed"
    | "Scan"
    | "MatchQ"
    | "StringQ"
    | "ListQ"
    | "VectorQ"
    | "MatrixQ"
    | "FullForm"
    | "TreeForm"
    | "Dimensions"
    | "Total"
    | "Mean"
    | "Median"
    | "Variance"
    | "StandardDeviation"
    | "Not"
    | "Nand"
    | "Nor"
    | "Xor"
    | "Implies"
    | "Equivalent"
    | "Equal"
    | "Unequal"
    | "Less"
    | "Greater"
    | "LessEqual"
    | "GreaterEqual"
    | "SameQ"
    | "UnsameQ"
    | "Null"
    | "None"
    | "Undefined"
    | "Automatic"
    | "All"
    | "PlotStyle"
    | "AxesLabel"
    | "PlotLabel"
    | "Axes"
    | "AspectRatio"
    | "BlankNullSequence"
    | "BlankSequence"
    | "Integer"
    | "Optional"
    | "Mesh"
    | "String"
    | "Scaled"
    | "PlotPoints"
    | "Needs"
    | "Center"
    | "Rational"
    | "Left"
    | "Real"
    | "Ticks"
    | "Boxed"
    | "Repeated"
    | "RepeatedNull"
    | "ViewPoint"
    | "BoxRatios"
    | "DisplayFunction"
    | "Right"
    | "Top"
    | "Bottom"
    | "WorkingPrecision"
    | "HoldAll"
    | "Lighting"
    | "Listable"
    | "HoldFirst"
    | "End"
    | "Begin"
    | "BeginPackage"
    | "EndPackage"
    | "Modulus"
    | "Character"
    | "Complex"
    | "Constants"
    | "Break"
    | "Byte"
    | "MaxIterations"
    | "AccuracyGoal"
    | "General"
    | "Default"
    | "NonConstants"
    | "NRoots"
    | "Number"
    | "NumberSeparator"
    | "Underflow"
    | "Update"
    | "VerifySolutions"
    | "Short"
    | "Flat"
    | "OneIdentity"
    | "Overflow"
    | "ReadProtected"
    | "Protected"
    | "HoldRest"
    | "SetOptions"
    | "Above"
    | "Below"
    | "Continue"
    | "Format"
    | "FormatType"
    | "Orderless"
    | "ScientificForm"
    | "Print"
    | "Echo"
    | "ToString"
    | "ToExpression"
    | "List"
    | "Association"
    | "SubsetQ"
    | "Complement"
    | "Intersection"
    | "Union"
    | "StringJoin"
    | "StringSplit"
    | "StringTake"
    | "StringDrop"
    | "StringPosition"
    | "StringReplace"
    | "StringCases"
    | "StringMatchQ"
    | "StringFreeQ"
    | "StringCount"
    | "Solve"
    | "SolveValues"
    | "NSolve"
    | "NSolveValues"
    | "Roots"
    | "Reduce"
    | "Eliminate"
    | "D"
    | "NIntegrate"
    | "Sum"
    | "Product"
    | "Expand"
    | "ExpandAll"
    | "Factor"
    | "Simplify"
    | "FullSimplify"
    | "Together"
    | "Apart"
    | "Cancel"
    | "Collect"
    | "Coefficient"
    | "CoefficientList"
    | "CoefficientRules"
    | "Exponent"
    | "PolynomialQ"
    | "PolynomialRemainder"
    | "PolynomialQuotient"
    | "Mod"
    | "Environment"
    | "ExpandDenominator"
    | "FortranForm"
    | "ExpandNumerator"
    | "Infix"
    | "NumberPoint"
    | "Postfix"
    | "Prefix"
    | "Quartics"
    | "Quotient"
    | "QuotientRemainder"
    | "Divisors"
    | "FactorInteger"
    | "Prime"
    | "NextPrime"
    | "RandomInteger"
    | "RandomReal"
    | "RandomChoice"
    | "RandomSample"
    | "SeedRandom"
    | "Dot"
    | "Cross"
    | "Projection"
    | "ConjugateTranspose"
    | "BoxMatrix"
    | "Transpose"
    | "Inverse"
    | "Det"
    | "Tr"
    | "LinearSolve"
    | "Eigenvalues"
    | "Eigenvectors"
    | "RowReduce"
    | "MatrixRank"
    | "NullSpace"
    | "IdentityMatrix"
    | "DiagonalMatrix"
    | "ConstantArray"
    | "Precision"
    | "Accuracy"
    | "MachinePrecision"
    | "Context"
    | "Contexts"
    | "Abort"
    | "Interrupt"
    | "Pause"
    | "Check"
    | "CheckAbort"
    | "FilterRules"
    | "Operate"
    | "ReverseSort"
    | "Quartiles"
    | "ContainsOnly"
    | "LengthWhile"
    | "TakeLargestBy"
    | "TakeSmallestBy"
    | "Pick"
    | "PowerExpand"
    | "Variables"
    | "PauliMatrix"
    | "Curl"
    | "PrimePowerQ"
    | "BellB"
    | "Fit"
    | "ZeroTest"
    | "Share"
    | "NameQ"
    | "FactorTermsList"
    | "FactorTerms"
    | "Delimiters"
    | "PrecedenceForm"
    | "TotalWidth"
    | "Word"
    | "Frame"
    | "Background"
    | "AxesStyle"
    | "ColorFunction"
    | "AxesOrigin"
    | "FrameStyle"
    | "GridLines"
    | "GridLinesStyle"
    | "LabelStyle"
    | "Smaller"
    | "Larger"
    | "Epilog"
    | "FrameTicks"
    | "Contours"
    | "MaxRecursion"
    | "ContourStyle"
    | "Direction"
    | "TableHeadings"
    | "MeshStyle"
    | "PrecisionGoal"
    | "Prolog"
    | "BoxStyle"
    | "ContourShading"
    | "MaxSteps"
    | "SphericalRegion"
    | "ComplexExpand"
    | "Residue"
    | "SetPrecision"
    | "FactorSquareFreeList"
    | "Goto"
    | "Label"
    | "Element"
    | "NotElement"
    | "Alternatives"
    | "ImageSize"
    | "FontSize"
    | "FontFamily"
    | "FaceGrids"
    | "FaceGridsStyle"
    | "LibraryFunctionLoad"
    | "BaseStyle"
    | "Rationalize" => {
      vec!["Protected"]
    }

    // NonThreadable + Protected
    "MatrixPower" | "MatrixExp" | "MatrixFunction" => {
      vec!["NonThreadable", "Protected"]
    }

    // NHoldAll + Protected + ReadProtected
    "InverseFunction" => vec!["NHoldAll", "Protected", "ReadProtected"],
    "PrintTemporary" => vec!["Protected", "ReadProtected"],

    // Protected + ReadProtected (additional)
    "Sound"
    | "Padding"
    | "Cells"
    | "PointLegend"
    | "Cuboid"
    | "Raster"
    | "InterpolatingFunction"
    | "BezierFunction"
    | "BSplineFunction"
    | "Information"
    | "Reals"
    | "Thick"
    | "Thin"
    | "Integrate" => {
      vec!["Protected", "ReadProtected"]
    }

    // Unknown symbol: empty attributes
    _ => vec![],
  }
}
