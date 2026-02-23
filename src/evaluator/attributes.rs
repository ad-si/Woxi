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
    | "Beta"
    | "Zeta"
    | "PolyGamma"
    | "AiryAi"
    | "Hypergeometric1F1"
    | "Hypergeometric2F1"
    | "HypergeometricU"
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
    | "BitNot" => {
      vec!["Listable", "NumericFunction", "Protected"]
    }

    // Exp has ReadProtected too
    "Exp" => vec!["Listable", "NumericFunction", "Protected", "ReadProtected"],

    // Listable + Protected (non-numeric)
    "Range" | "IntegerDigits" | "RealDigits" | "Rationalize"
    | "IntegerString" | "ToCharacterCode" | "FromCharacterCode"
    | "StringLength" | "Characters" | "ToUpperCase" | "ToLowerCase"
    | "Boole" | "Positive" | "Negative" | "NonPositive" | "NonNegative"
    | "EvenQ" | "OddQ" | "PrimeQ" | "IntegerQ" | "NumberQ" | "NumericQ"
    | "AtomQ" | "Clip" | "Rescale" | "Unitize" | "UnitStep" | "N" => {
      vec!["Listable", "Protected"]
    }

    // HoldAllComplete + Protected
    "HoldComplete" | "Unevaluated" => {
      vec!["HoldAllComplete", "Protected"]
    }

    // HoldAll + Protected
    "Hold" | "HoldForm" | "Table" | "Do" | "While" | "For" | "Module"
    | "Block" | "With" | "Assuming" | "Trace" | "Defer" | "Compile"
    | "CompoundExpression" | "Switch" | "Which" | "Catch" | "Throw"
    | "Clear" | "ClearAll" | "Condition" | "Off" | "On" | "TimeConstrained"
    | "Information" => {
      vec!["HoldAll", "Protected"]
    }
    "Remove" => vec!["HoldAll", "Locked", "Protected"],

    // Function is HoldAll + Protected
    "Function" => vec!["HoldAll", "Protected"],

    // HoldFirst + Protected
    "MessageName" | "Increment" | "Decrement" | "PreIncrement"
    | "PreDecrement" | "Unset" => {
      vec!["HoldFirst", "Protected", "ReadProtected"]
    }
    "Message" | "AddTo" | "SubtractFrom" | "TimesBy" | "DivideBy"
    | "ClearAttributes" => {
      vec!["HoldFirst", "Protected"]
    }
    "Set" => vec!["HoldFirst", "Protected", "SequenceHold"],
    "SetDelayed" | "TagSetDelayed" | "UpSetDelayed" => {
      vec!["HoldAll", "Protected", "SequenceHold"]
    }
    "TagSet" | "UpSet" => vec!["HoldFirst", "Protected", "SequenceHold"],

    // HoldRest + Protected
    "If" | "PatternTest" | "Save" => vec!["HoldRest", "Protected"],
    "Rule" => vec!["Protected", "SequenceHold"],
    "RuleDelayed" => vec!["HoldRest", "Protected", "SequenceHold"],

    // And / Or: Flat + HoldAll + OneIdentity + Protected
    "And" | "Or" => vec!["Flat", "HoldAll", "OneIdentity", "Protected"],

    // Flat + OneIdentity + Protected
    "NonCommutativeMultiply" => vec!["Flat", "OneIdentity", "Protected"],

    // Constants
    "Pi" | "E" | "EulerGamma" | "GoldenRatio" | "Catalan" | "Degree"
    | "Khinchin" | "Glaisher" => {
      vec!["Constant", "Protected", "ReadProtected"]
    }
    "I" => vec!["Locked", "Protected", "ReadProtected"],
    "Infinity" | "PlotRange" | "MatrixForm" | "Show" | "Plot3D"
    | "ListPlot3D" | "Input" | "SeriesData" => {
      vec!["Protected", "ReadProtected"]
    }

    // NHoldRest
    "Subscript" => vec!["NHoldRest"],
    "Superscript" => vec!["NHoldRest", "ReadProtected"],
    "NumberForm" => vec!["NHoldRest", "Protected"],

    // NHoldAll + Protected
    "SlotSequence" => vec!["NHoldAll", "Protected"],

    // Listable + NHoldFirst + Protected
    "Out" => vec!["Listable", "NHoldFirst", "Protected"],

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
    | "Depth"
    | "Head"
    | "Nest"
    | "NestList"
    | "NestWhile"
    | "NestWhileList"
    | "Fold"
    | "FoldList"
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
    | "True"
    | "False"
    | "Null"
    | "None"
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
    | "MaxIterations"
    | "AccuracyGoal"
    | "General"
    | "Default"
    | "NonConstants"
    | "Number"
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
    | "Label"
    | "Continue"
    | "Goto"
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
    | "NSolve"
    | "Roots"
    | "Reduce"
    | "Eliminate"
    | "FindRoot"
    | "D"
    | "Integrate"
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
    | "Exponent"
    | "PolynomialQ"
    | "PolynomialRemainder"
    | "PolynomialQuotient"
    | "GCD"
    | "LCM"
    | "Mod"
    | "Quotient"
    | "QuotientRemainder"
    | "Divisors"
    | "FactorInteger"
    | "PrimePi"
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
    | "Definition"
    | "Attributes"
    | "Context"
    | "Contexts"
    | "Abort"
    | "Interrupt"
    | "Pause"
    | "Check"
    | "CheckAbort"
    | "Quiet"
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
    | "Fit" => {
      vec!["Protected"]
    }

    // Unknown symbol: empty attributes
    _ => vec![],
  }
}
