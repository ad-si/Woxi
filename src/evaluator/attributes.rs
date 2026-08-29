use super::*;
use std::collections::HashSet;
use std::sync::LazyLock;

// Wolfram attributes
#[derive(Clone, Default)]
pub struct Attributes(u32); // bitmask
#[allow(non_upper_case_globals)]
impl Attributes {
  pub const None: u32 = 0;

  pub const Constant: u32 = 1 << 0;
  pub const Flat: u32 = 1 << 1;
  pub const HoldAll: u32 = 1 << 2;
  pub const HoldAllComplete: u32 = 1 << 3;
  pub const HoldFirst: u32 = 1 << 4;
  pub const HoldRest: u32 = 1 << 5;
  pub const Listable: u32 = 1 << 6;
  pub const Locked: u32 = 1 << 7;
  pub const NHoldAll: u32 = 1 << 8;
  pub const NHoldFirst: u32 = 1 << 9;
  pub const NHoldRest: u32 = 1 << 10;
  pub const NonThreadable: u32 = 1 << 11;
  pub const NumericFunction: u32 = 1 << 12;
  pub const OneIdentity: u32 = 1 << 13;
  pub const Orderless: u32 = 1 << 14;
  pub const Protected: u32 = 1 << 15;
  pub const ReadProtected: u32 = 1 << 16;
  pub const SequenceHold: u32 = 1 << 17;
  pub const Stub: u32 = 1 << 18;
  pub const Temporary: u32 = 1 << 19;

  const Masks: [&'static str; 20] = [
    "Constant",
    "Flat",
    "HoldAll",
    "HoldAllComplete",
    "HoldFirst",
    "HoldRest",
    "Listable",
    "Locked",
    "NHoldAll",
    "NHoldFirst",
    "NHoldRest",
    "NonThreadable",
    "NumericFunction",
    "OneIdentity",
    "Orderless",
    "Protected",
    "ReadProtected",
    "SequenceHold",
    "Stub",
    "Temporary",
  ];

  pub fn mask(name: &str) -> u32 {
    match Self::Masks.binary_search(&name) {
      Ok(index) => 1 << index,
      Err(_) => Self::None,
    }
  }
  pub fn masks(names: &Vec<String>) -> u32 {
    let mut mask = 0u32;
    for val in names {
      mask |= Self::mask(val.as_str());
    }
    mask
  }
  pub fn new(bits: u32) -> Self {
    Self(bits)
  }
  pub fn to_u32(&self) -> u32 {
    self.0
  }
  pub fn is_empty(&self) -> bool {
    self.0 == 0
  }
  pub fn contains(&self, mask: u32) -> bool {
    (self.0 & mask) != 0
  }
  pub fn add(&mut self, mask: u32) {
    self.0 |= mask;
  }
  pub fn remove(&mut self, mask: u32) {
    self.0 &= !mask;
  }
  pub fn to_vec(&self) -> Vec<&'static str> {
    let mut list: Vec<&'static str> = vec![];
    for i in 0..Self::Masks.len() {
      if self.contains(1 << i) {
        list.push(Self::Masks[i]);
      }
    }
    list
  }
}

/// `System`` symbols that Wolfram deliberately leaves *unprotected*, so that
/// user code can attach its own definitions to them (`CircleTimes[x_, y_] := …`)
/// or, for the `SystemModel*` family, because they live in a paclet context
/// that is not loaded by default. Verified against wolframscript by comparing
/// `Attributes[…]` for every name in `functions.csv`.
///
/// `get_builtin_attributes` consults this list before defaulting a known
/// built-in to `Protected`.
const UNPROTECTED_BUILTINS: &[&str] = &[
  "AllowVersionUpdate",
  "AngleBracket",
  "Application",
  "Assert",
  "AutoCopy",
  "Backslash",
  "Because",
  "BirnbaumSaundersDistribution",
  "Bra",
  "BraKet",
  "BracketingBar",
  "CachePersistence",
  "CalibratedSystemModel",
  "Cap",
  "CapitalDifferentialD",
  "CenterDot",
  "CircleDot",
  "CircleMinus",
  "CirclePlus",
  "CircleTimes",
  "ClearCookies",
  "CloudBase",
  "CloudObjectNameFormat",
  "CloudObjectURLType",
  "Colon",
  "Congruent",
  "ConnectSystemModelComponents",
  "ConnectSystemModelController",
  "CookieFunction",
  "Coproduct",
  "CovarianceEstimatorFunction",
  "CreateDataSystemModel",
  "CreateSystemModel",
  "Cup",
  "CupCap",
  "DataStructure",
  "Del",
  "DeliveryFunction",
  "Derivative",
  "Diamond",
  "DifferentialD",
  "DispersionEstimatorFunction",
  "DotEqual",
  "DoubleBracketingBar",
  "DoubleDownArrow",
  "DoubleLeftArrow",
  "DoubleLeftRightArrow",
  "DoubleLeftTee",
  "DoubleLongLeftArrow",
  "DoubleLongLeftRightArrow",
  "DoubleLongRightArrow",
  "DoubleRightArrow",
  "DoubleRightTee",
  "DoubleUpArrow",
  "DoubleUpDownArrow",
  "DoubleVerticalBar",
  "DownArrow",
  "DownArrowBar",
  "DownArrowUpArrow",
  "DownLeftRightVector",
  "DownLeftTeeVector",
  "DownLeftVector",
  "DownLeftVectorBar",
  "DownRightTeeVector",
  "DownRightVector",
  "DownRightVectorBar",
  "DownTee",
  "DownTeeArrow",
  "EndOfBuffer",
  "EpilogFunction",
  "EqualTilde",
  "Equilibrium",
  "EvaluationPrivileges",
  "ExcludedContexts",
  "ExponentialFamily",
  "ExponentialPowerDistribution",
  "FindCookies",
  "FindLibrary",
  "FindSystemModelEquilibrium",
  "FinishDynamic",
  "ForceVersionInstall",
  "GeneratedDocumentBinding",
  "GeneratedQuantityMagnitudes",
  "GeneratorDescription",
  "GeneratorHistoryLength",
  "GeneratorOutputType",
  "GreaterEqualLess",
  "GreaterFullEqual",
  "GreaterGreater",
  "GreaterLess",
  "GreaterSlantEqual",
  "GreaterTilde",
  "HalfNormalDistribution",
  "HumpDownHump",
  "HumpEqual",
  "IPAddress",
  "IconRules",
  "IncludeDefinitions",
  "IncludeGeneratorTasks",
  "IncludeQuantities",
  "IncludedContexts",
  "Install",
  "KeepExistingVersion",
  "Ket",
  "LeftArrow",
  "LeftArrowBar",
  "LeftArrowRightArrow",
  "LeftDownTeeVector",
  "LeftDownVector",
  "LeftDownVectorBar",
  "LeftRightArrow",
  "LeftRightVector",
  "LeftTee",
  "LeftTeeArrow",
  "LeftTeeVector",
  "LeftTriangle",
  "LeftTriangleBar",
  "LeftTriangleEqual",
  "LeftUpDownVector",
  "LeftUpTeeVector",
  "LeftUpVector",
  "LeftUpVectorBar",
  "LeftVector",
  "LeftVectorBar",
  "LessEqualGreater",
  "LessFullEqual",
  "LessGreater",
  "LessLess",
  "LessSlantEqual",
  "LessTilde",
  "LinearOffsetFunction",
  "LinkFunction",
  "LinkPatterns",
  "LongLeftArrow",
  "LongLeftRightArrow",
  "LongRightArrow",
  "LowerLeftArrow",
  "LowerRightArrow",
  "MakeBoxes",
  "MakeExpression",
  "MinusPlus",
  "Multicolumn",
  "MusicTempo",
  "NCache",
  "NestedGreaterGreater",
  "NestedLessLess",
  "NondimensionalizationTransform",
  "NotCongruent",
  "NotCupCap",
  "NotDoubleVerticalBar",
  "NotEqualTilde",
  "NotGreater",
  "NotGreaterEqual",
  "NotGreaterFullEqual",
  "NotGreaterGreater",
  "NotGreaterLess",
  "NotGreaterSlantEqual",
  "NotGreaterTilde",
  "NotHumpDownHump",
  "NotHumpEqual",
  "NotLeftTriangle",
  "NotLeftTriangleBar",
  "NotLeftTriangleEqual",
  "NotLess",
  "NotLessEqual",
  "NotLessFullEqual",
  "NotLessGreater",
  "NotLessLess",
  "NotLessSlantEqual",
  "NotLessTilde",
  "NotNestedGreaterGreater",
  "NotNestedLessLess",
  "NotPrecedes",
  "NotPrecedesEqual",
  "NotPrecedesSlantEqual",
  "NotPrecedesTilde",
  "NotReverseElement",
  "NotRightTriangle",
  "NotRightTriangleBar",
  "NotRightTriangleEqual",
  "NotSquareSubset",
  "NotSquareSubsetEqual",
  "NotSquareSuperset",
  "NotSquareSupersetEqual",
  "NotSubset",
  "NotSubsetEqual",
  "NotSucceeds",
  "NotSucceedsEqual",
  "NotSucceedsSlantEqual",
  "NotSucceedsTilde",
  "NotSuperset",
  "NotSupersetEqual",
  "NotTilde",
  "NotTildeEqual",
  "NotTildeFullEqual",
  "NotTildeTilde",
  "NotVerticalBar",
  "NotificationFunction",
  "OverBar",
  "OverDot",
  "OverHat",
  "OverTilde",
  "OverVector",
  "Overscript",
  "PacletSite",
  "Permissions",
  "PlusMinus",
  "Precedes",
  "PrecedesEqual",
  "PrecedesSlantEqual",
  "PrecedesTilde",
  "ProcessDirectory",
  "ProcessEnvironment",
  "Proportion",
  "Proportional",
  "RemoteInputFiles",
  "RemoteProviderSettings",
  "RestartInterval",
  "ReverseElement",
  "ReverseEquilibrium",
  "ReverseUpEquilibrium",
  "RightArrow",
  "RightArrowBar",
  "RightArrowLeftArrow",
  "RightDownTeeVector",
  "RightDownVector",
  "RightDownVectorBar",
  "RightTee",
  "RightTeeArrow",
  "RightTeeVector",
  "RightTriangle",
  "RightTriangleBar",
  "RightTriangleEqual",
  "RightUpDownVector",
  "RightUpTeeVector",
  "RightUpVector",
  "RightUpVectorBar",
  "RightVector",
  "RightVectorBar",
  "ScheduledTask",
  "SechDistribution",
  "SetCookies",
  "SetSystemModel",
  "SharingList",
  "ShortDownArrow",
  "ShortLeftArrow",
  "ShortRightArrow",
  "ShortUpArrow",
  "Skeleton",
  "SkewNormalDistribution",
  "SmallCircle",
  "SourceLink",
  "Square",
  "SquareIntersection",
  "SquareSubset",
  "SquareSubsetEqual",
  "SquareSuperset",
  "SquareSupersetEqual",
  "SquareUnion",
  "Star",
  "StringSkeleton",
  "SubMinus",
  "SubPlus",
  "SubStar",
  "Subscript",
  "Subset",
  "SubsetEqual",
  "Subsuperscript",
  "Succeeds",
  "SucceedsEqual",
  "SucceedsSlantEqual",
  "SucceedsTilde",
  "SuchThat",
  "SuperDagger",
  "SuperMinus",
  "SuperPlus",
  "SuperStar",
  "Superscript",
  "Superset",
  "SupersetEqual",
  "SystemModel",
  "SystemModelCalibrate",
  "SystemModelExamples",
  "SystemModelLinearize",
  "SystemModelMeasurements",
  "SystemModelParametricSimulate",
  "SystemModelPlot",
  "SystemModelReliability",
  "SystemModelSimulate",
  "SystemModelSimulateSensitivity",
  "SystemModelSimulationData",
  "SystemModelSurrogate",
  "SystemModelSurrogateTrain",
  "SystemModelUncertaintyPlot",
  "SystemModeler",
  "SystemModels",
  "Therefore",
  "Tilde",
  "TildeEqual",
  "TildeFullEqual",
  "TildeTilde",
  "URLResponseTime",
  "UnderBar",
  "Underoverscript",
  "Underscript",
  "Uninstall",
  "UnionPlus",
  "UpArrow",
  "UpArrowBar",
  "UpArrowDownArrow",
  "UpDownArrow",
  "UpEquilibrium",
  "UpTee",
  "UpTeeArrow",
  "UpdatePacletSites",
  "UpperLeftArrow",
  "UpperRightArrow",
  "VarianceEstimatorFunction",
  "Vee",
  "VerticalBar",
  "VerticalSeparator",
  "VerticalTilde",
  "VonMisesDistribution",
  "Wedge",
  "WignerSemicircleDistribution",
];

static UNPROTECTED_BUILTINS_HASH: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| UNPROTECTED_BUILTINS.iter().copied().collect());

/// True if Wolfram leaves `name` unprotected even though it is a built-in.
fn is_unprotected_builtin(name: &str) -> bool {
  UNPROTECTED_BUILTINS_HASH.contains(&name)
}

/// Returns the built-in attributes for a given symbol name.
/// Attributes are returned in alphabetical order, matching wolframscript output.
pub fn get_builtin_attributes(name: &str) -> Attributes {
  use Attributes as A;
  let mask = match name {
    // Arithmetic operators
    "Plus" | "Times" =>
      A::Flat |
      A::Listable |
      A::NumericFunction |
      A::OneIdentity |
      A::Orderless |
      A::Protected
    ,
    "GCD" | "LCM" => {
      A::Flat | A::Listable | A::OneIdentity | A::Orderless | A::Protected
    }
    "Composition" => A::Flat | A::OneIdentity | A::Protected,
    "Power" => A::Listable | A::NumericFunction | A::OneIdentity | A::Protected,
    "Max" | "Min" =>
      A::Flat |
      A::NumericFunction |
      A::OneIdentity |
      A::Orderless |
      A::Protected
    ,

    // Trigonometric and math functions (Listable + NumericFunction + Protected)
    "Sin"
    | "Cos"
    | "Tan"
    | "Cot"
    | "Sec"
    | "Csc"
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
    | "Log" | "Log10" | "Log2"
    | "Sqrt"
    | "Abs"
    | "Sign"
    | "Floor"
    | "Ceiling"
    | "Round"
    | "IntegerPart"
    | "FractionalPart"
    | "Gamma"
    | "LogGamma"
    | "Pochhammer"
    | "Factorial"
    | "Factorial2"
    | "Subfactorial"
    | "QFactorial"
    | "Erf"
    | "Erfc"
    | "Erfi"
    | "DawsonF"
    | "InverseErf" | "InverseErfc" // Listable delayed
    | "Beta"
    | "Zeta"
    | "HurwitzZeta"
    | "PolyGamma"
    | "Hypergeometric0F1"
    | "Hypergeometric0F1Regularized"
    | "Hypergeometric1F1"
    | "Hypergeometric2F1"
    | "HypergeometricU"
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
    | "BetaRegularized"
    | "GammaRegularized"
    | "Hypergeometric1F1Regularized"
    | "Mod" | "Quotient" | "Binomial" | "PascalBinomial"
    | "JacobiP" | "JacobiZeta"
    | "CarlsonRC" | "CarlsonRD" | "CarlsonRF" | "CarlsonRG" | "CarlsonRJ"
    | "EllipticPi"
    | "FactorialPower"
    | "LogisticSigmoid"
    | "RealSign"
    | "RealAbs" => {
      A::Listable | A::NumericFunction | A::Protected
    }

    // NumericFunction + Protected
    "Clip" => A::NumericFunction | A::Protected,

    // NumericFunction + Protected + ReadProtected
    "SinDegrees" | "CosDegrees" | "TanDegrees"
    | "SecDegrees" | "CscDegrees" | "CotDegrees"
    | "ArcSinDegrees" | "ArcCosDegrees" | "ArcTanDegrees"
    | "ArcCotDegrees" | "ArcSecDegrees" | "ArcCscDegrees"
    | "WeierstrassP" | "InverseWeierstrassP" | "WeierstrassPPrime"
    | "Rescale" => {
      A::NumericFunction | A::Protected | A::ReadProtected
    }

    // Listable + Orderless + Protected
    "CoprimeQ" => A::Listable | A::Orderless | A::Protected,

    // Listable + Protected + ReadProtected
    "Divisible"
    | "ThueMorse"
    | "RamanujanTau"
    | "NextPrime" // not Listable in wolframscript.
    // The rest of these have Listable delayed till first use in wolframscript.
    | "FiniteAbelianGroupCount" | "FiniteGroupCount"
    | "KroneckerSymbol"
    | "SquaresR"
    | "PrimePowerQ"
    | "MangoldtLambda"
    | "IntegerReverse"
    | "DigitSum"
    | "RudinShapiro" => A::Listable | A::Protected | A::ReadProtected,

    // Listable + NumericFunction + Protected + ReadProtected
    "Exp"
    | "AiryAi" | "AiryBi"
    | "ArcSin" | "ArcCos" | "ArcTan"
    | "ArcCot" | "ArcSec" | "ArcCsc"
    | "JacobiAmplitude" | "JacobiEpsilon"
    | "JacobiDN" | "JacobiSN" | "JacobiCN"
    | "JacobiSC" | "JacobiDC" | "JacobiCD"
    | "JacobiSD" | "JacobiCS" | "JacobiDS"
    | "JacobiNS" | "JacobiND" | "JacobiNC"
    | "InverseJacobiCN" | "InverseJacobiSN" | "InverseJacobiDN"
    | "InverseJacobiCD" | "InverseJacobiSC" | "InverseJacobiCS"
    | "InverseJacobiSD" | "InverseJacobiDS" | "InverseJacobiNS"
    | "InverseJacobiNC" | "InverseJacobiND" | "InverseJacobiDC"
    | "StruveH" | "StruveL"
    | "ParabolicCylinderD"
    | "AngerJ" | "WeberE"
    | "SphericalBesselJ" | "SphericalBesselY"
    | "SphericalHankelH1" | "SphericalHankelH2"
    | "SinIntegral" | "CosIntegral"
    | "SinhIntegral" | "CoshIntegral"
    | "HarmonicNumber" | "HyperHarmonicNumber" | "AlternatingHarmonicNumber"
    | "KelvinBei" | "KelvinBer" | "ModularLambda"
    | "KleinInvariantJ" | "MittagLefflerE"
    | "Fibonacci" | "LucasL" | "Hyperfactorial" | "BarnesG"
    | "EllipticNomeQ" | "InverseEllipticNomeQ" | "DedekindEta"
    | "Surd" | "CubeRoot"
    | "CatalanNumber" | "ZernikeR" => {
      A::Listable | A::NumericFunction | A::Protected | A::ReadProtected
    }
    "ArithmeticGeometricMean" =>
      A::Listable |
      A::NumericFunction |
      A::Orderless |
      A::Protected |
      A::ReadProtected
    ,
    "Multinomial" => {
      A::Listable | A::NumericFunction | A::Orderless | A::Protected
    }

    // Listable + Protected
    "Range" | "IntegerDigits" | "RealDigits"
    | "IntegerString"
    | "StringLength" | "Characters" | "ToUpperCase" | "ToLowerCase"
    | "Boole" | "Positive" | "Negative" | "NonPositive" | "NonNegative"
    | "EvenQ" | "OddQ" | "PrimeQ"
    | "Cyclotomic" | "PartitionsP" | "PartitionsQ"
    | "StirlingS1" | "StirlingS2" | "MixedFractionParts"
    | "AbsArg" | "Divisors" | "PrimitiveRoot"
    | "Numerator" | "Denominator" | "MoebiusMu"
    | "CompositeQ" | "RomanNumeral" | "FactorInteger"
    | "Resultant" | "Unitize" | "UnitStep" | "FactorSquareFree"
    | "PrimePi" | "BitGet" | "BitSet" | "BitClear" | "PowerMod"
    | "JacobiSymbol" | "IntegerExponent" | "CarmichaelLambda"
    | "IntegerLength" | "ContinuedFraction" | "MinimalPolynomial"
    | "Discriminant" | "BernoulliB" | "Prime" | "EulerPhi"
    | "ToExpression" => A::Listable | A::Protected,

    // HoldAllComplete + Protected
    "HoldComplete" | "HoldCompleteForm" | "Unevaluated"
    | "Association" => A::HoldAllComplete | A::Protected,

    // HoldAllComplete + Protected + ReadProtected
    "InterpretationBox" => {
      A::HoldAllComplete | A::Protected | A::ReadProtected
    }

    // HoldAll + Protected
    "Hold" | "HoldForm" | "HoldPattern" | "Table" | "Do" | "While" | "For"
    | "Module" | "DynamicModule" | "Block" | "With"
    | "Trace" | "TraceScan" | "TracePrint"
    | "Defer" | "Compile" | "CompiledFunction" | "Which"
    | "Clear" | "ClearAll" | "Condition" | "Off" | "On"
    | "TimeConstrained" | "MemoryConstrained" | "TagUnset" | "NProduct"
    | "Definition" | "FullDefinition" | "Quiet"
    | "OwnValues" | "DownValues" | "SubValues" | "UpValues"
    | "Protect" | "Unprotect"
    // NIntegrate has HoldAll in wolframscript, but this breaks tests so fix later.
    // | "NIntegrate"
    | "DefaultValues" | "FormatValues" | "NValues" | "Messages"
    // Function is HoldAll + Protected
    | "Function"
    // FindRoot holds its iterator `{var, x0}` so the variable name doesn't
    // get substituted by an OwnValue before the iteration starts.
    | "FindRoot" => A::HoldAll | A::Protected,

    // HoldAllComplete
    // Assert is the odd one out: wolframscript reports HoldAllComplete and
    // no Protected at all.
    "Assert"
    // MakeBoxes: HoldAllComplete only (matches wolframscript)
    | "MakeBoxes" => A::HoldAllComplete,

    // Manipulate: Protected + ReadProtected (matches wolframscript).
    // Wolfram does NOT expose HoldAll on Manipulate even though it
    // holds its body and variable specs in practice — the hold
    // behavior is implemented by the kernel internally (and in Woxi
    // by the explicit name-match in core_eval.rs), not via the
    // attribute. Adding HoldAll here would diverge from `Attributes[
    // Manipulate]` in wolframscript without changing semantics.
    "Manipulate" => A::Protected | A::ReadProtected,
    // Control: Protected (matches wolframscript). Like Manipulate it holds
    // its argument via the explicit name-match in core_eval.rs rather than a
    // HoldAll attribute.
    "Control" => A::Protected,
    // GeometricScene: Protected + ReadProtected (matches wolframscript). Like
    // Manipulate it holds its arguments via the explicit name-match in
    // core_eval.rs rather than a HoldAll attribute.
    "GeometricScene" => A::Protected | A::ReadProtected,
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
    | "ParallelSelect" | "ParallelCases"
    | "ParallelSubmit" => A::Protected | A::ReadProtected,

    // HoldAll + Locked + Protected
    "Remove" => A::HoldAll | A::Locked | A::Protected,

    // HoldFirst + Protected + ReadProtected
    "MessageName" | "Increment" | "Decrement" | "PreIncrement"
    | "PreDecrement" | "Unset"
    // Dynamic holds its displayed expression (Attributes[Dynamic] =
    // {HoldFirst, Protected, ReadProtected}). Without this, `Dynamic[
    // data[[i, j]]]` collapses to the cell's value and loses the reference
    // an interactive control (e.g. a Checkbox) needs to write back to.
    | "Dynamic"
    | "Enclose" => A::HoldFirst | A::Protected | A::ReadProtected,

    // HoldFirst + Protected
    "Message" | "AddTo" | "SubtractFrom" | "TimesBy" | "DivideBy"
    | "ClearAttributes" | "AssociateTo" | "KeyDropFrom" | "Inactivate"
    | "AppendTo" | "PrependTo"
    // Refresh has HoldFirst in wolframscript, but this breaks tests so fix later.
    // | "Refresh"
    // `Context` reports the context a *symbol* belongs to, so it must not
    // look at the symbol's value: `x = 1; Context[x]` is `Global``.
    | "Context"
    // `BlockRandom` only holds the body it localizes
    // the generator state around; its trailing options (`RandomSeeding -> …`)
    // are evaluated like any other option list.
    | "BlockRandom"
    | "Catch" | "Pattern" | "SetAttributes"
    | "ApplyTo" => A::HoldFirst | A::Protected,

    "Set" | "UpSet"
    | "RepeatedTiming" => A::HoldFirst | A::Protected | A::SequenceHold,

    "SetDelayed" | "TagSetDelayed" | "UpSetDelayed"
    | "AbsoluteTiming" | "Timing" 
    | "TagSet" => A::HoldAll | A::Protected | A::SequenceHold,

    // HoldRest + Protected
    "If" | "PatternTest" | "Save"
    // Switch evaluates its first argument and then
    // each pattern in turn; First and Last hold their default so it is only
    // evaluated when there is no element to return.
    | "Switch" | "First" | "Last"
    | "FirstPosition" | "SelectFirst" | "FirstCase"
    | "Assuming" => A::HoldRest | A::Protected,

    // `Button[label, action]` holds its action: the action is what happens
    // when the button is pressed, so merely building or displaying the
    // button must not run it.
    "Button" => A::HoldRest | A::Protected | A::ReadProtected,
    "Rule" => A::Protected | A::SequenceHold,
    "RuleDelayed" => A::HoldRest | A::Protected | A::SequenceHold,

    // And / Or: Flat + HoldAll + OneIdentity + Protected
    "And" | "Or" => A::Flat | A::HoldAll | A::OneIdentity | A::Protected,

    // Flat + OneIdentity + Protected
    "NonCommutativeMultiply" => {
      A::Flat | A::OneIdentity | A::Protected
    }

    // Constants
    "Pi" | "E" | "Degree" => {
      A::Constant | A::Protected | A::ReadProtected
    }
    "EulerGamma" | "GoldenRatio" | "Catalan" | "MachinePrecision"
    | "Khinchin" | "Glaisher" | "GoldenAngle" => {
      A::Constant | A::Protected
    }
    "ChampernowneNumber" => {
      A::Constant | A::Listable | A::NHoldFirst | A::NumericFunction | A::Protected | A::ReadProtected
    }

    "I" => A::Locked | A::Protected | A::ReadProtected,

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
      A::Protected | A::ReadProtected
    }

    // NHoldRest
    "Subscript" => A::NHoldRest,
    "Superscript" => A::NHoldRest | A::ReadProtected,
    "EngineeringForm" | "NumberForm" | "AccountingForm" | "PercentForm" => {
      A::NHoldRest | A::Protected
    }

    // Listable + NHoldAll + Protected
    "DivisorSigma" => A::Listable | A::NHoldAll | A::Protected,

    // NHoldAll + Protected
    "SlotSequence" => A::NHoldAll | A::Protected,

    // Listable + NHoldFirst + NumericFunction + Protected
    "EllipticTheta" => A::Listable | A::NHoldFirst | A::NumericFunction | A::Protected,

    // Listable + NHoldFirst + Protected
    "In" | "Out" => A::Listable | A::NHoldFirst | A::Protected,

    // Listable + NHoldAll + Protected + ReadProtected
    "DirichletL" => A::Listable | A::NHoldFirst | A::Protected | A::ReadProtected,

    // Listable + NHoldFirst + Protected + ReadProtected
    "BellB" | "StieltjesGamma" => {
      A::Listable | A::NHoldFirst | A::Protected | A::ReadProtected
    }

    // Locked + Protected (matches wolframscript: these symbols cannot be
    // unprotected).
    "List" | "Symbol"
    | "True" | "False"
    | "Locked" => A::Locked | A::Protected,

    // HoldAll + Protected + ReadProtected
    // Sum and Product hold their body and iterator so the iteration
    // variable is not substituted by an OwnValue before the sum starts.
    "Sum" | "Product"
    | "Piecewise" | "ValueQ"
    // ControlActive has HoldAll in wolframscript, fixing breaks tests.
    // | "ControlActive"
    // ForAll and Exists have HoldAll in wolframscript, fixing breaks tests.
    // | "ForAll" | "Exists"
    | "ContinuedFractionK" | "GraphPropertyDistribution"
    // Plot3D and Confirm* don't have HoldAll in wolframscript
    | "Plot3D"
    | "Confirm" | "ConfirmBy" | "ConfirmMatch" | "ConfirmAssert"
    | "ConfirmQuiet"
    | "CompoundExpression" => A::HoldAll | A::Protected | A::ReadProtected,

    // HoldAll + Listable + Protected
    "Attributes" => A::HoldAll | A::Listable | A::Protected,

    // Flat + OneIdentity + Protected
    "Join" | "StringJoin" => A::Flat | A::OneIdentity | A::Protected,
    "Union" | "Intersection" => {
      A::Flat | A::OneIdentity | A::Protected | A::ReadProtected
    }
    "Part" => A::NHoldRest | A::Protected | A::ReadProtected,
    "Slot" => A::NHoldAll | A::Protected,

    // Protected + ReadProtected
    "D" | "Limit" | "Mean" | "Median" | "Variance" | "Missing" => {
      A::Protected | A::ReadProtected
    }

    // NonThreadable + Protected
    "MatrixPower" | "MatrixExp" | "MatrixFunction" => {
      A::NonThreadable | A::Protected
    }

    // NHoldAll + Protected + ReadProtected
    // `C` is the default generated-parameter symbol of DSolve, RSolve,
    // Reduce, Solve, … (`C[1]`, `C[2]`, …), so it is a protected built-in
    // even though it is a bare single letter.
    "C" | "InverseFunction" => {
      A::NHoldAll | A::Protected | A::ReadProtected
    }
    "PrintTemporary" => A::Protected | A::ReadProtected,

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
    | "WignerD" | "PfaffianDet"
    | "Information"
    | "Reals"
    | "Thick"
    | "Thin"
    | "Integrate" => {
      A::Protected | A::ReadProtected
    }

    // Any other known built-in defaults to Protected, matching Wolfram, which
    // protects every `System`` symbol except those in `UNPROTECTED_BUILTINS`.
    // Unknown symbols have no attributes.
    _ => {
      if crate::evaluator::is_builtin_symbol(name)
        && !is_unprotected_builtin(name)
      {
        A::Protected
      } else {
        A::None
      }
    }
  };
  Attributes::new(mask)
}

/// Extract a symbol name from `Expr::Identifier(name)` or
/// `Expr::Constant(name)` (constants like Pi/E are parsed as `Expr::Constant`
/// so handlers that take "any symbol" must accept both).
fn symbol_name(e: &Expr) -> Option<String> {
  match e {
    Expr::Identifier(n) | Expr::Constant(n) => Some(n.clone()),
    _ => None,
  }
}

pub fn dispatch_attributes(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "SetAttributes" if args.len() == 2 => {
      let func_names: Vec<String> = match &args[0] {
        Expr::List(items) => items.iter().filter_map(symbol_name).collect(),
        _ => symbol_name(&args[0]).map(|n| vec![n]).unwrap_or_default(),
      };
      let Some(attr) = get_attributes(&args[1]) else {
        return Some(Ok(Expr::Identifier("Null".to_string())));
      };

      if !func_names.is_empty() {
        let mut locked = false;
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(Attributes::Locked)
            {
              crate::emit_message(&format!(
                "Attributes::locked: Symbol {func_name} is locked."
              ));
              locked = true;
              continue;
            }
            let mask = attr;
            attrs
              .entry(func_name.clone())
              .and_modify(|a| (*a).add(mask))
              .or_insert(Attributes::new(mask));
          }
        });
        // Re-adding a builtin attribute via SetAttributes prunes it from
        // the removed-tracking, so `Attributes[sym]` once again reports it.
        crate::FUNC_ATTRS_REMOVED.with(|m| {
          let mut removed = m.borrow_mut();
          let mask = attr;
          for func_name in &func_names {
            removed.entry(func_name.clone()).and_modify(|a| {
              (*a).remove(mask);
            });
          }
        });
        if locked {
          return Some(Ok(Expr::Identifier("Null".to_string())));
        }
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
    }
    "ClearAttributes" if args.len() == 2 => {
      let func_names: Vec<String> = match &args[0] {
        Expr::List(items) => items.iter().filter_map(symbol_name).collect(),
        _ => symbol_name(&args[0]).map(|n| vec![n]).unwrap_or_default(),
      };
      let to_remove: Vec<String> = match &args[1] {
        Expr::Identifier(a) => vec![a.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(a) = item {
              Some(a.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      if !func_names.is_empty() {
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          let mask = Attributes::masks(&to_remove);
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(Attributes::Locked)
            {
              crate::emit_message(&format!(
                "Attributes::locked: Symbol {func_name} is locked."
              ));
              continue;
            }

            attrs.entry(func_name.clone()).and_modify(|a| {
              (*a).remove(mask);
            });
          }
        });
        // Remove from builtin attributes via the removed-tracking, mirroring
        // how Unprotect handles the Protected attribute.
        crate::FUNC_ATTRS_REMOVED.with(|m| {
          let mut removed = m.borrow_mut();
          let mask = Attributes::masks(&to_remove);
          for func_name in &func_names {
            let builtin = get_builtin_attributes(func_name);
            let mask = mask & builtin.to_u32(); // ignore attributes not on builtin.
            if mask != 0 {
              removed
                .entry(func_name.clone())
                .and_modify(|a| (*a).add(mask))
                .or_insert(Attributes::new(mask));
            }
          }
        });
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
    }
    "Protect" => {
      let mut protected_syms = Vec::new();
      for arg in args {
        if let Some(sym) = symbol_name(arg) {
          let sym = &sym;
          // If Protected is a builtin attribute that was previously removed,
          // restore it by pruning FUNC_ATTRS_REMOVED. Otherwise add as a
          // user-set attribute.
          let builtin = get_builtin_attributes(sym);
          let was_builtin = builtin.contains(Attributes::Protected);
          if was_builtin {
            crate::FUNC_ATTRS_REMOVED.with(|m| {
              let mut removed = m.borrow_mut();
              removed.entry(sym.clone()).and_modify(|a| {
                (*a).remove(Attributes::Protected);
              });
            });
          } else {
            crate::FUNC_ATTRS.with(|m| {
              let mut attrs = m.borrow_mut();
              let mask = Attributes::Protected;
              attrs
                .entry(sym.clone())
                .and_modify(|a| (*a).add(mask))
                .or_insert(Attributes::new(mask));
            });
          }
          protected_syms.push(Expr::String(sym.clone()));
        }
      }
      return Some(Ok(Expr::List(protected_syms.into())));
    }
    "Unprotect" => {
      let mut unprotected_syms = Vec::new();
      for arg in args {
        if let Some(sym) = symbol_name(arg) {
          let sym = &sym;
          let is_locked = {
            let builtin = get_builtin_attributes(sym);
            if builtin.contains(Attributes::Locked) {
              true
            } else {
              crate::func_attrs_contains(sym.as_str(), Attributes::Locked)
            }
          };
          if is_locked {
            crate::emit_message(&format!(
              "Protect::locked: Symbol {sym} is locked."
            ));
            continue;
          }
          // A symbol counts as Protected if either its builtin default
          // attributes or its user-stored attributes contain "Protected".
          let was_user_protected = crate::FUNC_ATTRS.with(|m| {
            let mut attrs = m.borrow_mut();
            if let Some(entry) = attrs.get_mut(sym) {
              let before = entry.to_u32();
              let after = before & !Attributes::Protected;
              if before == after {
                false
              } else {
                attrs.insert(sym.clone(), Attributes::new(after));
                true
              }
            } else {
              false
            }
          });
          let builtin = get_builtin_attributes(sym);
          let was_builtin_protected = builtin.contains(Attributes::Protected);
          if was_builtin_protected {
            crate::FUNC_ATTRS_REMOVED.with(|m| {
              let mut removed = m.borrow_mut();
              let mask = Attributes::Protected;
              removed
                .entry(sym.clone())
                .and_modify(|a| (*a).add(mask))
                .or_insert(Attributes::new(mask));
            });
          }
          if was_user_protected || was_builtin_protected {
            unprotected_syms.push(Expr::String(sym.clone()));
          }
        }
      }
      return Some(Ok(Expr::List(unprotected_syms.into())));
    }
    "Clear" => {
      for arg in args {
        match arg {
          Expr::Identifier(sym) | Expr::Constant(sym) => {
            // A Protected symbol keeps its definitions; wolframscript
            // reports `Clear::wrsym` and moves on to the next argument.
            if crate::evaluator::pattern_matching::is_symbol_protected(sym) {
              crate::emit_message(&format!(
                "Clear::wrsym: Symbol {sym} is Protected."
              ));
              continue;
            }
            ENV.with(|e| e.borrow_mut().remove(sym));
            crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
            crate::MEMO_VALUES.with(|m| m.borrow_mut().remove(sym));
          }
          Expr::String(pattern) => {
            for sym in matching_user_symbols(pattern) {
              ENV.with(|e| e.borrow_mut().remove(&sym));
              crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(&sym));
              crate::MEMO_VALUES.with(|m| m.borrow_mut().remove(&sym));
            }
          }
          _ => {}
        }
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    "ClearAll" => {
      // `ClearAll` is `Block`'s localization without the putting-back: take
      // every value the symbol has — own, down, sub, up, n, format, options,
      // messages, attributes — and drop the snapshot on the floor.
      let clear_one = |sym: &str| {
        drop(crate::evaluator::symbol_values::take_symbol_values(sym));
      };
      for arg in args {
        match arg {
          Expr::Identifier(sym) | Expr::Constant(sym) => {
            // Same Protected guard as `Clear`, with the `ClearAll` tag.
            if crate::evaluator::pattern_matching::is_symbol_protected(sym) {
              crate::emit_message(&format!(
                "ClearAll::wrsym: Symbol {sym} is Protected."
              ));
              continue;
            }
            clear_one(sym);
          }
          Expr::String(pattern) => {
            for sym in matching_user_symbols(pattern) {
              clear_one(&sym);
            }
          }
          _ => {}
        }
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    _ => {}
  }
  None
}

/// Resolve a Wolfram-style symbol pattern (e.g. `"Global`*"`, `"x*"`,
/// `"Global`x"`) to the matching user-defined symbols tracked by Woxi.
/// Woxi stores user symbols without a context prefix, so `Global`x` and
/// `x` refer to the same symbol here.
fn matching_user_symbols(pattern: &str) -> Vec<String> {
  let simple_pattern = pattern.strip_prefix("Global`").unwrap_or(pattern);
  // Pre-compute the user-defined symbol list once so we don't borrow
  // ENV/FUNC_DEFS while they are being mutated by the caller.
  let names = crate::get_defined_names();
  if !simple_pattern.contains('*') && !simple_pattern.contains('@') {
    return if names.iter().any(|n| n == simple_pattern) {
      vec![simple_pattern.to_string()]
    } else {
      Vec::new()
    };
  }
  let regex_pattern = format!(
    "^{}$",
    simple_pattern
      .replace('.', "\\.")
      .replace('*', ".*")
      .replace('@', "[a-z]+")
  );
  match regex::Regex::new(&regex_pattern) {
    Ok(re) => names.into_iter().filter(|n| re.is_match(n)).collect(),
    Err(_) => Vec::new(),
  }
}

#[cfg(test)]
mod unprotected_builtins_tests {
  use super::*;

  /// `is_unprotected_builtin` binary-searches, so an out-of-order entry would
  /// silently stop matching.
  #[test]
  fn the_list_is_free_of_duplicates() {
    assert!(
      UNPROTECTED_BUILTINS.len() == UNPROTECTED_BUILTINS_HASH.len(),
      "UNPROTECTED_BUILTINS must be duplicate-free"
    );
    assert!(
      UNPROTECTED_BUILTINS
        .iter()
        .all(|n| is_unprotected_builtin(n))
    );
  }
}
