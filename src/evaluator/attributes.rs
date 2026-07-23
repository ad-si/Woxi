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
      let attr: Vec<String> = match &args[1] {
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
        let mut locked = false;
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(&"Locked".to_string())
            {
              crate::emit_message(&format!(
                "Attributes::locked: Symbol {} is locked.",
                func_name
              ));
              locked = true;
              continue;
            }
            let entry = attrs.entry(func_name.clone()).or_insert_with(Vec::new);
            for a in &attr {
              if !entry.contains(a) {
                entry.push(a.clone());
              }
            }
          }
        });
        // Re-adding a builtin attribute via SetAttributes prunes it from
        // the removed-tracking, so `Attributes[sym]` once again reports it.
        crate::FUNC_ATTRS_REMOVED.with(|m| {
          let mut removed = m.borrow_mut();
          for func_name in &func_names {
            if let Some(entry) = removed.get_mut(func_name) {
              entry.retain(|a| !attr.contains(a));
            }
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
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(&"Locked".to_string())
            {
              crate::emit_message(&format!(
                "Attributes::locked: Symbol {} is locked.",
                func_name
              ));
              continue;
            }
            if let Some(entry) = attrs.get_mut(func_name) {
              entry.retain(|a| !to_remove.contains(a));
            }
          }
        });
        // Remove from builtin attributes via the removed-tracking, mirroring
        // how Unprotect handles the Protected attribute.
        crate::FUNC_ATTRS_REMOVED.with(|m| {
          let mut removed = m.borrow_mut();
          for func_name in &func_names {
            let builtin = get_builtin_attributes(func_name);
            for a in &to_remove {
              if builtin.contains(&a.as_str()) {
                let entry =
                  removed.entry(func_name.clone()).or_insert_with(Vec::new);
                if !entry.contains(a) {
                  entry.push(a.clone());
                }
              }
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
          let was_builtin = get_builtin_attributes(sym).contains(&"Protected");
          if was_builtin {
            crate::FUNC_ATTRS_REMOVED.with(|m| {
              let mut removed = m.borrow_mut();
              if let Some(entry) = removed.get_mut(sym) {
                entry.retain(|a| a != "Protected");
              }
            });
          } else {
            crate::FUNC_ATTRS.with(|m| {
              let mut attrs = m.borrow_mut();
              let entry = attrs.entry(sym.clone()).or_insert_with(Vec::new);
              if !entry.contains(&"Protected".to_string()) {
                entry.push("Protected".to_string());
              }
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
            if builtin.contains(&"Locked") {
              true
            } else {
              crate::FUNC_ATTRS.with(|m| {
                m.borrow()
                  .get(sym.as_str())
                  .is_some_and(|attrs| attrs.contains(&"Locked".to_string()))
              })
            }
          };
          if is_locked {
            crate::emit_message(&format!(
              "Protect::locked: Symbol {} is locked.",
              sym
            ));
            continue;
          }
          // A symbol counts as Protected if either its builtin default
          // attributes or its user-stored attributes contain "Protected".
          let was_user_protected = crate::FUNC_ATTRS.with(|m| {
            let mut attrs = m.borrow_mut();
            if let Some(entry) = attrs.get_mut(sym) {
              let before_len = entry.len();
              entry.retain(|a| a != "Protected");
              before_len != entry.len()
            } else {
              false
            }
          });
          let was_builtin_protected =
            get_builtin_attributes(sym).contains(&"Protected");
          if was_builtin_protected {
            crate::FUNC_ATTRS_REMOVED.with(|m| {
              let mut removed = m.borrow_mut();
              let entry = removed.entry(sym.clone()).or_insert_with(Vec::new);
              if !entry.contains(&"Protected".to_string()) {
                entry.push("Protected".to_string());
              }
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
      let clear_one = |sym: &str| {
        ENV.with(|e| e.borrow_mut().remove(sym));
        crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
        crate::MEMO_VALUES.with(|m| m.borrow_mut().remove(sym));
        crate::FUNC_ATTRS.with(|m| m.borrow_mut().remove(sym));
        crate::FUNC_OPTIONS.with(|m| m.borrow_mut().remove(sym));
        crate::FUNC_OPTIONS_DELAYED.with(|m| {
          m.borrow_mut().remove(sym);
        });
        // Drop any `Default[sym, …] := v` rules — they live keyed under
        // `Default` but reference this symbol as their first slot, and
        // ClearAll[sym] removes the symbol's DefaultValues alongside its
        // DownValues / Options / etc.
        crate::FUNC_DEFS.with(|m| {
          if let Some(entries) = m.borrow_mut().get_mut("Default") {
            entries.retain(|(params, _, _, _, _, _)| {
              params.first().is_none_or(|p| p != sym)
            });
          }
        });
        crate::FUNC_OPTS_INLINE.with(|m| m.borrow_mut().remove(sym));
        crate::evaluator::assignment::FORMAT_VALUES
          .with(|m| m.borrow_mut().remove(sym));
        crate::evaluator::assignment::SUB_VALUES
          .with(|m| m.borrow_mut().remove(sym));
        crate::evaluator::assignment::N_VALUES
          .with(|m| m.borrow_mut().remove(sym));
        // Mark every builtin attribute as removed so `Attributes[sym]`
        // returns `{}` after ClearAll, matching wolframscript.
        let builtin = get_builtin_attributes(sym);
        if !builtin.is_empty() {
          crate::FUNC_ATTRS_REMOVED.with(|m| {
            let mut removed = m.borrow_mut();
            let entry = removed.entry(sym.to_string()).or_insert_with(Vec::new);
            for a in builtin {
              if !entry.contains(&a.to_string()) {
                entry.push(a.to_string());
              }
            }
          });
        }
        let up_defs = crate::UPVALUES.with(|m| m.borrow_mut().remove(sym));
        if let Some(up_defs) = up_defs {
          for (
            outer_func,
            params,
            _conds,
            _defaults,
            _heads,
            body,
            _orig_lhs,
            _orig_body,
          ) in &up_defs
          {
            let body_str = expr_to_string(body);
            crate::FUNC_DEFS.with(|m| {
              if let Some(entry) = m.borrow_mut().get_mut(outer_func) {
                entry.retain(|(p, _, _, _, _, b)| {
                  !(p == params && expr_to_string(b) == body_str)
                });
              }
            });
          }
        }
      };
      for arg in args {
        match arg {
          Expr::Identifier(sym) | Expr::Constant(sym) => clear_one(sym),
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
