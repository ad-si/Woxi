//! Differential fuzzer: woxi vs. wolframscript.
//!
//! Generates random, terminating Wolfram Language expressions built only
//! from functions that `functions.csv` marks as implemented, evaluates each
//! through `woxi eval` and through an oracle (`wolframscript -code`), and
//! reports every output divergence — the project's contract is that the two
//! must match byte for byte.
//!
//! Strategy:
//!  1. Cases are pre-filtered in batches: one program per batch containing
//!     `Print[InputForm[e_i]]` statements separated by marker prints, so a
//!     single (slow) wolframscript start-up covers many cases.
//!  2. Any case whose batch segment differs is re-checked individually with
//!     the bare expression — exactly the `woxi eval` / `wolframscript
//!     -code` pairing the CLI doc tests use — so batch scaffolding can
//!     never produce a false positive.
//!  3. Confirmed divergences are greedily shrunk to a minimal expression
//!     before being reported.
//!
//! Outputs are compared as a sorted line-bag over stdout + stderr because
//! the two tools route messages to different streams (the CLI doc tests
//! need `output_stream: combined` for the same reason); `--compare` gives
//! stricter modes.
//!
//! Gated behind the non-default `diff-fuzz` feature. Run via `make fuzz-diff`
//! or:
//!   cargo run --features diff-fuzz --bin woxi-diff-fuzz -- --cases 200 --seed 42

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use clap::Parser;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
  name = "woxi-diff-fuzz",
  about = "Differential fuzzer comparing woxi against wolframscript"
)]
struct Cli {
  /// Number of random cases to generate and compare
  #[arg(long, default_value_t = 100)]
  cases: usize,

  /// Cases per wolframscript invocation (batching amortizes its start-up)
  #[arg(long, default_value_t = 20)]
  batch_size: usize,

  /// Master seed (default: derived from the clock; printed for replay)
  #[arg(long)]
  seed: Option<u64>,

  /// Maximum expression nesting depth
  #[arg(long, default_value_t = 4)]
  max_depth: u32,

  /// Oracle: "auto", "wolframscript", or "woxi" (self-check)
  #[arg(long, default_value = "auto")]
  oracle: String,

  /// Path of the wolframscript binary
  #[arg(long, default_value = "wolframscript")]
  wolframscript: String,

  /// Path of the woxi binary (default: sibling of this executable)
  #[arg(long)]
  woxi: Option<PathBuf>,

  /// Per-invocation oracle timeout in seconds
  #[arg(long, default_value_t = 60)]
  timeout: u64,

  /// Output comparison mode: "bag" (sorted combined lines, default),
  /// "combined" (exact stdout+stderr), or "stdout" (exact stdout only)
  #[arg(long, default_value = "bag")]
  compare: String,

  /// Maximum oracle calls spent shrinking each finding (0 disables)
  #[arg(long, default_value_t = 32)]
  shrink_budget: u32,

  /// Only print the generated case programs, one per line, without
  /// evaluating anything (also handy for seeding the libFuzzer corpus)
  #[arg(long)]
  print_cases: bool,
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64) — no dependency, stable across platforms
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
  fn new(seed: u64) -> Self {
    Rng(seed)
  }

  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  /// Uniform value in `0..n` (n > 0).
  fn below(&mut self, n: u64) -> u64 {
    self.next_u64() % n
  }

  /// Uniform value in `lo..=hi`.
  fn range(&mut self, lo: i64, hi: i64) -> i64 {
    debug_assert!(lo <= hi);
    lo + self.below((hi - lo + 1) as u64) as i64
  }

  /// True with probability `num/den`.
  fn chance(&mut self, num: u64, den: u64) -> bool {
    self.below(den) < num
  }

  fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
    &xs[self.below(xs.len() as u64) as usize]
  }
}

// ---------------------------------------------------------------------------
// Expression AST + printer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Expr {
  Int(i64),
  /// numerator / denominator (denominator > 0), printed as Divide[a, b]
  Rational(i64, i64),
  /// Value in tenths, e.g. 35 → 3.5 — keeps real literals short and exact
  RealTenths(i64),
  Str(String),
  Sym(&'static str),
  /// Raw code printed verbatim (pure functions like "#^2 &")
  Raw(&'static str),
  List(Vec<Expr>),
  Call(&'static str, Vec<Expr>),
}

impl Expr {
  fn to_code(&self) -> String {
    match self {
      Expr::Int(n) => n.to_string(),
      Expr::Rational(a, b) => format!("Divide[{a}, {b}]"),
      Expr::RealTenths(t) => {
        let sign = if *t < 0 { "-" } else { "" };
        format!("{sign}{}.{}", t.abs() / 10, t.abs() % 10)
      }
      Expr::Str(s) => format!("\"{s}\""),
      Expr::Sym(s) => (*s).to_string(),
      Expr::Raw(s) => format!("({s})"),
      Expr::List(xs) => {
        let inner: Vec<String> = xs.iter().map(Expr::to_code).collect();
        format!("{{{}}}", inner.join(", "))
      }
      Expr::Call(head, args) => {
        let inner: Vec<String> = args.iter().map(Expr::to_code).collect();
        format!("{head}[{}]", inner.join(", "))
      }
    }
  }

  /// Structural size used by the shrinker to guarantee progress.
  fn size(&self) -> u64 {
    match self {
      Expr::Int(n) => 1 + n.unsigned_abs(),
      Expr::Rational(a, b) => 2 + a.unsigned_abs() + b.unsigned_abs(),
      Expr::RealTenths(t) => 2 + t.unsigned_abs(),
      Expr::Str(s) => 1 + s.len() as u64,
      Expr::Sym(_) | Expr::Raw(_) => 1,
      Expr::List(xs) => 2 + xs.iter().map(Expr::size).sum::<u64>(),
      Expr::Call(_, args) => 2 + args.iter().map(Expr::size).sum::<u64>(),
    }
  }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Argument shapes a generated function call can request.
#[derive(Clone, Copy, Debug)]
enum Arg {
  /// Any expression (numbers, lists, strings, booleans, …)
  Any,
  /// Numeric-ish expression (may nest arithmetic, may stay symbolic via Pi)
  Num,
  /// Boolean-ish expression (True/False or a comparison)
  Bool,
  /// Short string literal from a shell-safe alphabet
  Str,
  /// List of numeric expressions (possibly empty)
  ListNum,
  /// Non-empty list of numeric expressions
  ListNum1,
  /// List of anything, possibly nested (for Flatten, Length, …)
  ListAny,
  /// Integer literal in 0..=max
  Nat(i64),
  /// Integer literal in lo..=hi
  IntIn(i64, i64),
  /// Polynomial in x (for Expand, D, Simplify, …)
  Poly,
  /// Product/power of small polynomials (for Expand, Factor)
  PolyProd,
  /// Rational function: Poly / Poly (for Together, Cancel, Apart, …)
  RatFn,
  /// Polynomial in x and y (for multivariate Expand, Factor, …)
  PolyXY,
  /// Arithmetic over small square roots (radical canonicalization)
  RadNum,
  /// Pure function mapping numbers to numbers (for Map)
  PureFn,
  /// Predicate (for Select)
  PredFn,
  /// The symbol x (differentiation variable)
  VarX,
  /// The symbol y (second variable, for multivariate forms)
  VarY,

  // --- Numeric shapes -------------------------------------------------
  /// Exact numeric leaf: integer, rational, Pi or E — never a machine
  /// real, so the special functions below compare symbolic results
  /// instead of the last digit of a float.
  ExactNum,
  /// Rational number (ContinuedFraction, Rationalize, …)
  RatNum,
  /// Argument for the trigonometric family: mostly rational multiples of
  /// Pi, whose exact values the CAS is expected to know.
  TrigNum,
  /// Numeric value in [-1, 1] (ArcSin, ArcTanh, InverseErf, …)
  UnitNum,
  /// Strictly positive numeric (Log, Gamma, GeometricMean, …)
  PosNum,
  /// Complex number `a + b I`
  ComplexNum,
  /// Odd integer in 1..=max (JacobiSymbol's modulus, …)
  OddNat(i64),
  /// Fraction `k/den` with k in 0..=max_k — a bounded ratio for Quantile,
  /// TrimmedMean and friends, which reject anything outside their range.
  Frac(i64, i64),

  // --- List shapes ----------------------------------------------------
  /// List of small integers (possibly empty)
  ListInt,
  /// Non-empty list of small integers
  ListInt1,
  /// List of at least two small integers (Variance, Skewness, …)
  ListInt2,
  /// Non-empty list of strictly positive small integers
  ListPosInt1,
  /// Non-empty list of decimal digits (FromDigits, …)
  ListDigit,
  /// Non-empty list of character codes (FromCharacterCode)
  ListCharCode,
  /// List of short strings
  ListStr,
  /// Two distinct short strings (AssociationThread keys)
  StrPair,
  /// Square integer matrix, 2×2 or 3×3
  Matrix,
  /// Square integer matrix, always 2×2 — keeps Inverse/Eigenvalues short
  Matrix2,
  /// Vector of exactly two small integers
  Vec2,
  /// Vector of exactly three small integers
  Vec3,
  /// Two equal-length vectors, as `{{…}, {…}}` (MapThread)
  VecPair,
  /// Pair of positive dimensions, as `{m, n}` (ArrayReshape)
  DimPair,
  /// Association from short string keys to small integers
  Assoc,

  // --- Pattern / rule / operator shapes -------------------------------
  /// Blank pattern, possibly head-constrained (`_`, `_Integer`, …)
  TypePat,
  /// Rule between two small integers (Replace, ReplacePart, …)
  IntRule,
  /// Rule between two one-character strings (StringReplace)
  StrRule,
  /// Rule `x -> value` (Limit, ReplaceAll on polynomials)
  VarRule,
  /// A bare builtin usable as an operator head (Apply, Fold, Outer, …)
  OpFn,
  /// One-character string, so string searches actually hit
  CharStr,

  // --- Symbolic shapes ------------------------------------------------
  /// Expression built from Sin/Cos/Tan of x and y
  TrigExpr,
  /// Expression built from Exp/Log/Sqrt of x and y
  ExpExpr,
  /// Polynomial in x of degree at most two, with a non-zero leading term
  LowPoly,
  /// Equation `LowPoly == 0` (Solve, NSolve, SolveValues)
  PolyEq,
  /// Series specification `{x, 0, n}`
  SeriesSpec,
  /// Iterator `{i, 1, n}` — pairs with `IterBody`, which uses the same `i`
  Iter,
  /// Summand/product term in `i`, for the `Iter` shapes
  IterBody,
}

struct FnSpec {
  name: &'static str,
  args: &'static [Arg],
}

const fn f(name: &'static str, args: &'static [Arg]) -> FnSpec {
  FnSpec { name, args }
}

/// Curated table of implemented functions with argument shapes that keep
/// generated programs terminating and (mostly) message-free. The table is
/// cross-checked against functions.csv at start-up so entries silently
/// dropped from the CSV can't linger here (a unit test guards the reverse).
const FN_SPECS: &[FnSpec] = &[
  // Arithmetic
  f("Plus", &[Arg::Num, Arg::Num]),
  f("Plus", &[Arg::Num, Arg::Num, Arg::Num]),
  f("Times", &[Arg::Num, Arg::Num]),
  f("Times", &[Arg::Num, Arg::Num, Arg::Num]),
  f("Subtract", &[Arg::Num, Arg::Num]),
  f("Divide", &[Arg::Num, Arg::Num]),
  f("Minus", &[Arg::Num]),
  f("Power", &[Arg::Num, Arg::IntIn(-3, 3)]),
  f("Abs", &[Arg::Num]),
  f("Sign", &[Arg::Num]),
  f("Min", &[Arg::Num, Arg::Num]),
  f("Max", &[Arg::Num, Arg::Num]),
  f("Min", &[Arg::ListNum1]),
  f("Max", &[Arg::ListNum1]),
  f("Mod", &[Arg::IntIn(-50, 50), Arg::IntIn(1, 9)]),
  f("Quotient", &[Arg::IntIn(-50, 50), Arg::IntIn(1, 9)]),
  f("GCD", &[Arg::IntIn(-60, 60), Arg::IntIn(-60, 60)]),
  f("LCM", &[Arg::IntIn(1, 30), Arg::IntIn(1, 30)]),
  f("Floor", &[Arg::Num]),
  f("Ceiling", &[Arg::Num]),
  f("Round", &[Arg::Num]),
  f("Total", &[Arg::ListNum]),
  f("Mean", &[Arg::ListNum1]),
  f("Factorial", &[Arg::Nat(8)]),
  f("Fibonacci", &[Arg::Nat(20)]),
  f("Sqrt", &[Arg::Nat(50)]),
  f("Numerator", &[Arg::Num]),
  f("Denominator", &[Arg::Num]),
  f("Boole", &[Arg::Bool]),
  f("Log", &[Arg::IntIn(1, 9)]),
  // Integer functions
  f("PrimeQ", &[Arg::IntIn(-20, 100)]),
  f("EvenQ", &[Arg::IntIn(-50, 50)]),
  f("OddQ", &[Arg::IntIn(-50, 50)]),
  f("IntegerQ", &[Arg::Any]),
  f("Divisors", &[Arg::IntIn(1, 60)]),
  f("IntegerDigits", &[Arg::Nat(100_000)]),
  // Lists
  f("Length", &[Arg::ListAny]),
  f("First", &[Arg::ListNum1]),
  f("Last", &[Arg::ListNum1]),
  f("Rest", &[Arg::ListNum1]),
  f("Most", &[Arg::ListNum1]),
  f("Reverse", &[Arg::ListAny]),
  f("Sort", &[Arg::ListNum]),
  f("Join", &[Arg::ListNum, Arg::ListNum]),
  f("Append", &[Arg::ListAny, Arg::Any]),
  f("Prepend", &[Arg::ListAny, Arg::Any]),
  f("Flatten", &[Arg::ListAny]),
  f("Union", &[Arg::ListNum, Arg::ListNum]),
  f("Intersection", &[Arg::ListNum, Arg::ListNum]),
  f("Complement", &[Arg::ListNum, Arg::ListNum]),
  f("Range", &[Arg::Nat(8)]),
  f("Count", &[Arg::ListNum, Arg::IntIn(-5, 5)]),
  f("MemberQ", &[Arg::ListNum, Arg::IntIn(-5, 5)]),
  f("Map", &[Arg::PureFn, Arg::ListNum]),
  f("Select", &[Arg::PredFn, Arg::ListNum]),
  // Strings
  f("StringLength", &[Arg::Str]),
  f("StringJoin", &[Arg::Str, Arg::Str]),
  f("ToUpperCase", &[Arg::Str]),
  f("ToLowerCase", &[Arg::Str]),
  f("StringReverse", &[Arg::Str]),
  f("Characters", &[Arg::Str]),
  // Logic + comparison
  f("Not", &[Arg::Bool]),
  f("And", &[Arg::Bool, Arg::Bool]),
  f("Or", &[Arg::Bool, Arg::Bool]),
  f("Xor", &[Arg::Bool, Arg::Bool]),
  f("If", &[Arg::Bool, Arg::Any, Arg::Any]),
  f("Equal", &[Arg::Num, Arg::Num]),
  f("Unequal", &[Arg::Num, Arg::Num]),
  f("Less", &[Arg::Num, Arg::Num]),
  f("LessEqual", &[Arg::Num, Arg::Num]),
  f("Greater", &[Arg::Num, Arg::Num]),
  f("GreaterEqual", &[Arg::Num, Arg::Num]),
  // Symbolic / CAS
  f("Expand", &[Arg::PolyProd]),
  f("Factor", &[Arg::PolyProd]),
  f("Simplify", &[Arg::Poly]),
  f("Together", &[Arg::Poly]),
  f("D", &[Arg::Poly, Arg::VarX]),
  f("Head", &[Arg::Any]),
  // Regression coverage for canonical-order and arithmetic fixes:
  // mixed-content sorts exercise the number-vs-symbol comparators, the
  // 3-argument forms exercise offsets/chains, and Factor/FactorInteger
  // exercise the Zassenhaus and Gaussian paths.
  f("Sort", &[Arg::ListAny]),
  f("Union", &[Arg::ListAny]),
  f("Max", &[Arg::Num, Arg::Num, Arg::Num]),
  f("Min", &[Arg::Num, Arg::Num, Arg::Num]),
  f(
    "Mod",
    &[Arg::IntIn(-50, 50), Arg::IntIn(1, 9), Arg::IntIn(-2, 2)],
  ),
  f("FactorInteger", &[Arg::IntIn(1, 500)]),
  f("Binomial", &[Arg::IntIn(-10, 12), Arg::IntIn(-3, 12)]),
  f("Rationalize", &[Arg::Num]),
  f("IntegerPartitions", &[Arg::Nat(8)]),
  // Previously unfuzzed list/string surface
  f("Accumulate", &[Arg::ListNum]),
  f("Differences", &[Arg::ListNum1]),
  f("Tally", &[Arg::ListNum]),
  f("DeleteDuplicates", &[Arg::ListNum]),
  f("Gather", &[Arg::ListNum]),
  f("Riffle", &[Arg::ListNum, Arg::IntIn(-5, 5)]),
  f("Partition", &[Arg::ListNum1, Arg::IntIn(1, 3)]),
  f("Position", &[Arg::ListNum, Arg::IntIn(-5, 5)]),
  f("Nest", &[Arg::PureFn, Arg::Num, Arg::Nat(4)]),
  f("ToString", &[Arg::Any]),
  f("StringSplit", &[Arg::Str]),
  f("StringRepeat", &[Arg::Str, Arg::Nat(3)]),
  f("Clip", &[Arg::Num]),
  f("UnitStep", &[Arg::Num]),
  f("Ramp", &[Arg::Num]),
  f("KroneckerDelta", &[Arg::Num, Arg::Num]),
  // Number theory
  f("EulerPhi", &[Arg::IntIn(1, 200)]),
  f("MoebiusMu", &[Arg::IntIn(1, 200)]),
  f("DivisorSigma", &[Arg::IntIn(0, 3), Arg::IntIn(1, 60)]),
  f("PrimePi", &[Arg::Nat(200)]),
  f("Prime", &[Arg::IntIn(1, 50)]),
  f("NextPrime", &[Arg::IntIn(-20, 100)]),
  f("CoprimeQ", &[Arg::IntIn(-60, 60), Arg::IntIn(-60, 60)]),
  f("SquareFreeQ", &[Arg::IntIn(1, 200)]),
  f("LucasL", &[Arg::Nat(20)]),
  f("CatalanNumber", &[Arg::Nat(10)]),
  f("HarmonicNumber", &[Arg::Nat(12)]),
  f("Multinomial", &[Arg::Nat(6), Arg::Nat(6)]),
  f("Pochhammer", &[Arg::IntIn(-5, 5), Arg::Nat(5)]),
  f("StirlingS2", &[Arg::Nat(8), Arg::Nat(8)]),
  f("BitAnd", &[Arg::IntIn(-100, 100), Arg::IntIn(-100, 100)]),
  f("BitOr", &[Arg::IntIn(-100, 100), Arg::IntIn(-100, 100)]),
  f("BitXor", &[Arg::IntIn(-100, 100), Arg::IntIn(-100, 100)]),
  f("DigitCount", &[Arg::Nat(100_000)]),
  f("IntegerString", &[Arg::Nat(100_000), Arg::IntIn(2, 16)]),
  // Numeric parts
  f("IntegerPart", &[Arg::Num]),
  f("FractionalPart", &[Arg::Num]),
  f("Chop", &[Arg::Num]),
  f("Unitize", &[Arg::Num]),
  // More list surface
  f("RotateLeft", &[Arg::ListAny, Arg::IntIn(-3, 3)]),
  f("Split", &[Arg::ListNum]),
  f("Ordering", &[Arg::ListNum]),
  f("Median", &[Arg::ListNum1]),
  f("Commonest", &[Arg::ListNum1]),
  f("Norm", &[Arg::ListNum]),
  f("ConstantArray", &[Arg::Any, Arg::Nat(4)]),
  f("PadLeft", &[Arg::ListNum, Arg::Nat(6)]),
  f("Tuples", &[Arg::ListNum, Arg::IntIn(1, 2)]),
  // Predicates + canonical order probes
  f("Positive", &[Arg::Num]),
  f("Negative", &[Arg::Num]),
  f("NonNegative", &[Arg::Num]),
  f("NumberQ", &[Arg::Any]),
  f("AtomQ", &[Arg::Any]),
  f("FreeQ", &[Arg::ListNum, Arg::IntIn(-5, 5)]),
  f("SameQ", &[Arg::Any, Arg::Any]),
  f("UnsameQ", &[Arg::Any, Arg::Any]),
  f("Order", &[Arg::Any, Arg::Any]),
  f("OrderedQ", &[Arg::ListAny]),
  // More CAS surface
  f("FactorTerms", &[Arg::Poly]),
  f("FactorSquareFree", &[Arg::PolyProd]),
  f("Coefficient", &[Arg::Poly, Arg::VarX]),
  f("Exponent", &[Arg::Poly, Arg::VarX]),
  f("Discriminant", &[Arg::Poly, Arg::VarX]),
  // More string surface
  f("StringContainsQ", &[Arg::Str, Arg::Str]),
  f("StringCount", &[Arg::Str, Arg::Str]),
  f("StringPadLeft", &[Arg::Str, Arg::Nat(12)]),
  f("StringTrim", &[Arg::Str]),
  // Rational functions (fraction combining / cancellation paths)
  f("Together", &[Arg::RatFn]),
  f("Cancel", &[Arg::RatFn]),
  f("Apart", &[Arg::RatFn]),
  f("Simplify", &[Arg::RatFn]),
  f("Numerator", &[Arg::RatFn]),
  f("Denominator", &[Arg::RatFn]),
  f("D", &[Arg::RatFn, Arg::VarX]),
  // Multivariate polynomials
  f("Expand", &[Arg::PolyXY]),
  f("Factor", &[Arg::PolyXY]),
  f("FactorTerms", &[Arg::PolyXY]),
  f("FactorSquareFree", &[Arg::PolyXY]),
  f("D", &[Arg::PolyXY, Arg::VarX]),
  f("Coefficient", &[Arg::PolyXY, Arg::VarX]),
  f("Exponent", &[Arg::PolyXY, Arg::VarX]),
  // Radical arithmetic (canonicalization of square roots)
  f("Simplify", &[Arg::RadNum]),
  f("Expand", &[Arg::RadNum]),
  f("Abs", &[Arg::RadNum]),
  f("Sign", &[Arg::RadNum]),
  f("Numerator", &[Arg::RadNum]),
  // Trigonometric and hyperbolic. The arguments are rational multiples of
  // Pi far more often than not, so these compare closed forms rather than
  // float digits.
  f("Sin", &[Arg::TrigNum]),
  f("Cos", &[Arg::TrigNum]),
  f("Tan", &[Arg::TrigNum]),
  f("Cot", &[Arg::TrigNum]),
  f("Sec", &[Arg::TrigNum]),
  f("Csc", &[Arg::TrigNum]),
  f("Sinh", &[Arg::TrigNum]),
  f("Cosh", &[Arg::TrigNum]),
  f("Tanh", &[Arg::TrigNum]),
  f("Coth", &[Arg::TrigNum]),
  f("Sech", &[Arg::TrigNum]),
  f("Csch", &[Arg::TrigNum]),
  f("Haversine", &[Arg::TrigNum]),
  f("Sinc", &[Arg::TrigNum]),
  f("ArcSin", &[Arg::UnitNum]),
  f("ArcCos", &[Arg::UnitNum]),
  f("ArcTanh", &[Arg::UnitNum]),
  f("ArcTan", &[Arg::ExactNum]),
  f("ArcTan", &[Arg::IntIn(-4, 4), Arg::IntIn(-4, 4)]),
  f("ArcCot", &[Arg::ExactNum]),
  f("ArcSec", &[Arg::ExactNum]),
  f("ArcCsc", &[Arg::ExactNum]),
  f("ArcSinh", &[Arg::ExactNum]),
  f("ArcCosh", &[Arg::ExactNum]),
  f("ArcCoth", &[Arg::ExactNum]),
  f("Gudermannian", &[Arg::ExactNum]),
  // Exponentials, logarithms and roots
  f("Exp", &[Arg::ExactNum]),
  f("Log", &[Arg::IntIn(2, 12), Arg::PosNum]),
  f("Log2", &[Arg::PosNum]),
  f("Log10", &[Arg::PosNum]),
  f("Sqrt", &[Arg::ExactNum]),
  f("CubeRoot", &[Arg::ExactNum]),
  f("Surd", &[Arg::IntIn(-40, 40), Arg::IntIn(2, 5)]),
  f("Power", &[Arg::RadNum, Arg::IntIn(-2, 3)]),
  f("N", &[Arg::ExactNum]),
  f("N", &[Arg::ExactNum, Arg::IntIn(3, 25)]),
  // Special functions. Integer/rational arguments keep the results exact,
  // which is where the interesting reductions (and disagreements) live.
  f("Gamma", &[Arg::ExactNum]),
  f("Gamma", &[Arg::IntIn(1, 6), Arg::IntIn(0, 6)]),
  f("LogGamma", &[Arg::PosNum]),
  f("Beta", &[Arg::IntIn(1, 8), Arg::IntIn(1, 8)]),
  f("PolyGamma", &[Arg::PosNum]),
  f("PolyGamma", &[Arg::Nat(3), Arg::IntIn(1, 6)]),
  f("Erf", &[Arg::ExactNum]),
  f("Erfc", &[Arg::ExactNum]),
  f("Erfi", &[Arg::ExactNum]),
  f("InverseErf", &[Arg::UnitNum]),
  f("Zeta", &[Arg::IntIn(-6, 8)]),
  f("PolyLog", &[Arg::IntIn(-2, 4), Arg::UnitNum]),
  f("ExpIntegralE", &[Arg::Nat(4), Arg::PosNum]),
  f("ExpIntegralEi", &[Arg::ExactNum]),
  f("LogIntegral", &[Arg::PosNum]),
  f("SinIntegral", &[Arg::ExactNum]),
  f("CosIntegral", &[Arg::PosNum]),
  f("SinhIntegral", &[Arg::ExactNum]),
  f("CoshIntegral", &[Arg::PosNum]),
  f("FresnelS", &[Arg::ExactNum]),
  f("FresnelC", &[Arg::ExactNum]),
  f("AiryAi", &[Arg::ExactNum]),
  f("AiryBi", &[Arg::ExactNum]),
  f("BesselJ", &[Arg::IntIn(0, 4), Arg::ExactNum]),
  f("BesselY", &[Arg::IntIn(0, 4), Arg::PosNum]),
  f("BesselI", &[Arg::IntIn(0, 4), Arg::ExactNum]),
  f("BesselK", &[Arg::IntIn(0, 4), Arg::PosNum]),
  f("ArithmeticGeometricMean", &[Arg::PosNum, Arg::PosNum]),
  f(
    "Hypergeometric2F1",
    &[
      Arg::IntIn(-3, 3),
      Arg::IntIn(-3, 3),
      Arg::IntIn(1, 4),
      Arg::UnitNum,
    ],
  ),
  f(
    "Hypergeometric1F1",
    &[Arg::IntIn(-3, 3), Arg::IntIn(1, 4), Arg::ExactNum],
  ),
  // Orthogonal polynomials — symbolic in x as well as evaluated at a
  // point, since the two paths are usually separate code.
  f("LegendreP", &[Arg::Nat(5), Arg::ExactNum]),
  f("LegendreP", &[Arg::Nat(4), Arg::VarX]),
  f("ChebyshevT", &[Arg::Nat(6), Arg::ExactNum]),
  f("ChebyshevT", &[Arg::Nat(4), Arg::VarX]),
  f("ChebyshevU", &[Arg::Nat(6), Arg::ExactNum]),
  f("HermiteH", &[Arg::Nat(6), Arg::ExactNum]),
  f("HermiteH", &[Arg::Nat(4), Arg::VarX]),
  f("LaguerreL", &[Arg::Nat(6), Arg::ExactNum]),
  f(
    "GegenbauerC",
    &[Arg::Nat(5), Arg::IntIn(1, 3), Arg::ExactNum],
  ),
  f(
    "JacobiP",
    &[
      Arg::Nat(4),
      Arg::IntIn(0, 3),
      Arg::IntIn(0, 3),
      Arg::ExactNum,
    ],
  ),
  // Combinatorial sequences
  f("BernoulliB", &[Arg::Nat(12)]),
  f("EulerE", &[Arg::Nat(10)]),
  f("StirlingS1", &[Arg::Nat(8), Arg::Nat(8)]),
  f("Subfactorial", &[Arg::Nat(10)]),
  f("Hyperfactorial", &[Arg::Nat(5)]),
  f("BarnesG", &[Arg::IntIn(1, 8)]),
  f("Factorial2", &[Arg::IntIn(-1, 12)]),
  f("PartitionsP", &[Arg::Nat(30)]),
  f("PartitionsQ", &[Arg::Nat(30)]),
  // Number theory
  f(
    "PowerMod",
    &[Arg::IntIn(-20, 20), Arg::Nat(10), Arg::IntIn(1, 30)],
  ),
  f("ModularInverse", &[Arg::IntIn(1, 30), Arg::IntIn(2, 30)]),
  f(
    "MultiplicativeOrder",
    &[Arg::IntIn(2, 20), Arg::IntIn(3, 30)],
  ),
  f("JacobiSymbol", &[Arg::IntIn(-30, 30), Arg::OddNat(31)]),
  f(
    "KroneckerSymbol",
    &[Arg::IntIn(-30, 30), Arg::IntIn(-30, 30)],
  ),
  f("PrimeOmega", &[Arg::IntIn(1, 500)]),
  f("PrimeNu", &[Arg::IntIn(1, 500)]),
  f("DivisorSum", &[Arg::IntIn(1, 60), Arg::PureFn]),
  f("PrimePowerQ", &[Arg::IntIn(-20, 200)]),
  f("CompositeQ", &[Arg::IntIn(-20, 200)]),
  f(
    "IntegerExponent",
    &[Arg::IntIn(-1000, 1000), Arg::IntIn(2, 10)],
  ),
  f("IntegerLength", &[Arg::IntIn(-100_000, 100_000)]),
  f("IntegerReverse", &[Arg::IntIn(-100_000, 100_000)]),
  f("DigitSum", &[Arg::Nat(100_000)]),
  f("FromDigits", &[Arg::ListDigit]),
  f("FromDigits", &[Arg::ListDigit, Arg::IntIn(2, 16)]),
  f("RealDigits", &[Arg::PosNum]),
  f("ContinuedFraction", &[Arg::RatNum]),
  f("FromContinuedFraction", &[Arg::ListPosInt1]),
  f("Convergents", &[Arg::ListPosInt1]),
  f("ExtendedGCD", &[Arg::IntIn(-60, 60), Arg::IntIn(-60, 60)]),
  f("Cyclotomic", &[Arg::IntIn(1, 12), Arg::VarX]),
  f("CarmichaelLambda", &[Arg::IntIn(1, 200)]),
  f("SquaresR", &[Arg::IntIn(1, 4), Arg::Nat(60)]),
  // Complex numbers
  f("Re", &[Arg::ComplexNum]),
  f("Im", &[Arg::ComplexNum]),
  f("Abs", &[Arg::ComplexNum]),
  f("Arg", &[Arg::ComplexNum]),
  f("Conjugate", &[Arg::ComplexNum]),
  f("AbsArg", &[Arg::ComplexNum]),
  f("ReIm", &[Arg::ComplexNum]),
  f("Sqrt", &[Arg::ComplexNum]),
  f("Exp", &[Arg::ComplexNum]),
  f("Power", &[Arg::ComplexNum, Arg::IntIn(-2, 3)]),
  // List surface: structural edits, pattern-driven queries and the
  // higher-order functions
  f("Drop", &[Arg::ListInt, Arg::IntIn(-3, 3)]),
  f("Delete", &[Arg::ListInt1, Arg::IntIn(-3, 3)]),
  f(
    "Insert",
    &[Arg::ListInt, Arg::IntIn(-9, 9), Arg::IntIn(1, 3)],
  ),
  f("ReplacePart", &[Arg::ListInt1, Arg::IntRule]),
  f("Replace", &[Arg::ListInt, Arg::IntRule]),
  f("ReplaceAll", &[Arg::ListInt, Arg::IntRule]),
  f("ReplaceAll", &[Arg::Poly, Arg::VarRule]),
  f("Cases", &[Arg::ListAny, Arg::TypePat]),
  f("DeleteCases", &[Arg::ListAny, Arg::TypePat]),
  f("Count", &[Arg::ListAny, Arg::TypePat]),
  f("Position", &[Arg::ListAny, Arg::TypePat]),
  f("FreeQ", &[Arg::ListAny, Arg::TypePat]),
  f("MemberQ", &[Arg::ListAny, Arg::TypePat]),
  f("Subsets", &[Arg::ListInt]),
  f("Subsets", &[Arg::ListInt, Arg::IntIn(0, 2)]),
  f("Permutations", &[Arg::ListInt]),
  f("SortBy", &[Arg::ListInt, Arg::PureFn]),
  f("MinimalBy", &[Arg::ListInt1, Arg::PureFn]),
  f("MaximalBy", &[Arg::ListInt1, Arg::PureFn]),
  f("TakeLargest", &[Arg::ListInt1, Arg::IntIn(1, 3)]),
  f("TakeSmallest", &[Arg::ListInt1, Arg::IntIn(1, 3)]),
  f("Counts", &[Arg::ListInt]),
  f("GatherBy", &[Arg::ListInt, Arg::PureFn]),
  f("SplitBy", &[Arg::ListInt, Arg::PureFn]),
  f("Select", &[Arg::ListInt, Arg::PredFn]),
  f("SubsetQ", &[Arg::ListInt, Arg::ListInt]),
  f("ContainsAll", &[Arg::ListInt, Arg::ListInt]),
  f("ContainsAny", &[Arg::ListInt, Arg::ListInt]),
  f("ContainsOnly", &[Arg::ListInt, Arg::ListInt]),
  f("ContainsExactly", &[Arg::ListInt, Arg::ListInt]),
  f("Apply", &[Arg::OpFn, Arg::ListNum]),
  f("Fold", &[Arg::OpFn, Arg::IntIn(-5, 5), Arg::ListInt]),
  f("FoldList", &[Arg::OpFn, Arg::IntIn(-5, 5), Arg::ListInt]),
  f("NestList", &[Arg::PureFn, Arg::Num, Arg::Nat(4)]),
  f("MapIndexed", &[Arg::PureFn, Arg::ListInt]),
  f("MapThread", &[Arg::OpFn, Arg::VecPair]),
  f("Outer", &[Arg::OpFn, Arg::Vec2, Arg::Vec2]),
  f("Array", &[Arg::PureFn, Arg::IntIn(1, 4)]),
  f("Ratios", &[Arg::ListInt1]),
  f("RotateRight", &[Arg::ListAny, Arg::IntIn(-3, 3)]),
  f("ArrayReshape", &[Arg::ListInt, Arg::DimPair]),
  f("ArrayPad", &[Arg::ListInt, Arg::IntIn(0, 3)]),
  f("Depth", &[Arg::ListAny]),
  f("Level", &[Arg::ListAny, Arg::IntIn(1, 2)]),
  f("Identity", &[Arg::Any]),
  f("Hold", &[Arg::Num]),
  f("AllTrue", &[Arg::ListInt, Arg::PredFn]),
  f("AnyTrue", &[Arg::ListInt, Arg::PredFn]),
  f("NoneTrue", &[Arg::ListInt, Arg::PredFn]),
  // Associations
  f("Keys", &[Arg::Assoc]),
  f("Values", &[Arg::Assoc]),
  f("KeySort", &[Arg::Assoc]),
  f("KeyDrop", &[Arg::Assoc, Arg::CharStr]),
  f("KeyTake", &[Arg::Assoc, Arg::CharStr]),
  f("Lookup", &[Arg::Assoc, Arg::CharStr]),
  f("Normal", &[Arg::Assoc]),
  f("Length", &[Arg::Assoc]),
  f("Map", &[Arg::PureFn, Arg::Assoc]),
  f("AssociationThread", &[Arg::StrPair, Arg::Vec2]),
  // Strings
  f("StringTake", &[Arg::Str, Arg::IntIn(-3, 3)]),
  f("StringDrop", &[Arg::Str, Arg::IntIn(-3, 3)]),
  f("StringPosition", &[Arg::Str, Arg::CharStr]),
  f("StringReplace", &[Arg::Str, Arg::StrRule]),
  f("StringInsert", &[Arg::Str, Arg::CharStr, Arg::IntIn(1, 3)]),
  f("StringDelete", &[Arg::Str, Arg::CharStr]),
  f("StringPadRight", &[Arg::Str, Arg::Nat(12)]),
  f("StringRiffle", &[Arg::ListStr]),
  f("StringRiffle", &[Arg::ListStr, Arg::CharStr]),
  f("StringPartition", &[Arg::Str, Arg::IntIn(1, 3)]),
  f("StringCases", &[Arg::Str, Arg::CharStr]),
  f("StringSplit", &[Arg::Str, Arg::CharStr]),
  f("StringMatchQ", &[Arg::Str, Arg::Str]),
  f("StringStartsQ", &[Arg::Str, Arg::CharStr]),
  f("StringEndsQ", &[Arg::Str, Arg::CharStr]),
  f("StringFreeQ", &[Arg::Str, Arg::CharStr]),
  f("StringRotateLeft", &[Arg::Str, Arg::IntIn(-3, 3)]),
  f("Capitalize", &[Arg::Str]),
  f("Decapitalize", &[Arg::Str]),
  f("ToCharacterCode", &[Arg::Str]),
  f("FromCharacterCode", &[Arg::ListCharCode]),
  f("LetterQ", &[Arg::CharStr]),
  f("DigitQ", &[Arg::CharStr]),
  f("UpperCaseQ", &[Arg::Str]),
  f("LowerCaseQ", &[Arg::Str]),
  f("EditDistance", &[Arg::Str, Arg::Str]),
  f("LongestCommonSubsequence", &[Arg::Str, Arg::Str]),
  f("HammingDistance", &[Arg::Vec3, Arg::Vec3]),
  // Linear algebra. Matrices stay 2×2/3×3 with small integer entries, so
  // exact results (including eigenvalue radicals) remain short enough to
  // diff.
  f("Det", &[Arg::Matrix]),
  f("Tr", &[Arg::Matrix]),
  f("Transpose", &[Arg::Matrix]),
  f("Dimensions", &[Arg::Matrix]),
  f("ArrayDepth", &[Arg::Matrix]),
  f("Diagonal", &[Arg::Matrix]),
  f("MatrixRank", &[Arg::Matrix]),
  f("NullSpace", &[Arg::Matrix]),
  f("RowReduce", &[Arg::Matrix]),
  f("Minors", &[Arg::Matrix]),
  f("Permanent", &[Arg::Matrix]),
  f("Total", &[Arg::Matrix]),
  f("Total", &[Arg::Matrix, Arg::IntIn(1, 2)]),
  f("Flatten", &[Arg::Matrix, Arg::IntIn(1, 2)]),
  f("Inverse", &[Arg::Matrix2]),
  f("Adjugate", &[Arg::Matrix2]),
  f("Eigenvalues", &[Arg::Matrix2]),
  f("MatrixPower", &[Arg::Matrix2, Arg::IntIn(0, 4)]),
  f("CharacteristicPolynomial", &[Arg::Matrix2, Arg::VarX]),
  f("LinearSolve", &[Arg::Matrix2, Arg::Vec2]),
  f("KroneckerProduct", &[Arg::Matrix2, Arg::Matrix2]),
  f("Dot", &[Arg::Matrix2, Arg::Matrix2]),
  f("Dot", &[Arg::Vec3, Arg::Vec3]),
  f("Norm", &[Arg::Matrix2]),
  f("Cross", &[Arg::Vec3, Arg::Vec3]),
  f("Normalize", &[Arg::Vec3]),
  f("DiagonalMatrix", &[Arg::Vec3]),
  f("HankelMatrix", &[Arg::Vec3]),
  f("ToeplitzMatrix", &[Arg::Vec3]),
  f("IdentityMatrix", &[Arg::IntIn(1, 3)]),
  f("HilbertMatrix", &[Arg::IntIn(1, 4)]),
  // Polynomial algebra
  f("Solve", &[Arg::PolyEq, Arg::VarX]),
  f("SolveValues", &[Arg::PolyEq, Arg::VarX]),
  f("NSolve", &[Arg::PolyEq, Arg::VarX]),
  f("PolynomialQuotient", &[Arg::Poly, Arg::LowPoly, Arg::VarX]),
  f("PolynomialRemainder", &[Arg::Poly, Arg::LowPoly, Arg::VarX]),
  f("PolynomialGCD", &[Arg::PolyProd, Arg::PolyProd]),
  f("PolynomialLCM", &[Arg::LowPoly, Arg::LowPoly]),
  f("Resultant", &[Arg::LowPoly, Arg::LowPoly, Arg::VarX]),
  f("PolynomialQ", &[Arg::RatFn, Arg::VarX]),
  f("Collect", &[Arg::PolyXY, Arg::VarX]),
  f("CoefficientList", &[Arg::Poly, Arg::VarX]),
  f("Variables", &[Arg::PolyXY]),
  f("HornerForm", &[Arg::Poly]),
  f("Decompose", &[Arg::PolyProd, Arg::VarX]),
  f("MonomialList", &[Arg::PolyXY]),
  f("FactorList", &[Arg::PolyProd]),
  f("FactorSquareFreeList", &[Arg::PolyProd]),
  f("ExpandAll", &[Arg::PolyProd]),
  f("MinimalPolynomial", &[Arg::RadNum, Arg::VarX]),
  // Calculus
  f("Integrate", &[Arg::Poly, Arg::VarX]),
  f("Integrate", &[Arg::RatFn, Arg::VarX]),
  f("Integrate", &[Arg::TrigExpr, Arg::VarX]),
  f("D", &[Arg::TrigExpr, Arg::VarX]),
  f("D", &[Arg::ExpExpr, Arg::VarX]),
  f("D", &[Arg::PolyXY, Arg::VarX, Arg::VarY]),
  f("Limit", &[Arg::RatFn, Arg::VarRule]),
  f("Series", &[Arg::RatFn, Arg::SeriesSpec]),
  f("Sum", &[Arg::IterBody, Arg::Iter]),
  f("Product", &[Arg::IterBody, Arg::Iter]),
  f("Table", &[Arg::IterBody, Arg::Iter]),
  // Rewriters
  f("FullSimplify", &[Arg::RadNum]),
  f("FullSimplify", &[Arg::RatFn]),
  f("Simplify", &[Arg::TrigExpr]),
  f("TrigExpand", &[Arg::TrigExpr]),
  f("TrigReduce", &[Arg::TrigExpr]),
  f("TrigToExp", &[Arg::TrigExpr]),
  f("ExpToTrig", &[Arg::ExpExpr]),
  f("PowerExpand", &[Arg::ExpExpr]),
  f("ComplexExpand", &[Arg::TrigExpr]),
  // Logic
  f("Implies", &[Arg::Bool, Arg::Bool]),
  f("Nand", &[Arg::Bool, Arg::Bool]),
  f("Nor", &[Arg::Bool, Arg::Bool]),
  f("Xnor", &[Arg::Bool, Arg::Bool]),
  f("Equivalent", &[Arg::Bool, Arg::Bool]),
  f("TrueQ", &[Arg::Any]),
  // Descriptive statistics
  f("Variance", &[Arg::ListInt2]),
  f("StandardDeviation", &[Arg::ListInt2]),
  f("Kurtosis", &[Arg::ListInt2]),
  f("Skewness", &[Arg::ListInt2]),
  f("Standardize", &[Arg::ListInt2]),
  f("Quantile", &[Arg::ListInt1, Arg::Frac(4, 4)]),
  f("RootMeanSquare", &[Arg::ListInt1]),
  f("MeanDeviation", &[Arg::ListInt1]),
  f("MedianDeviation", &[Arg::ListInt1]),
  f("InterquartileRange", &[Arg::ListInt1]),
  f("Quartiles", &[Arg::ListInt1]),
  f("Rescale", &[Arg::ListInt1]),
  f("CentralMoment", &[Arg::ListInt1, Arg::IntIn(1, 4)]),
  f("Moment", &[Arg::ListInt1, Arg::IntIn(1, 4)]),
  f("TrimmedMean", &[Arg::ListInt1, Arg::Frac(3, 8)]),
  f("WinsorizedMean", &[Arg::ListInt1, Arg::Frac(3, 8)]),
  f("HarmonicMean", &[Arg::ListPosInt1]),
  f("GeometricMean", &[Arg::ListPosInt1]),
  f("Covariance", &[Arg::Vec3, Arg::Vec3]),
  // Rounding with an explicit granularity
  f("Round", &[Arg::Num, Arg::IntIn(1, 5)]),
  f("Floor", &[Arg::Num, Arg::IntIn(1, 5)]),
  f("Ceiling", &[Arg::Num, Arg::IntIn(1, 5)]),
  f("Mod", &[Arg::Num, Arg::IntIn(1, 9)]),
];

const PURE_FNS: &[&str] = &["#^2 &", "# + 1 &", "2*# &", "-# &", "Abs[#] &"];
const PRED_FNS: &[&str] = &["EvenQ", "OddQ", "PrimeQ", "# > 0 &", "# < 2 &"];
const STR_CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCXYZ0123456789 ";
/// Blank patterns for Cases/Count/Position/DeleteCases. `Expr::Raw`
/// parenthesises what it prints, which is harmless around a pattern.
const TYPE_PATS: &[&str] =
  &["_", "_Integer", "_Real", "_String", "_Rational", "_List"];
/// Builtins passed as a bare operator head (Apply, Fold, Outer, …).
const OP_FNS: &[&str] = &["Plus", "Times", "Max", "Min", "List", "Subtract"];
/// Association keys, used prefix-wise so generated keys stay distinct.
const ASSOC_KEYS: &[&str] = &["a", "b", "c", "d"];

struct Generator {
  specs: Vec<&'static FnSpec>,
  max_depth: u32,
}

impl Generator {
  /// Build a generator using only functions `functions.csv` marks ✅.
  fn new(implemented: &HashSet<String>, max_depth: u32) -> Self {
    let (specs, dropped): (Vec<_>, Vec<_>) = FN_SPECS
      .iter()
      .partition(|spec| implemented.contains(spec.name));
    for spec in dropped {
      eprintln!(
        "warning: skipping {} — not marked implemented in functions.csv",
        spec.name
      );
    }
    Generator { specs, max_depth }
  }

  fn gen_case(&self, rng: &mut Rng) -> Expr {
    // Mostly spec-driven calls; occasionally bare data or a bounded
    // Part/Take (which need index/list coupling a static spec can't say).
    // The special cases stay rare: they are three shapes competing with
    // several hundred table entries for the same budget.
    match rng.below(20) {
      0 => self.gen_part(rng),
      1 => self.gen_take(rng),
      2 => self.gen_num(rng, self.max_depth),
      _ => self.gen_spec_call(rng, self.max_depth),
    }
  }

  fn gen_spec_call(&self, rng: &mut Rng, depth: u32) -> Expr {
    let spec = *rng.pick(&self.specs);
    let args = spec
      .args
      .iter()
      .map(|arg| self.gen_arg(rng, *arg, depth.saturating_sub(1)))
      .collect();
    Expr::Call(spec.name, args)
  }

  fn gen_arg(&self, rng: &mut Rng, arg: Arg, depth: u32) -> Expr {
    match arg {
      Arg::Any => self.gen_any(rng, depth),
      Arg::Num => self.gen_num(rng, depth),
      Arg::Bool => self.gen_bool(rng, depth),
      Arg::Str => gen_str(rng),
      Arg::ListNum => self.gen_list_num(rng, depth, 0),
      Arg::ListNum1 => self.gen_list_num(rng, depth, 1),
      Arg::ListAny => self.gen_list_any(rng, depth),
      Arg::Nat(max) => Expr::Int(rng.range(0, max)),
      Arg::IntIn(lo, hi) => Expr::Int(rng.range(lo, hi)),
      Arg::Poly => gen_poly(rng),
      Arg::PolyProd => gen_poly_prod(rng),
      Arg::RatFn => Expr::Call("Divide", vec![gen_poly(rng), gen_poly(rng)]),
      Arg::PolyXY => gen_poly_xy(rng),
      Arg::RadNum => gen_rad_num(rng, 2),
      // The explicit deref is load-bearing: without it inference resolves
      // the slice element type as `str` and the call fails to compile.
      #[allow(clippy::explicit_auto_deref)]
      Arg::PureFn => Expr::Raw(*rng.pick(PURE_FNS)),
      #[allow(clippy::explicit_auto_deref)]
      Arg::PredFn => Expr::Raw(*rng.pick(PRED_FNS)),
      Arg::VarX => Expr::Sym("x"),
      Arg::VarY => Expr::Sym("y"),

      Arg::ExactNum => gen_exact_num(rng),
      Arg::RatNum => Expr::Rational(rng.range(-60, 60), rng.range(1, 20)),
      Arg::TrigNum => gen_trig_arg(rng),
      Arg::UnitNum => gen_unit_num(rng),
      Arg::PosNum => gen_pos_num(rng),
      Arg::ComplexNum => gen_complex(rng),
      Arg::OddNat(max) => Expr::Int(2 * rng.range(0, (max - 1) / 2) + 1),
      Arg::Frac(max_k, den) => Expr::Call(
        "Divide",
        vec![Expr::Int(rng.range(0, max_k)), Expr::Int(den)],
      ),

      Arg::ListInt => gen_int_list(rng, 0, 4, -5, 5),
      Arg::ListInt1 => gen_int_list(rng, 1, 4, -5, 5),
      Arg::ListInt2 => gen_int_list(rng, 2, 5, -5, 5),
      Arg::ListPosInt1 => gen_int_list(rng, 1, 4, 1, 9),
      Arg::ListDigit => gen_int_list(rng, 1, 5, 0, 9),
      Arg::ListCharCode => gen_int_list(rng, 1, 4, 65, 122),
      Arg::ListStr => {
        Expr::List((0..rng.below(4)).map(|_| gen_str(rng)).collect::<Vec<_>>())
      }
      Arg::StrPair => {
        Expr::List(vec![Expr::Str("a".into()), Expr::Str("b".into())])
      }
      Arg::Matrix => {
        let n = 2 + rng.below(2);
        gen_matrix(rng, n)
      }
      Arg::Matrix2 => gen_matrix(rng, 2),
      Arg::Vec2 => gen_int_list(rng, 2, 2, -6, 6),
      Arg::Vec3 => gen_int_list(rng, 3, 3, -6, 6),
      Arg::VecPair => Expr::List(vec![
        gen_int_list(rng, 3, 3, -6, 6),
        gen_int_list(rng, 3, 3, -6, 6),
      ]),
      Arg::DimPair => {
        Expr::List(vec![Expr::Int(rng.range(1, 3)), Expr::Int(rng.range(1, 3))])
      }
      Arg::Assoc => gen_assoc(rng),

      #[allow(clippy::explicit_auto_deref)]
      Arg::TypePat => Expr::Raw(*rng.pick(TYPE_PATS)),
      Arg::IntRule => Expr::Call(
        "Rule",
        vec![Expr::Int(rng.range(-5, 5)), Expr::Int(rng.range(-5, 5))],
      ),
      Arg::StrRule => {
        Expr::Call("Rule", vec![gen_char_str(rng), gen_char_str(rng)])
      }
      Arg::VarRule => {
        Expr::Call("Rule", vec![Expr::Sym("x"), gen_exact_num(rng)])
      }
      #[allow(clippy::explicit_auto_deref)]
      Arg::OpFn => Expr::Sym(*rng.pick(OP_FNS)),
      Arg::CharStr => gen_char_str(rng),

      Arg::TrigExpr => gen_trig_expr(rng, 2),
      Arg::ExpExpr => gen_exp_expr(rng),
      Arg::LowPoly => gen_low_poly(rng),
      Arg::PolyEq => Expr::Call("Equal", vec![gen_low_poly(rng), Expr::Int(0)]),
      Arg::SeriesSpec => Expr::List(vec![
        Expr::Sym("x"),
        Expr::Int(0),
        Expr::Int(rng.range(1, 4)),
      ]),
      Arg::Iter => Expr::List(vec![
        Expr::Sym("i"),
        Expr::Int(1),
        Expr::Int(rng.range(1, 6)),
      ]),
      Arg::IterBody => gen_iter_body(rng),
    }
  }

  fn gen_any(&self, rng: &mut Rng, depth: u32) -> Expr {
    match rng.below(9) {
      0 => gen_str(rng),
      1 => self.gen_bool(rng, depth),
      2 => self.gen_list_num(rng, depth, 0),
      // The generic heads (Head, Depth, ToString, SameQ, …) are the ones
      // most likely to mishandle an unusual argument, so feed them
      // associations, matrices and complex numbers too.
      3 => gen_assoc(rng),
      4 => gen_matrix(rng, 2),
      5 => gen_complex(rng),
      _ if depth > 0 && rng.chance(1, 3) => self.gen_spec_call(rng, depth),
      _ => self.gen_num(rng, depth),
    }
  }

  fn gen_num(&self, rng: &mut Rng, depth: u32) -> Expr {
    if depth == 0 || rng.chance(1, 2) {
      return gen_num_leaf(rng);
    }
    let head = *rng.pick(&["Plus", "Times", "Subtract", "Divide"]);
    let a = self.gen_num(rng, depth - 1);
    let b = self.gen_num(rng, depth - 1);
    Expr::Call(head, vec![a, b])
  }

  fn gen_bool(&self, rng: &mut Rng, depth: u32) -> Expr {
    if depth == 0 || rng.chance(1, 3) {
      return Expr::Sym(if rng.chance(1, 2) { "True" } else { "False" });
    }
    let head = *rng.pick(&["Less", "Greater", "Equal", "LessEqual"]);
    let a = self.gen_num(rng, depth - 1);
    let b = self.gen_num(rng, depth - 1);
    Expr::Call(head, vec![a, b])
  }

  fn gen_list_num(&self, rng: &mut Rng, depth: u32, min_len: u64) -> Expr {
    let len = min_len + rng.below(5);
    let xs = (0..len)
      .map(|_| {
        if depth > 0 && rng.chance(1, 4) {
          self.gen_num(rng, depth - 1)
        } else {
          gen_num_leaf(rng)
        }
      })
      .collect();
    Expr::List(xs)
  }

  fn gen_list_any(&self, rng: &mut Rng, depth: u32) -> Expr {
    let len = rng.below(5);
    let xs = (0..len)
      .map(|_| match rng.below(5) {
        0 => gen_str(rng),
        1 if depth > 0 => self.gen_list_num(rng, depth - 1, 0),
        _ => gen_num_leaf(rng),
      })
      .collect();
    Expr::List(xs)
  }

  /// Part with an in-bounds index (out-of-bounds behavior is exercised
  /// rarely enough by shrinking; a static spec would flood messages).
  fn gen_part(&self, rng: &mut Rng) -> Expr {
    let list = self.gen_list_num(rng, 1, 1);
    let len = match &list {
      Expr::List(xs) => xs.len() as i64,
      _ => unreachable!(),
    };
    let idx = rng.range(1, len);
    Expr::Call("Part", vec![list, Expr::Int(idx)])
  }

  fn gen_take(&self, rng: &mut Rng) -> Expr {
    let list = self.gen_list_num(rng, 1, 1);
    let len = match &list {
      Expr::List(xs) => xs.len() as i64,
      _ => unreachable!(),
    };
    let n = rng.range(1, len);
    let n = if rng.chance(1, 3) { -n } else { n };
    Expr::Call("Take", vec![list, Expr::Int(n)])
  }
}

fn gen_num_leaf(rng: &mut Rng) -> Expr {
  match rng.below(10) {
    0..=5 => Expr::Int(rng.range(-100, 100)),
    6 | 7 => Expr::Rational(rng.range(-20, 20), rng.range(1, 9)),
    8 => Expr::RealTenths(rng.range(-200, 200)),
    _ => Expr::Sym("Pi"),
  }
}

fn gen_str(rng: &mut Rng) -> Expr {
  let len = rng.below(7);
  let s = (0..len)
    .map(|_| STR_CHARS[rng.below(STR_CHARS.len() as u64) as usize] as char)
    .collect();
  Expr::Str(s)
}

/// One-character string. String searches against a random 0..6-character
/// haystack would almost never hit; a single character does, so
/// StringCases/StringReplace/StringPosition exercise their match paths
/// instead of always returning the empty answer.
fn gen_char_str(rng: &mut Rng) -> Expr {
  let c = STR_CHARS[rng.below(STR_CHARS.len() as u64) as usize] as char;
  Expr::Str(c.to_string())
}

/// Exact numeric leaf — integer, rational or a symbolic constant. Machine
/// reals are deliberately excluded: the special-function specs are there
/// to compare closed forms, and a float argument collapses every one of
/// them to a last-digit comparison instead.
fn gen_exact_num(rng: &mut Rng) -> Expr {
  match rng.below(8) {
    0..=4 => Expr::Int(rng.range(-12, 12)),
    5 | 6 => Expr::Rational(rng.range(-8, 8), rng.range(1, 6)),
    _ => Expr::Sym(if rng.chance(1, 2) { "Pi" } else { "E" }),
  }
}

/// Argument for the trigonometric family, weighted towards rational
/// multiples of Pi — the values with closed forms, and therefore the ones
/// where a CAS can disagree with another CAS.
fn gen_trig_arg(rng: &mut Rng) -> Expr {
  match rng.below(10) {
    0..=4 => {
      let num = rng.range(-4, 4);
      let den = *rng.pick(&[1i64, 2, 3, 4, 6, 8, 12]);
      let scaled = if num == 1 {
        Expr::Sym("Pi")
      } else {
        Expr::Call("Times", vec![Expr::Int(num), Expr::Sym("Pi")])
      };
      if den == 1 {
        scaled
      } else {
        Expr::Call("Divide", vec![scaled, Expr::Int(den)])
      }
    }
    5..=7 => Expr::Int(rng.range(-6, 6)),
    8 => Expr::Rational(rng.range(-6, 6), rng.range(1, 5)),
    _ => Expr::RealTenths(rng.range(-30, 30)),
  }
}

/// Numeric value in [-1, 1], the domain of ArcSin, ArcTanh, InverseErf
/// and the Quantile-style options.
fn gen_unit_num(rng: &mut Rng) -> Expr {
  match rng.below(6) {
    0 | 1 => Expr::Int(rng.range(-1, 1)),
    2..=4 => {
      let den = rng.range(2, 6);
      Expr::Rational(rng.range(-den, den), den)
    }
    _ => Expr::RealTenths(rng.range(-10, 10)),
  }
}

/// Strictly positive numeric value (Log, Gamma, GeometricMean, …).
fn gen_pos_num(rng: &mut Rng) -> Expr {
  match rng.below(8) {
    0..=3 => Expr::Int(rng.range(1, 12)),
    4 | 5 => Expr::Rational(rng.range(1, 10), rng.range(1, 6)),
    6 => Expr::RealTenths(rng.range(1, 60)),
    _ => Expr::Sym("Pi"),
  }
}

/// Complex number `a + b I`, printed in full form.
fn gen_complex(rng: &mut Rng) -> Expr {
  Expr::Call(
    "Plus",
    vec![
      Expr::Int(rng.range(-6, 6)),
      Expr::Call("Times", vec![Expr::Int(rng.range(-6, 6)), Expr::Sym("I")]),
    ],
  )
}

/// List of `min_len..=max_len` integers drawn from `lo..=hi`.
fn gen_int_list(
  rng: &mut Rng,
  min_len: u64,
  max_len: u64,
  lo: i64,
  hi: i64,
) -> Expr {
  let len = min_len + rng.below(max_len - min_len + 1);
  Expr::List((0..len).map(|_| Expr::Int(rng.range(lo, hi))).collect())
}

/// Square `n × n` integer matrix.
fn gen_matrix(rng: &mut Rng, n: u64) -> Expr {
  Expr::List(
    (0..n)
      .map(|_| {
        Expr::List((0..n).map(|_| Expr::Int(rng.range(-4, 4))).collect())
      })
      .collect(),
  )
}

/// Association with distinct short-string keys and integer values.
fn gen_assoc(rng: &mut Rng) -> Expr {
  let len = 1 + rng.below(ASSOC_KEYS.len() as u64);
  Expr::Call(
    "Association",
    (0..len)
      .map(|i| {
        Expr::Call(
          "Rule",
          vec![
            Expr::Str(ASSOC_KEYS[i as usize].to_string()),
            Expr::Int(rng.range(-9, 9)),
          ],
        )
      })
      .collect(),
  )
}

/// Expression over Sin/Cos/Tan of `x` (and sometimes `y`) — the input
/// shape the trig rewriters (TrigExpand, TrigReduce, TrigToExp) are for.
fn gen_trig_expr(rng: &mut Rng, depth: u32) -> Expr {
  if depth == 0 || rng.chance(1, 2) {
    let head = *rng.pick(&["Sin", "Cos", "Tan"]);
    let arg = match rng.below(4) {
      0 => Expr::Sym("x"),
      1 => {
        Expr::Call("Times", vec![Expr::Int(rng.range(2, 3)), Expr::Sym("x")])
      }
      2 => Expr::Call("Plus", vec![Expr::Sym("x"), Expr::Sym("y")]),
      _ => Expr::Call("Times", vec![Expr::Sym("x"), Expr::Sym("y")]),
    };
    let call = Expr::Call(head, vec![arg]);
    return if rng.chance(1, 3) {
      Expr::Call("Power", vec![call, Expr::Int(rng.range(2, 3))])
    } else {
      call
    };
  }
  let a = gen_trig_expr(rng, depth - 1);
  let b = gen_trig_expr(rng, depth - 1);
  #[allow(clippy::explicit_auto_deref)]
  let head = *rng.pick(&["Plus", "Times", "Subtract"]);
  Expr::Call(head, vec![a, b])
}

/// Expression over Exp/Log/Sqrt of `x` and `y` — what PowerExpand,
/// ExpToTrig and the log/radical simplifiers act on.
fn gen_exp_expr(rng: &mut Rng) -> Expr {
  let x = Expr::Sym("x");
  let y = Expr::Sym("y");
  match rng.below(6) {
    0 => Expr::Call(
      "Exp",
      vec![Expr::Call("Times", vec![Expr::Int(rng.range(1, 3)), x])],
    ),
    1 => Expr::Call("Log", vec![Expr::Call("Times", vec![x, y])]),
    2 => Expr::Call(
      "Log",
      vec![Expr::Call("Power", vec![x, Expr::Int(rng.range(2, 4))])],
    ),
    3 => Expr::Call("Sqrt", vec![Expr::Call("Times", vec![x, y])]),
    4 => Expr::Call("Exp", vec![Expr::Call("Plus", vec![x, y])]),
    _ => Expr::Call("Sqrt", vec![Expr::Call("Power", vec![x, Expr::Int(2)])]),
  }
}

/// Polynomial in x of degree 1 or 2 with a non-zero leading coefficient —
/// Solve, Resultant and PolynomialQuotient all want a known degree.
fn gen_low_poly(rng: &mut Rng) -> Expr {
  let degree = rng.range(1, 2);
  let mut terms = vec![Expr::Int(rng.range(-6, 6))];
  for k in 1..=degree {
    let coeff = if k == degree {
      let c = rng.range(1, 4);
      if rng.chance(1, 2) { -c } else { c }
    } else {
      rng.range(-4, 4)
    };
    let power = if k == 1 {
      Expr::Sym("x")
    } else {
      Expr::Call("Power", vec![Expr::Sym("x"), Expr::Int(k)])
    };
    terms.push(Expr::Call("Times", vec![Expr::Int(coeff), power]));
  }
  Expr::Call("Plus", terms)
}

/// Term in the iteration variable `i`, for Sum/Product/Table.
fn gen_iter_body(rng: &mut Rng) -> Expr {
  let i = Expr::Sym("i");
  match rng.below(6) {
    0 => i,
    1 => Expr::Call("Power", vec![i, Expr::Int(rng.range(2, 3))]),
    2 => Expr::Call("Times", vec![Expr::Int(rng.range(-4, 4)), i]),
    3 => Expr::Call("Divide", vec![Expr::Int(1), i]),
    4 => Expr::Call("Plus", vec![i, Expr::Int(rng.range(-4, 4))]),
    _ => Expr::Call("Power", vec![Expr::Int(rng.range(2, 3)), i]),
  }
}

/// Polynomial in x: sum of c * x^k terms, small everything.
fn gen_poly(rng: &mut Rng) -> Expr {
  let terms = 1 + rng.below(3);
  let xs = (0..=terms)
    .map(|k| {
      let c = Expr::Int(rng.range(-5, 5));
      match k {
        0 => c,
        1 => Expr::Call("Times", vec![c, Expr::Sym("x")]),
        _ => Expr::Call(
          "Times",
          vec![
            c,
            Expr::Call("Power", vec![Expr::Sym("x"), Expr::Int(k as i64)]),
          ],
        ),
      }
    })
    .collect();
  Expr::Call("Plus", xs)
}

fn gen_poly_prod(rng: &mut Rng) -> Expr {
  if rng.chance(1, 3) {
    Expr::Call("Power", vec![gen_poly(rng), Expr::Int(rng.range(2, 3))])
  } else {
    Expr::Call("Times", vec![gen_poly(rng), gen_poly(rng)])
  }
}

/// Small polynomial in x and y: sum of c * x^i * y^j monomials.
fn gen_poly_xy(rng: &mut Rng) -> Expr {
  let terms = 2 + rng.below(3);
  let xs = (0..terms)
    .map(|_| {
      let c = Expr::Int(rng.range(-5, 5));
      let i = rng.range(0, 2);
      let j = rng.range(0, 2);
      let mut factors = vec![c];
      if i == 1 {
        factors.push(Expr::Sym("x"));
      } else if i == 2 {
        factors.push(Expr::Call("Power", vec![Expr::Sym("x"), Expr::Int(2)]));
      }
      if j == 1 {
        factors.push(Expr::Sym("y"));
      } else if j == 2 {
        factors.push(Expr::Call("Power", vec![Expr::Sym("y"), Expr::Int(2)]));
      }
      if factors.len() == 1 {
        factors.pop().unwrap()
      } else {
        Expr::Call("Times", factors)
      }
    })
    .collect();
  Expr::Call("Plus", xs)
}

/// Exact arithmetic over small square roots, e.g. Sqrt[8] + 3*Sqrt[2],
/// Sqrt[12]*Sqrt[3], Sqrt[2]/Sqrt[8] — exercises radical canonicalization.
fn gen_rad_num(rng: &mut Rng, depth: u32) -> Expr {
  if depth == 0 || rng.chance(1, 3) {
    let radicand = Expr::Int(rng.range(2, 30));
    return if rng.chance(1, 3) {
      Expr::Call(
        "Times",
        vec![
          Expr::Int(rng.range(-4, 4)),
          Expr::Call("Sqrt", vec![radicand]),
        ],
      )
    } else {
      Expr::Call("Sqrt", vec![radicand])
    };
  }
  let a = gen_rad_num(rng, depth - 1);
  let b = gen_rad_num(rng, depth - 1);
  match rng.below(4) {
    0 => Expr::Call("Plus", vec![a, b]),
    1 => Expr::Call("Subtract", vec![a, b]),
    2 => Expr::Call("Times", vec![a, b]),
    _ => Expr::Call("Divide", vec![a, b]),
  }
}

// ---------------------------------------------------------------------------
// Process runners
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RunOutput {
  stdout: String,
  stderr: String,
  exit_code: Option<i32>,
  timed_out: bool,
}

/// Run a command with piped stdio, killing it after `timeout`.
fn run_with_timeout(
  mut cmd: Command,
  timeout: Duration,
) -> std::io::Result<RunOutput> {
  cmd
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
  let mut child = cmd.spawn()?;

  let mut stdout_pipe = child.stdout.take().expect("stdout piped");
  let mut stderr_pipe = child.stderr.take().expect("stderr piped");
  let stdout_thread = std::thread::spawn(move || {
    let mut buf = Vec::new();
    let _ = stdout_pipe.read_to_end(&mut buf);
    buf
  });
  let stderr_thread = std::thread::spawn(move || {
    let mut buf = Vec::new();
    let _ = stderr_pipe.read_to_end(&mut buf);
    buf
  });

  let start = Instant::now();
  let (exit_code, timed_out) = loop {
    if let Some(status) = child.try_wait()? {
      break (status.code(), false);
    }
    if start.elapsed() > timeout {
      let _ = child.kill();
      let _ = child.wait();
      break (None, true);
    }
    std::thread::sleep(Duration::from_millis(10));
  };

  let stdout =
    String::from_utf8_lossy(&stdout_thread.join().unwrap_or_default())
      .into_owned();
  let stderr =
    String::from_utf8_lossy(&stderr_thread.join().unwrap_or_default())
      .into_owned();
  Ok(RunOutput {
    stdout,
    stderr,
    exit_code,
    timed_out,
  })
}

enum Oracle {
  Wolframscript {
    path: String,
  },
  /// Self-check mode: woxi plays its own oracle. Expect zero divergences;
  /// useful for validating the harness plumbing without a Wolfram install.
  Woxi {
    path: PathBuf,
  },
}

/// Make a snippet safe to pass as a CLI argument.
///
/// `wolframscript -code -2` exits 255 with "-code called with no argument":
/// its argument parser takes anything starting with `-` for another flag.
/// The shrinker reaches such snippets routinely — hoisting an argument out
/// of a call leaves bare negative literals — and the resulting transport
/// failure looks exactly like a divergence, so shrinking would then run off
/// towards a reproducer that is not one. Parenthesising changes nothing
/// semantically, and both tools get the identical string.
fn cli_safe(code: &str) -> String {
  if code.starts_with('-') {
    format!("({code})")
  } else {
    code.to_string()
  }
}

impl Oracle {
  fn describe(&self) -> String {
    match self {
      Oracle::Wolframscript { path } => format!("wolframscript ({path})"),
      Oracle::Woxi { path } => format!("woxi self-check ({})", path.display()),
    }
  }

  fn eval(&self, code: &str, timeout: Duration) -> std::io::Result<RunOutput> {
    match self {
      Oracle::Wolframscript { path } => {
        let mut cmd = Command::new(path);
        cmd.arg("-code").arg(cli_safe(code));
        run_with_timeout(cmd, timeout)
      }
      Oracle::Woxi { path } => {
        let mut cmd = Command::new(path);
        cmd.arg("eval").arg(cli_safe(code));
        run_with_timeout(cmd, timeout)
      }
    }
  }
}

fn woxi_eval(
  woxi: &Path,
  code: &str,
  timeout: Duration,
) -> std::io::Result<RunOutput> {
  let mut cmd = Command::new(woxi);
  cmd.arg("eval").arg(cli_safe(code));
  run_with_timeout(cmd, timeout)
}

// ---------------------------------------------------------------------------
// Output comparison
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum CompareMode {
  /// Sorted line-bag over stdout + stderr: insensitive to which stream a
  /// message landed on and to capture interleaving (the default, mirrors
  /// scrut's `output_stream: combined` with reordering tolerance).
  Bag,
  /// Exact stdout + stderr concatenation.
  Combined,
  /// Exact stdout only.
  Stdout,
}

fn canonicalize(out: &RunOutput, mode: CompareMode) -> String {
  match mode {
    CompareMode::Stdout => out.stdout.trim_end().to_string(),
    CompareMode::Combined => {
      format!("{}\n{}", out.stdout.trim_end(), out.stderr.trim_end())
    }
    CompareMode::Bag => {
      let mut lines: Vec<&str> = out
        .stdout
        .lines()
        .chain(out.stderr.lines())
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();
      lines.sort_unstable();
      lines.join("\n")
    }
  }
}

fn outputs_match(a: &RunOutput, b: &RunOutput, mode: CompareMode) -> bool {
  canonicalize(a, mode) == canonicalize(b, mode)
}

// ---------------------------------------------------------------------------
// Batch construction and segmentation
// ---------------------------------------------------------------------------

fn marker(i: usize) -> String {
  format!("<<<WOXIDIFF:{i}>>>")
}
const END_MARKER: &str = "<<<WOXIDIFF:END>>>";

/// One program evaluating every case, each preceded by a marker print.
/// InputForm keeps `"5"` and `5` distinguishable in the printed stream.
fn build_batch_program(cases: &[String]) -> String {
  let mut prog = String::new();
  for (i, code) in cases.iter().enumerate() {
    prog.push_str(&format!(
      "Print[\"{}\"]; Print[InputForm[{code}]]; ",
      marker(i)
    ));
  }
  prog.push_str(&format!("Print[\"{END_MARKER}\"];"));
  prog
}

/// Split a batch stdout into per-case segments; a case missing its marker
/// (crash or hang upstream) gets None and is re-checked individually.
fn split_segments(stdout: &str, n_cases: usize) -> Vec<Option<String>> {
  let mut segments = vec![None; n_cases];
  let mut current: Option<usize> = None;
  let mut acc = String::new();
  for line in stdout.lines() {
    let trimmed = line.trim_end();
    let is_end = trimmed == END_MARKER;
    let next_case = trimmed
      .strip_prefix("<<<WOXIDIFF:")
      .and_then(|rest| rest.strip_suffix(">>>"))
      .and_then(|num| num.parse::<usize>().ok());
    if is_end || next_case.is_some() {
      if let Some(i) = current
        && i < n_cases
      {
        segments[i] = Some(std::mem::take(&mut acc));
      }
      acc.clear();
      current = if is_end { None } else { next_case };
      continue;
    }
    if current.is_some() {
      acc.push_str(trimmed);
      acc.push('\n');
    }
  }
  // A case still open when output ended (killed mid-batch) stays None so
  // it is re-checked individually.
  segments
}

// ---------------------------------------------------------------------------
// Shrinking
// ---------------------------------------------------------------------------

/// Strictly-smaller replacement candidates (by `Expr::size`), covering the
/// whole tree one step at a time: hoist children, simplify atoms, drop
/// list/call elements, recurse into children.
fn shrink_candidates(expr: &Expr) -> Vec<Expr> {
  let mut out = Vec::new();
  match expr {
    Expr::Int(n) => {
      if *n != 0 {
        out.push(Expr::Int(0));
      }
      // Int(1) has size 2, so it only strictly shrinks |n| > 1.
      if n.unsigned_abs() > 1 && *n != 1 {
        out.push(Expr::Int(1));
      }
      if n.unsigned_abs() > 2 {
        out.push(Expr::Int(n / 2));
      }
    }
    Expr::Rational(..) | Expr::RealTenths(..) => {
      out.push(Expr::Int(0));
      // Int(1) size is 2, so it only strictly shrinks larger literals
      // (RealTenths(0) is also size 2).
      if expr.size() > 2 {
        out.push(Expr::Int(1));
      }
    }
    Expr::Str(s) => {
      if !s.is_empty() {
        let mut shorter = s.clone();
        shorter.pop();
        out.push(Expr::Str(shorter));
      }
    }
    Expr::Sym(_) | Expr::Raw(_) => {}
    Expr::List(xs) => {
      for i in 0..xs.len() {
        let mut fewer = xs.clone();
        fewer.remove(i);
        out.push(Expr::List(fewer));
      }
      for (i, x) in xs.iter().enumerate() {
        for cand in shrink_candidates(x) {
          let mut ys = xs.clone();
          ys[i] = cand;
          out.push(Expr::List(ys));
        }
      }
    }
    Expr::Call(head, args) => {
      // Hoist each argument over the call (Foo[x] → x).
      out.extend(args.iter().cloned());
      for (i, a) in args.iter().enumerate() {
        for cand in shrink_candidates(a) {
          let mut bs = args.clone();
          bs[i] = cand;
          out.push(Expr::Call(head, bs));
        }
      }
    }
  }
  debug_assert!(out.iter().all(|c| c.size() < expr.size()));
  out
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

struct Finding {
  case_seed: u64,
  original: String,
  shrunk: Option<String>,
  woxi: RunOutput,
  oracle: RunOutput,
}

struct Harness {
  woxi: PathBuf,
  oracle: Oracle,
  timeout: Duration,
  mode: CompareMode,
}

impl Harness {
  /// Compare one code snippet through both tools. Returns Some((woxi,
  /// oracle)) when they diverge, None when they agree or the oracle is
  /// unusable (timeout/transport failure — those cases are skipped, never
  /// reported as findings).
  fn diverges(&self, code: &str) -> Option<(RunOutput, RunOutput)> {
    let woxi_out = woxi_eval(&self.woxi, code, self.timeout).ok()?;
    let oracle_out = self.oracle.eval(code, self.timeout).ok()?;
    if oracle_out.timed_out {
      eprintln!("  oracle timeout — skipping: {code}");
      return None;
    }
    // A woxi crash (signal/kill) or hang is always a finding.
    let woxi_broken = woxi_out.timed_out || woxi_out.exit_code != Some(0);
    if woxi_broken || !outputs_match(&woxi_out, &oracle_out, self.mode) {
      Some((woxi_out, oracle_out))
    } else {
      None
    }
  }

  fn shrink(&self, expr: &Expr, mut budget: u32) -> Option<Expr> {
    let mut current = expr.clone();
    let mut improved = false;
    'outer: loop {
      for cand in shrink_candidates(&current) {
        if budget == 0 {
          break 'outer;
        }
        budget -= 1;
        if self.diverges(&cand.to_code()).is_some() {
          current = cand;
          improved = true;
          continue 'outer;
        }
      }
      break;
    }
    improved.then_some(current)
  }
}

fn load_implemented(csv_path: &std::path::Path) -> HashSet<String> {
  let Ok(content) = std::fs::read_to_string(csv_path) else {
    eprintln!(
      "warning: cannot read {} — using the full curated table",
      csv_path.display()
    );
    return FN_SPECS.iter().map(|s| s.name.to_string()).collect();
  };
  content
    .lines()
    .skip(1)
    .filter_map(|line| {
      // Only the first column can be split naively (descriptions contain
      // commas); the status column always follows the description, so
      // instead check for the ✅ cell delimiter-wrapped anywhere.
      let name = line.split(',').next()?;
      line.contains(",✅,").then(|| name.to_string())
    })
    .collect()
}

fn resolve_woxi(cli_woxi: &Option<PathBuf>) -> Result<PathBuf, String> {
  if let Some(path) = cli_woxi {
    return Ok(path.clone());
  }
  let sibling = std::env::current_exe().ok().and_then(|exe| {
    Some(
      exe
        .parent()?
        .join(format!("woxi{}", std::env::consts::EXE_SUFFIX)),
    )
  });
  match sibling {
    Some(path) if path.exists() => Ok(path),
    _ => Err(
      "woxi binary not found next to woxi-diff-fuzz — run `cargo build` \
       first or pass --woxi <path>"
        .to_string(),
    ),
  }
}

fn resolve_oracle(cli: &Cli, woxi: &Path) -> Result<Oracle, String> {
  let probe_wolframscript = || {
    Command::new(&cli.wolframscript)
      .arg("-h")
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .spawn()
      .map(|mut child| {
        let _ = child.wait();
      })
      .is_ok()
  };
  match cli.oracle.as_str() {
    "wolframscript" => Ok(Oracle::Wolframscript {
      path: cli.wolframscript.clone(),
    }),
    "woxi" => Ok(Oracle::Woxi {
      path: woxi.to_path_buf(),
    }),
    "auto" => {
      if probe_wolframscript() {
        Ok(Oracle::Wolframscript {
          path: cli.wolframscript.clone(),
        })
      } else {
        Err(
          "no oracle reachable: wolframscript is not on PATH — pass \
           --oracle/--wolframscript (or --oracle woxi for a plumbing \
           self-check)"
            .to_string(),
        )
      }
    }
    other => Err(format!(
      "unknown --oracle '{other}' (expected auto|wolframscript|woxi)"
    )),
  }
}

fn main() {
  let cli = Cli::parse();

  let mode = match cli.compare.as_str() {
    "bag" => CompareMode::Bag,
    "combined" => CompareMode::Combined,
    "stdout" => CompareMode::Stdout,
    other => {
      eprintln!("unknown --compare '{other}' (expected bag|combined|stdout)");
      std::process::exit(2);
    }
  };

  let csv_path =
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("functions.csv");
  let implemented = load_implemented(&csv_path);
  let generator = Generator::new(&implemented, cli.max_depth);
  if generator.specs.is_empty() {
    eprintln!("error: no usable functions left after the functions.csv check");
    std::process::exit(2);
  }

  let master_seed = cli.seed.unwrap_or_else(|| {
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos() as u64)
      .unwrap_or(0x00DE_FA17)
  });

  // Case seeds derive from the master seed by index, so any single case
  // can be replayed with --seed <master> regardless of batch layout.
  let case_seed = |i: usize| {
    let mut rng = Rng::new(master_seed ^ (i as u64).wrapping_mul(0xA5A5_A5A5));
    rng.next_u64()
  };
  let gen_expr_for = |i: usize| {
    let mut rng = Rng::new(case_seed(i));
    generator.gen_case(&mut rng)
  };

  if cli.print_cases {
    for i in 0..cli.cases {
      println!("{}", gen_expr_for(i).to_code());
    }
    return;
  }

  let woxi = match resolve_woxi(&cli.woxi) {
    Ok(p) => p,
    Err(e) => {
      eprintln!("error: {e}");
      std::process::exit(2);
    }
  };
  let oracle = match resolve_oracle(&cli, &woxi) {
    Ok(o) => o,
    Err(e) => {
      eprintln!("error: {e}");
      std::process::exit(2);
    }
  };

  let harness = Harness {
    woxi,
    oracle,
    timeout: Duration::from_secs(cli.timeout),
    mode,
  };

  eprintln!(
    "woxi-diff-fuzz: {} cases, batch size {}, seed {master_seed}, oracle: {}",
    cli.cases,
    cli.batch_size,
    harness.oracle.describe()
  );

  let mut findings: Vec<Finding> = Vec::new();
  let mut checked = 0usize;

  let batch_size = cli.batch_size.max(1);
  for batch_start in (0..cli.cases).step_by(batch_size) {
    let indices: Vec<usize> =
      (batch_start..(batch_start + batch_size).min(cli.cases)).collect();
    let exprs: Vec<Expr> = indices.iter().map(|&i| gen_expr_for(i)).collect();
    let codes: Vec<String> = exprs.iter().map(Expr::to_code).collect();

    // Batch prefilter: identical program text through both tools, so any
    // segment difference is worth a closer look (and only those cost an
    // extra oracle round-trip each).
    let program = build_batch_program(&codes);
    let batch_timeout =
      harness.timeout + Duration::from_secs(codes.len() as u64);
    let woxi_run = woxi_eval(&harness.woxi, &program, batch_timeout);
    let oracle_run = harness.oracle.eval(&program, batch_timeout);

    let suspicious: Vec<usize> = match (&woxi_run, &oracle_run) {
      (Ok(w), Ok(o)) if !w.timed_out && !o.timed_out => {
        let w_segs = split_segments(&w.stdout, codes.len());
        let o_segs = split_segments(&o.stdout, codes.len());
        (0..codes.len())
          .filter(|&k| match (&w_segs[k], &o_segs[k]) {
            (Some(a), Some(b)) => a != b,
            _ => true, // missing segment: crash/hang inside the batch
          })
          .collect()
      }
      // Whole batch unusable (hang, crash, transport error): check
      // every case individually.
      _ => (0..codes.len()).collect(),
    };

    for &k in &suspicious {
      let bare = &codes[k];
      // Primary confirmation: the exact pairing the CLI doc tests use.
      let mut divergence = harness.diverges(bare).map(|d| (bare.clone(), d));
      if divergence.is_none() {
        // The batch saw an InputForm-level difference the bare form
        // hides (e.g. "5" vs 5) — confirm with that exact statement.
        let wrapped = format!("Print[InputForm[{bare}]]");
        divergence = harness.diverges(&wrapped).map(|d| (wrapped, d));
      }
      if let Some((repro, (woxi_out, oracle_out))) = divergence {
        eprintln!("divergence: {repro}");
        let shrunk = if cli.shrink_budget > 0 {
          harness
            .shrink(&exprs[k], cli.shrink_budget)
            .map(|e| e.to_code())
        } else {
          None
        };
        findings.push(Finding {
          case_seed: case_seed(indices[k]),
          original: repro,
          shrunk,
          woxi: woxi_out,
          oracle: oracle_out,
        });
      }
    }

    checked += codes.len();
    eprintln!(
      "checked {checked}/{} cases, {} divergence(s)",
      cli.cases,
      findings.len()
    );
  }

  // -------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------
  if findings.is_empty() {
    println!(
      "OK: {} cases, no divergences (seed {master_seed})",
      cli.cases
    );
    return;
  }

  println!(
    "\n{} divergence(s) in {} cases (seed {master_seed}):\n",
    findings.len(),
    cli.cases
  );
  for (n, finding) in findings.iter().enumerate() {
    println!(
      "=== Divergence #{} (case seed {}) ===",
      n + 1,
      finding.case_seed
    );
    println!("code:    {}", finding.original);
    if let Some(shrunk) = &finding.shrunk {
      println!("shrunk:  {shrunk}");
    }
    let describe = |out: &RunOutput| {
      let mut s = out.stdout.trim_end().to_string();
      if !out.stderr.trim().is_empty() {
        s.push_str(&format!("\n  [stderr] {}", out.stderr.trim_end()));
      }
      if out.timed_out {
        s.push_str("\n  [timed out]");
      } else if out.exit_code != Some(0) {
        s.push_str(&format!("\n  [exit code {:?}]", out.exit_code));
      }
      s
    };
    println!(
      "woxi:    {}",
      describe(&finding.woxi).replace('\n', "\n         ")
    );
    println!(
      "oracle:  {}",
      describe(&finding.oracle).replace('\n', "\n         ")
    );
    println!();
  }
  std::process::exit(1);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn test_generator() -> Generator {
    let implemented: HashSet<String> =
      FN_SPECS.iter().map(|s| s.name.to_string()).collect();
    Generator::new(&implemented, 4)
  }

  /// Every curated function must be marked implemented in functions.csv —
  /// fails when a table entry goes stale (or the CSV format changes).
  #[test]
  fn curated_table_matches_functions_csv() {
    let csv_path =
      std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("functions.csv");
    let implemented = load_implemented(&csv_path);
    assert!(
      implemented.len() > 100,
      "functions.csv parse looks broken: only {} implemented entries",
      implemented.len()
    );
    let missing: Vec<&str> = FN_SPECS
      .iter()
      .map(|s| s.name)
      .filter(|name| !implemented.contains(*name))
      .collect();
    assert!(
      missing.is_empty(),
      "curated diff-fuzz functions not implemented per functions.csv: {missing:?}"
    );
  }

  /// Same seed must generate byte-identical programs (replayability).
  #[test]
  fn generator_is_deterministic() {
    let generator = test_generator();
    for seed in 0..50u64 {
      let a = generator.gen_case(&mut Rng::new(seed)).to_code();
      let b = generator.gen_case(&mut Rng::new(seed)).to_code();
      assert_eq!(a, b, "seed {seed} generated different programs");
    }
  }

  /// Every generated case must be accepted by the woxi parser: the fuzzer
  /// tests evaluation semantics, not syntax-error handling.
  #[test]
  fn generated_cases_parse() {
    let generator = test_generator();
    for seed in 0..500u64 {
      let code = generator.gen_case(&mut Rng::new(seed)).to_code();
      assert!(
        woxi::parse(&code).is_ok(),
        "seed {seed} generated unparsable code: {code}"
      );
    }
  }

  /// `generated_cases_parse` samples the table; with several hundred
  /// entries a broken argument shape can hide between samples. Exercise
  /// every spec directly instead, several times each.
  #[test]
  fn every_spec_generates_parsable_code() {
    let generator = test_generator();
    assert_eq!(
      generator.specs.len(),
      FN_SPECS.len(),
      "test generator dropped specs"
    );
    for spec in &generator.specs {
      for seed in 0..20u64 {
        let mut rng = Rng::new(seed);
        let args = spec
          .args
          .iter()
          .map(|arg| generator.gen_arg(&mut rng, *arg, 2))
          .collect();
        let code = Expr::Call(spec.name, args).to_code();
        assert!(
          woxi::parse(&code).is_ok(),
          "{} (seed {seed}) generated unparsable code: {code}",
          spec.name
        );
      }
    }
  }

  /// Shrink candidates must strictly decrease `size`, which guarantees
  /// shrinking terminates even without a budget.
  #[test]
  fn shrink_candidates_strictly_smaller() {
    let generator = test_generator();
    for seed in 0..200u64 {
      let expr = generator.gen_case(&mut Rng::new(seed));
      for cand in shrink_candidates(&expr) {
        assert!(
          cand.size() < expr.size(),
          "candidate {} (size {}) not smaller than {} (size {})",
          cand.to_code(),
          cand.size(),
          expr.to_code(),
          expr.size()
        );
      }
    }
  }

  /// Batch programs must round-trip through the segment splitter.
  #[test]
  fn batch_segments_round_trip() {
    let codes = vec![
      "1 + 1".to_string(),
      "\"x\"".to_string(),
      "{1, 2}".to_string(),
    ];
    let program = build_batch_program(&codes);
    assert!(program.contains("<<<WOXIDIFF:0>>>"));
    assert!(program.contains(END_MARKER));

    // Simulated interpreter output for the three cases.
    let stdout = format!(
      "{}\n2\n{}\n\"x\"\n{}\n{{1, 2}}\n{END_MARKER}\nNull\n",
      marker(0),
      marker(1),
      marker(2)
    );
    let segments = split_segments(&stdout, 3);
    assert_eq!(segments[0].as_deref(), Some("2\n"));
    assert_eq!(segments[1].as_deref(), Some("\"x\"\n"));
    assert_eq!(segments[2].as_deref(), Some("{1, 2}\n"));

    // Truncated output (crash after case 0) leaves later cases None.
    let truncated = format!("{}\n2\n{}\npartial", marker(0), marker(1));
    let segments = split_segments(&truncated, 3);
    assert_eq!(segments[0].as_deref(), Some("2\n"));
    assert_eq!(
      segments[1], None,
      "unterminated segment must stay suspicious"
    );
    assert_eq!(segments[2], None);
  }

  /// A snippet starting with `-` must be parenthesised before it reaches
  /// either CLI: `wolframscript -code -2` takes the snippet for a flag and
  /// fails, which the harness would otherwise report as a divergence.
  /// Shrinking produces such snippets routinely.
  #[test]
  fn leading_minus_is_parenthesised() {
    assert_eq!(cli_safe("-2"), "(-2)");
    assert_eq!(cli_safe("-Sqrt[2]"), "(-Sqrt[2])");
    assert_eq!(cli_safe("Plus[1, 2]"), "Plus[1, 2]");
    assert_eq!(cli_safe("{-1, -2}"), "{-1, -2}");
  }

  /// The bag comparator must tolerate messages landing on different
  /// streams, but still catch genuinely different output.
  #[test]
  fn bag_compare_is_stream_insensitive() {
    let woxi_out = RunOutput {
      stdout: "Power::indet: Indeterminate expression\nIndeterminate\n".into(),
      stderr: String::new(),
      exit_code: Some(0),
      timed_out: false,
    };
    let wolfram_out = RunOutput {
      stdout: "Indeterminate\n".into(),
      stderr: "Power::indet: Indeterminate expression\n".into(),
      exit_code: Some(0),
      timed_out: false,
    };
    assert!(outputs_match(&woxi_out, &wolfram_out, CompareMode::Bag));
    assert!(!outputs_match(&woxi_out, &wolfram_out, CompareMode::Stdout));

    let different = RunOutput {
      stdout: "ComplexInfinity\n".into(),
      stderr: String::new(),
      exit_code: Some(0),
      timed_out: false,
    };
    assert!(!outputs_match(&woxi_out, &different, CompareMode::Bag));
  }
}
