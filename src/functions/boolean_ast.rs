//! AST-native boolean functions.
//!
//! These functions work directly with `Expr` AST nodes.

use std::collections::BTreeSet;

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::Expr;

/// Helper to check if an Expr is True or False
fn as_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// And[expr1, expr2, ...] - Logical AND
pub fn and_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(false) => return Ok(Expr::Identifier("False".to_string())),
      Some(true) => {} // Skip True values
      None => remaining.push(evaluated),
    }
  }
  match remaining.len() {
    0 => Ok(Expr::Identifier("True".to_string())),
    1 => Ok(remaining.into_iter().next().unwrap()),
    _ => Ok(Expr::FunctionCall {
      name: "And".to_string(),
      args: remaining.into(),
    }),
  }
}

/// Or[expr1, expr2, ...] - Logical OR
pub fn or_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => return Ok(Expr::Identifier("True".to_string())),
      Some(false) => {} // Skip False values
      None => remaining.push(evaluated),
    }
  }
  match remaining.len() {
    0 => Ok(Expr::Identifier("False".to_string())),
    1 => Ok(remaining.into_iter().next().unwrap()),
    _ => Ok(Expr::FunctionCall {
      name: "Or".to_string(),
      args: remaining.into(),
    }),
  }
}

/// Not[expr] - Logical NOT
pub fn not_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    // Return unevaluated for wrong number of arguments
    return Ok(Expr::FunctionCall {
      name: "Not".to_string(),
      args: args.to_vec().into(),
    });
  }

  let evaluated = evaluate_expr_to_expr(&args[0])?;
  match as_bool(&evaluated) {
    Some(true) => Ok(Expr::Identifier("False".to_string())),
    Some(false) => Ok(Expr::Identifier("True".to_string())),
    None => {
      // Negate a single comparison: Not[a > b] → a <= b, etc.
      if let Expr::Comparison {
        operands,
        operators,
      } = &evaluated
        && operators.len() == 1
        && let Some(neg_op) = negate_comparison_op(&operators[0])
      {
        return Ok(Expr::Comparison {
          operands: operands.clone(),
          operators: vec![neg_op],
        });
      }
      Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Not,
        operand: Box::new(evaluated),
      })
    }
  }
}

fn negate_comparison_op(
  op: &crate::syntax::ComparisonOp,
) -> Option<crate::syntax::ComparisonOp> {
  use crate::syntax::ComparisonOp;
  match op {
    ComparisonOp::Equal => Some(ComparisonOp::NotEqual),
    ComparisonOp::NotEqual => Some(ComparisonOp::Equal),
    ComparisonOp::Less => Some(ComparisonOp::GreaterEqual),
    ComparisonOp::LessEqual => Some(ComparisonOp::Greater),
    ComparisonOp::Greater => Some(ComparisonOp::LessEqual),
    ComparisonOp::GreaterEqual => Some(ComparisonOp::Less),
    // SameQ/UnsameQ aren't negated via the comparison-op channel; leave
    // them to fall through to the symbolic `Not[...]` form.
    _ => None,
  }
}

/// Xor[expr1, expr2, ...] - Logical XOR
pub fn xor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Xor[] = False (empty XOR is the identity element)
  if args.is_empty() {
    return Ok(Expr::Identifier("False".to_string()));
  }
  // Single argument: Xor[x] => x
  if args.len() == 1 {
    return evaluate_expr_to_expr(&args[0]);
  }

  let mut true_count = 0;
  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => true_count += 1,
      Some(false) => {} // Skip False
      None => remaining.push(evaluated),
    }
  }
  // If there are symbolic args, combine: known true values flip parity
  if !remaining.is_empty() {
    // If odd number of True values, add True to remaining
    if true_count % 2 == 1 {
      remaining.insert(0, Expr::Identifier("True".to_string()));
    }
    return match remaining.len() {
      1 => Ok(remaining.into_iter().next().unwrap()),
      _ => Ok(Expr::FunctionCall {
        name: "Xor".to_string(),
        args: remaining.into(),
      }),
    };
  }
  Ok(Expr::Identifier(
    if true_count % 2 == 1 { "True" } else { "False" }.to_string(),
  ))
}

pub fn xnor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Xnor[] => True (0 true values, even)
  if args.is_empty() {
    return Ok(Expr::Identifier("True".to_string()));
  }
  // Xnor is the negation of Xor (returns True when even number of args are True)
  let mut true_count = 0;
  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => true_count += 1,
      Some(false) => {} // Skip False
      None => remaining.push(evaluated),
    }
  }
  if !remaining.is_empty() {
    // Symbolic case: Xnor[...] stays unevaluated
    let mut all_args = Vec::new();
    if true_count % 2 == 1 {
      all_args.push(Expr::Identifier("True".to_string()));
    }
    all_args.extend(remaining);
    return Ok(Expr::FunctionCall {
      name: "Xnor".to_string(),
      args: all_args.into(),
    });
  }
  // All boolean: True when even number of True values
  Ok(Expr::Identifier(
    if true_count % 2 == 0 { "True" } else { "False" }.to_string(),
  ))
}

/// SameQ[expr1, expr2] - Tests whether expressions are identical
pub fn same_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // SameQ[] and SameQ[x] return True (vacuously true)
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  let first = evaluate_expr_to_expr(&args[0])?;
  let first_str = crate::syntax::expr_to_string(&first);

  for arg in args.iter().skip(1) {
    let val = evaluate_expr_to_expr(arg)?;
    let val_str = crate::syntax::expr_to_string(&val);
    if val_str != first_str && !same_q_real_bigfloat(&first, &val) {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// SameQ between two floating-point operands. Matches wolframscript's
/// precision-aware semantics: both values are rounded to the *lower* of
/// their two precisions and compared. Machine-precision `Real` is
/// treated as having ≈15.954589770191003 digits of precision (Wolfram's
/// `$MachinePrecision`).
///
/// - Two plain Reals fall back to bit-exact equality (handled by the
///   string-comparison path before this is reached, so we return false
///   here to defer).
/// - When *both* operands sit in the machine-precision band we still
///   require bit-exact f64 agreement — wolframscript treats two machine
///   reals strictly.
/// - Otherwise (at least one tagged precision below the machine band),
///   we compare with a ½-ULP tolerance at the lower precision:
///   `|a - b| ≤ ½ · 10^(⌊log10 max(|a|,|b|)⌋ - p_min + 1)`. So
///   `N[2/9, 4] === .2222` is True (both round to `0.2222` at 4 digits)
///   even though `.2222` carries the full machine precision.
pub fn same_q_real_bigfloat(a: &Expr, b: &Expr) -> bool {
  // $MachinePrecision in Wolfram, used when comparing a machine-precision
  // Real against a tagged BigFloat.
  const MACHINE_PRECISION: f64 = 15.954589770191003;
  fn as_pair(e: &Expr) -> Option<(f64, f64, bool)> {
    // (value, precision_digits, is_machine_real)
    match e {
      Expr::Real(f) => Some((*f, MACHINE_PRECISION, true)),
      Expr::BigFloat(s, p) => s.parse::<f64>().ok().map(|f| (f, *p, false)),
      _ => None,
    }
  }
  let Some((va, pa, a_machine)) = as_pair(a) else {
    return false;
  };
  let Some((vb, pb, b_machine)) = as_pair(b) else {
    return false;
  };
  let machine_band = |p: f64| -> bool { (15.0..=16.5).contains(&p) };
  // Two plain Reals — defer to string equality (handled by the caller).
  if a_machine && b_machine {
    return false;
  }
  // Both operands in the machine band (machine Real or BigFloat tagged
  // ~15.95): require bit-exact f64 equality, matching wolframscript's
  // strict treatment of machine-precision values.
  if machine_band(pa) && machine_band(pb) {
    return va.to_bits() == vb.to_bits();
  }
  // Otherwise: round to the lower precision and compare with ½-ULP
  // tolerance at that position.
  let p_min = pa.min(pb);
  if !p_min.is_finite() || p_min <= 0.0 {
    return false;
  }
  let scale = va.abs().max(vb.abs());
  if scale == 0.0 {
    return va == vb;
  }
  let exponent = scale.log10().floor();
  let tol = 0.5 * 10f64.powf(exponent - p_min + 1.0);
  (va - vb).abs() <= tol
}

/// UnsameQ[expr1, expr2] - Tests whether expressions are not identical
pub fn unsame_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // UnsameQ[] and UnsameQ[x] return True (vacuously true)
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Evaluate all arguments and get string representations
  let mut strs = Vec::with_capacity(args.len());
  for arg in args {
    let val = evaluate_expr_to_expr(arg)?;
    strs.push(crate::syntax::expr_to_string(&val));
  }

  // UnsameQ is True only if ALL pairs are different
  for i in 0..strs.len() {
    for j in (i + 1)..strs.len() {
      if strs[i] == strs[j] {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Which[test1, value1, test2, value2, ...] - Multi-way conditional
pub fn which_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || !args.len().is_multiple_of(2) {
    return Err(InterpreterError::EvaluationError(
      "Which expects an even number of arguments (test-value pairs)".into(),
    ));
  }

  for i in (0..args.len()).step_by(2) {
    let test = evaluate_expr_to_expr(&args[i])?;
    match as_bool(&test) {
      Some(true) => return evaluate_expr_to_expr(&args[i + 1]),
      Some(false) => {} // Skip this pair
      None => {
        // Non-boolean condition: return Which with remaining pairs
        let mut remaining = vec![test, args[i + 1].clone()];
        for j in ((i + 2)..args.len()).step_by(1) {
          remaining.push(args[j].clone());
        }
        return Ok(Expr::FunctionCall {
          name: "Which".to_string(),
          args: remaining.into(),
        });
      }
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// While[test, body] or While[test] - While loop
/// Single-arg form: evaluates test repeatedly (do-while pattern with side effects)
pub fn while_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "While expects 1 or 2 arguments".into(),
    ));
  }

  // Safety cap to catch runaway loops in interactive use. Wolframscript
  // has no fixed iteration limit; we set a very high one so that practical
  // loops (e.g. `While[SessionTime[] - start < 2.5, …]` which on a fast
  // machine can spin millions of times before the wall-clock condition
  // trips) aren't truncated, while a true infinite loop still terminates
  // eventually instead of hanging the host.
  const MAX_ITERATIONS: usize = 1_000_000_000;
  let mut iterations: usize = 0;

  loop {
    let test = evaluate_expr_to_expr(&args[0])?;
    match as_bool(&test) {
      Some(true) => {
        if args.len() == 2 {
          match evaluate_expr_to_expr(&args[1]) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => {
              return Ok(Expr::FunctionCall {
                name: "Return".to_string(),
                args: vec![*val].into(),
              });
            }
            Err(e) => return Err(e),
          }
        }
        iterations += 1;
        if iterations >= MAX_ITERATIONS {
          return Err(InterpreterError::EvaluationError(
            "While: maximum iterations exceeded".into(),
          ));
        }
      }
      Some(false) => break,
      None => {
        return Err(InterpreterError::EvaluationError(
          "While: test must evaluate to True or False".into(),
        ));
      }
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// Equal[a, b] or a == b - Tests for equality
/// Returns True if all args are identical, False if all are numeric and differ,
/// or stays symbolic (unevaluated) if args contain symbols and aren't identical.
/// Compare two BigFloat digit strings truncated to `n` significant digits.
/// Strips a leading `-` sign and the decimal point, then takes the first
/// `n` non-sign digits from each side. Returns true iff those prefixes
/// are equal.
pub fn bigfloat_digits_match_to(d0: &str, d1: &str, n: usize) -> bool {
  fn sig_prefix(s: &str, n: usize) -> String {
    let s = s.strip_prefix('-').unwrap_or(s);
    let mut out = String::new();
    let mut count = 0usize;
    let mut seen_nonzero = false;
    for c in s.chars() {
      if c == '.' {
        continue;
      }
      if !seen_nonzero {
        if c == '0' {
          out.push(c);
          continue;
        }
        seen_nonzero = true;
      }
      if count == n {
        break;
      }
      out.push(c);
      count += 1;
    }
    out
  }
  // Need matching signs: both negative or both non-negative.
  if d0.starts_with('-') != d1.starts_with('-') {
    return false;
  }
  sig_prefix(d0, n) == sig_prefix(d1, n)
}

pub fn equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Equal[] and Equal[x] return True (like wolframscript)
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  use crate::functions::math_ast::try_eval_to_f64;

  let first_str = crate::syntax::expr_to_string(&args[0]);
  let mut all_identical = true;

  for arg in args.iter().skip(1) {
    let val_str = crate::syntax::expr_to_string(arg);
    if val_str != first_str {
      all_identical = false;
      break;
    }
  }

  if all_identical {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Check if all args are numeric
  let nums: Vec<Option<f64>> = args.iter().map(try_eval_to_f64).collect();
  if nums.iter().all(|n| n.is_some()) {
    // `Equal` compares machine-precision Reals up to the last ~7 bits (the
    // f64 "guard" bits wolframscript ignores). Promote to exact comparison
    // only if no operand is a Real; otherwise two values that differ by at
    // most `2^-46 · max(|a|, |b|)` are considered equal.
    //
    // When any operand is a BigFloat with low precision p, also widen the
    // tolerance to `10^-p · max(|a|, |b|)`: 3.1416 == 3.14`2 should be
    // True because the shorter operand only commits to ~2 digits.
    let any_real = args.iter().any(|a| {
      matches!(a, Expr::Real(_) | Expr::BigFloat(_, _))
        || matches!(a, Expr::UnaryOp { operand, .. }
          if matches!(operand.as_ref(), Expr::Real(_)))
    });
    let min_bigfloat_precision: Option<f64> = args
      .iter()
      .filter_map(|a| match a {
        Expr::BigFloat(_, p) => Some(*p),
        _ => None,
      })
      .reduce(f64::min);
    // When two BigFloats both carry more than ~16 digits of precision,
    // their stored digit strings can differ even though both collapse to
    // the same f64 mantissa. Compare the digit strings truncated to the
    // *shared* precision so high-precision near-equal literals (e.g.
    // `0.7390…642…0` vs `0.7390…641…0`, both 26 digits) are reported
    // correctly while values that agree up to the shared precision (e.g.
    // `N[E, 100]` vs `N[E, 150]`) remain equal.
    let first = nums[0].unwrap();
    for (i, n) in nums.iter().enumerate().skip(1) {
      let v = n.unwrap();
      if let (Expr::BigFloat(d0, p0), Expr::BigFloat(d1, p1)) =
        (&args[0], &args[i])
        && p0.min(*p1) > 16.0
      {
        // Compare to (shared_precision - 1) digits so a 1-ULP difference
        // at the last shared digit is treated as "equal within
        // tolerance" — matches Wolfram, where `0.7390…642 == 0.7390…641`
        // is True for 18-digit literals.
        let shared = (p0.min(*p1).floor() as usize).saturating_sub(1);
        if shared > 0 && !bigfloat_digits_match_to(d0, d1, shared) {
          return Ok(Expr::Identifier("False".to_string()));
        }
      }
      if any_real {
        let mut tol = f64::max(first.abs(), v.abs()) * (2.0_f64).powi(-46);
        if let Some(p) = min_bigfloat_precision {
          // wolframscript's `Equal` is more lenient than
          // `|a - b| < 10^-p`: it returns True whenever the
          // precision of the difference is below 1 (i.e. the
          // result has no significant digit). Practically this
          // works out to roughly an extra decade of slack —
          // `13.1416``4 == 13.1413``4 → True` even though
          // `|Δ| = 3e-4 > 10^-5.12`. Use `10^-(p - 1)` as the
          // tolerance for precision-tagged operands so low-
          // precision near-equal accuracy literals match
          // wolframscript.
          let widened_p = (p - 1.0).max(0.0);
          let prec_tol =
            f64::max(first.abs(), v.abs()) * 10.0_f64.powf(-widened_p);
          if prec_tol > tol {
            tol = prec_tol;
          }
        }
        if (first - v).abs() > tol && first != v {
          return Ok(Expr::Identifier("False".to_string()));
        }
      } else if v != first {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Only stay symbolic if at least one arg has free symbols
  if args.iter().any(crate::evaluator::has_free_symbols) {
    Ok(Expr::FunctionCall {
      name: "Equal".to_string(),
      args: args.to_vec().into(),
    })
  } else {
    // No free symbols, not identical → False
    Ok(Expr::Identifier("False".to_string()))
  }
}

/// Unequal[a, b] or a != b - Tests for inequality
/// Returns False if any args are identical, True if all are numeric and pairwise different,
/// or stays symbolic (unevaluated) if args contain symbols and aren't identical.
pub fn unequal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Unequal expects at least 2 arguments".into(),
    ));
  }

  use crate::functions::math_ast::try_eval_to_f64;

  let strs: Vec<String> =
    args.iter().map(crate::syntax::expr_to_string).collect();
  let has_free = args.iter().any(crate::evaluator::has_free_symbols);

  // For symbolic chains, Wolfram only collapses Unequal to False when an
  // *adjacent* pair is structurally identical (a != a != b → False, but
  // a != b != a stays unevaluated). For all-numeric / all-string args,
  // any duplicate (even non-adjacent) collapses to False.
  if !has_free {
    for i in 0..strs.len() {
      for j in i + 1..strs.len() {
        if strs[i] == strs[j] {
          return Ok(Expr::Identifier("False".to_string()));
        }
      }
    }
  } else {
    for i in 0..strs.len() - 1 {
      if strs[i] == strs[i + 1] {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }

  // Check if all args are numeric
  let nums: Vec<Option<f64>> = args.iter().map(try_eval_to_f64).collect();
  if nums.iter().all(|n| n.is_some()) {
    // All numeric and pairwise different (checked above via strings)
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Only stay symbolic if at least one arg has free symbols
  if has_free {
    Ok(Expr::FunctionCall {
      name: "Unequal".to_string(),
      args: args.to_vec().into(),
    })
  } else {
    // No free symbols, pairwise different → True
    Ok(Expr::Identifier("True".to_string()))
  }
}

/// Helper to extract numeric value from Expr — delegates to try_eval_to_f64_with_infinity for full recursive evaluation
fn expr_to_num(expr: &Expr) -> Option<f64> {
  crate::functions::math_ast::try_eval_to_f64_with_infinity(expr)
}

/// Compare two expressions exactly when both are integer-valued
/// (Integer or BigInteger). f64 conversion loses ULPs above ~2^53,
/// so `2^60 < 2^60 + 1` would round to a tie and return False; bypass
/// that by comparing the BigInts directly. Returns `None` if either
/// side isn't an exact integer.
fn compare_exact_integers(a: &Expr, b: &Expr) -> Option<std::cmp::Ordering> {
  fn as_bigint(e: &Expr) -> Option<num_bigint::BigInt> {
    match e {
      Expr::Integer(n) => Some(num_bigint::BigInt::from(*n)),
      Expr::BigInteger(n) => Some(n.clone()),
      _ => None,
    }
  }
  Some(as_bigint(a)?.cmp(&as_bigint(b)?))
}

/// Less[a, b] or a < b - Tests if a is less than b
pub fn less_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Less expects at least 2 arguments".into(),
    ));
  }

  // Handle Interval comparisons
  if let Some(result) =
    crate::functions::interval_ast::try_interval_compare(args, "Less")
  {
    return result;
  }

  // Detect impossible numeric ordering anywhere in the chain. For
  // `Less[a1, ..., an]` the chain is True only if every pair (ai, aj)
  // with i<j satisfies ai < aj transitively, so any pair with both
  // values numeric and ai >= aj forces the whole chain to False even
  // when intermediate values are symbolic. E.g. `Less[1, 3, x, 2]` is
  // False because 3 < ... < 2 cannot hold.
  let nums: Vec<(usize, f64)> = args
    .iter()
    .enumerate()
    .filter_map(|(i, a)| expr_to_num(a).map(|n| (i, n)))
    .collect();
  for w in nums.windows(2) {
    // Use exact-integer compare first when both endpoints are
    // BigInteger/Integer (f64 loses 1-ULP precision above ~2^53,
    // turning `2^60 < 2^60 + 1` into a tie). Only fall back to the
    // f64 ordering when at least one side isn't an exact integer.
    if let Some(ord) = compare_exact_integers(&args[w[0].0], &args[w[1].0]) {
      if !matches!(ord, std::cmp::Ordering::Less) {
        return Ok(Expr::Identifier("False".to_string()));
      }
    } else if w[0].1 >= w[1].1 {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }

  for w in args.windows(2) {
    let (a, b) = (&w[0], &w[1]);
    if let Some(ord) = compare_exact_integers(a, b) {
      if !matches!(ord, std::cmp::Ordering::Less) {
        return Ok(Expr::Identifier("False".to_string()));
      }
      continue;
    }
    let prev = match expr_to_num(a) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Less".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    let curr = match expr_to_num(b) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Less".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    if prev >= curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Greater[a, b] or a > b - Tests if a is greater than b
pub fn greater_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Greater expects at least 2 arguments".into(),
    ));
  }

  // Handle Interval comparisons
  if let Some(result) =
    crate::functions::interval_ast::try_interval_compare(args, "Greater")
  {
    return result;
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Greater".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Greater".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    if prev <= curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// LessEqual[a, b] or a <= b - Tests if a is less than or equal to b
pub fn less_equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "LessEqual expects at least 2 arguments".into(),
    ));
  }

  // Handle Interval comparisons
  if let Some(result) =
    crate::functions::interval_ast::try_interval_compare(args, "LessEqual")
  {
    return result;
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "LessEqual".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "LessEqual".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    if prev > curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// GreaterEqual[a, b] or a >= b - Tests if a is greater than or equal to b
pub fn greater_equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "GreaterEqual expects at least 2 arguments".into(),
    ));
  }

  // Handle Interval comparisons
  if let Some(result) =
    crate::functions::interval_ast::try_interval_compare(args, "GreaterEqual")
  {
    return result;
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "GreaterEqual".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "GreaterEqual".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    if prev < curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Boole[expr] - Converts True to 1 and False to 0
pub fn boole_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Boole expects exactly 1 argument".into(),
    ));
  }

  let evaluated = evaluate_expr_to_expr(&args[0])?;
  match as_bool(&evaluated) {
    Some(true) => Ok(Expr::Integer(1)),
    Some(false) => Ok(Expr::Integer(0)),
    None => Ok(Expr::FunctionCall {
      name: "Boole".to_string(),
      args: vec![evaluated].into(),
    }),
  }
}

/// TrueQ[expr] - Returns True if expr is explicitly True
pub fn true_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "TrueQ expects exactly 1 argument".into(),
    ));
  }

  let evaluated = evaluate_expr_to_expr(&args[0])?;
  Ok(Expr::Identifier(
    if matches!(&evaluated, Expr::Identifier(s) if s == "True") {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

/// Implies[a, b] - Logical implication (a implies b, i.e., !a || b)
pub fn implies_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Implies expects exactly 2 arguments".into(),
    ));
  }

  let a = evaluate_expr_to_expr(&args[0])?;
  match as_bool(&a) {
    Some(false) => Ok(Expr::Identifier("True".to_string())), // False implies anything
    Some(true) => {
      let b = evaluate_expr_to_expr(&args[1])?;
      match as_bool(&b) {
        Some(val) => Ok(Expr::Identifier(
          if val { "True" } else { "False" }.to_string(),
        )),
        None => Ok(b), // True implies symbolic expr → return the expr
      }
    }
    None => Ok(Expr::FunctionCall {
      name: "Implies".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Nand[expr1, expr2, ...] - Logical NAND (Not And)
pub fn nand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Nand[] = Not[And[]] = Not[True] = False
  if args.is_empty() {
    return Ok(Expr::Identifier("False".to_string()));
  }
  // Nand[a] = Not[a]
  if args.len() == 1 {
    let evaluated = evaluate_expr_to_expr(&args[0])?;
    return not_ast(&[evaluated]);
  }

  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(false) => return Ok(Expr::Identifier("True".to_string())),
      Some(true) => {} // Skip True values
      None => remaining.push(evaluated),
    }
  }
  if remaining.is_empty() {
    // All were True → Nand is False
    Ok(Expr::Identifier("False".to_string()))
  } else {
    // Some symbolic: Nand[remaining...]
    Ok(Expr::FunctionCall {
      name: "Nand".to_string(),
      args: remaining.into(),
    })
  }
}

/// Nor[expr1, expr2, ...] - Logical NOR (Not Or)
pub fn nor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Nor[] = Not[Or[]] = Not[False] = True
  if args.is_empty() {
    return Ok(Expr::Identifier("True".to_string()));
  }
  // Nor[a] = Not[a]
  if args.len() == 1 {
    let evaluated = evaluate_expr_to_expr(&args[0])?;
    return not_ast(&[evaluated]);
  }

  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => return Ok(Expr::Identifier("False".to_string())),
      Some(false) => {} // Skip False values
      None => remaining.push(evaluated),
    }
  }
  if remaining.is_empty() {
    // All were False → Nor is True
    Ok(Expr::Identifier("True".to_string()))
  } else {
    // Some symbolic: Nor[remaining...]
    Ok(Expr::FunctionCall {
      name: "Nor".to_string(),
      args: remaining.into(),
    })
  }
}

/// Equivalent[expr1, expr2, ...] - True if all args have the same truth value
pub fn equivalent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Equivalent[] and Equivalent[a] are vacuously True.
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  let mut has_true = false;
  let mut has_false = false;
  let mut remaining = Vec::new();

  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => has_true = true,
      Some(false) => has_false = true,
      None => remaining.push(evaluated),
    }
  }

  // If we have both True and False, it's False
  if has_true && has_false {
    return Ok(Expr::Identifier("False".to_string()));
  }

  // If all are known and the same value, it's True
  if remaining.is_empty() {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // If some symbolic: the presence of an explicit True or False reduces
  // the chain to logical conjunction/disjunction over the symbolic terms:
  // - Equivalent[..., True, ...] = And[symbolic...]
  // - Equivalent[..., False, ...] = And[Not[symbolic]...]  (symbolic all False)
  // Without any boolean literal, keep the call symbolic.
  if has_true {
    if remaining.len() == 1 {
      return Ok(remaining.into_iter().next().unwrap());
    }
    return Ok(Expr::FunctionCall {
      name: "And".to_string(),
      args: remaining.into(),
    });
  }
  if has_false {
    let negated: Vec<Expr> = remaining
      .into_iter()
      .map(|e| Expr::FunctionCall {
        name: "Not".to_string(),
        args: vec![e].into(),
      })
      .collect();
    if negated.len() == 1 {
      return Ok(negated.into_iter().next().unwrap());
    }
    return Ok(Expr::FunctionCall {
      name: "And".to_string(),
      args: negated.into(),
    });
  }
  Ok(Expr::FunctionCall {
    name: "Equivalent".to_string(),
    args: remaining.into(),
  })
}

/// BooleanTable[expr, {var1, var2, ...}] - Truth table for a boolean expression
/// Evaluates expr for all 2^n combinations of True/False for the given variables.
/// First variable cycles slowest (MSB), last variable cycles fastest (LSB).
pub fn boolean_table_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BooleanTable expects exactly 2 arguments".into(),
    ));
  }

  let body = &args[0];
  let vars = match &args[1] {
    Expr::List(items) => items
      .iter()
      .map(|item| match item {
        Expr::Identifier(name) => Ok(name.clone()),
        _ => Err(InterpreterError::EvaluationError(
          "BooleanTable: variables must be symbols".into(),
        )),
      })
      .collect::<Result<Vec<String>, _>>()?,
    Expr::Identifier(name) => vec![name.clone()],
    _ => {
      return Err(InterpreterError::EvaluationError(
        "BooleanTable: second argument must be a list of variables".into(),
      ));
    }
  };

  let n = vars.len();
  let num_combos = 1usize << n;
  let mut results = Vec::with_capacity(num_combos);

  for i in 0..num_combos {
    // Build substitutions: first variable = MSB, last = LSB
    let mut substituted = body.clone();
    for (j, var_name) in vars.iter().enumerate() {
      let bit = (i >> (n - 1 - j)) & 1;
      let val = if bit == 0 {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("False".to_string())
      };
      substituted =
        crate::syntax::substitute_variable(&substituted, var_name, &val);
    }
    let result = evaluate_expr_to_expr(&substituted)?;
    results.push(result);
  }

  Ok(Expr::List(results.into()))
}

/// LogicalExpand[expr] - expand logical expression to disjunctive normal form
pub fn logical_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogicalExpand expects exactly 1 argument".into(),
    ));
  }

  let expr = evaluate_expr_to_expr(&args[0])?;
  Ok(normalize_not(&to_dnf(&expr)))
}

/// Recursively convert FunctionCall("Not", [x]) to UnaryOp(Not, x)
fn normalize_not(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if name == "Not" && args.len() == 1 => {
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Not,
        operand: Box::new(normalize_not(&args[0])),
      }
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args.iter().map(normalize_not).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    _ => expr.clone(),
  }
}

/// Convert an expression to disjunctive normal form (DNF).
/// Steps: eliminate compound connectives → push Not inward → distribute And over Or.
fn to_dnf(expr: &Expr) -> Expr {
  let eliminated = eliminate_connectives(expr);
  let negated = push_not_inward(&eliminated);
  distribute_and_over_or(&negated)
}

/// Step 1: Eliminate Implies, Equivalent, Xor, Nand, Nor
fn eliminate_connectives(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Implies" if args.len() == 2 => {
          // Implies[a, b] → Or[b, Not[a]]
          let a = eliminate_connectives(&args[0]);
          let b = eliminate_connectives(&args[1]);
          Expr::FunctionCall {
            name: "Or".to_string(),
            args: vec![
              b,
              Expr::FunctionCall {
                name: "Not".to_string(),
                args: vec![a].into(),
              },
            ]
            .into(),
          }
        }
        "Equivalent" if args.len() >= 2 => {
          // Equivalent[a, b] → And[Or[Not[a], b], Or[a, Not[b]]]
          // For more args: pairwise equivalence
          let elim_args: Vec<Expr> =
            args.iter().map(eliminate_connectives).collect();
          if elim_args.len() == 2 {
            let a = &elim_args[0];
            let b = &elim_args[1];
            // (a && b) || (!a && !b)
            Expr::FunctionCall {
              name: "Or".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "And".to_string(),
                  args: vec![a.clone(), b.clone()].into(),
                },
                Expr::FunctionCall {
                  name: "And".to_string(),
                  args: vec![
                    Expr::FunctionCall {
                      name: "Not".to_string(),
                      args: vec![a.clone()].into(),
                    },
                    Expr::FunctionCall {
                      name: "Not".to_string(),
                      args: vec![b.clone()].into(),
                    },
                  ]
                  .into(),
                },
              ]
              .into(),
            }
          } else {
            // Pairwise: And[Equivalent[a1,a2], Equivalent[a2,a3], ...]
            let mut pairs = Vec::new();
            for i in 0..elim_args.len() - 1 {
              pairs.push(eliminate_connectives(&Expr::FunctionCall {
                name: "Equivalent".to_string(),
                args: vec![elim_args[i].clone(), elim_args[i + 1].clone()]
                  .into(),
              }));
            }
            Expr::FunctionCall {
              name: "And".to_string(),
              args: pairs.into(),
            }
          }
        }
        "Xor" if args.len() >= 2 => {
          // Xor[a, b] → Or[And[a, Not[b]], And[b, Not[a]]] (Wolfram order)
          let elim_args: Vec<Expr> =
            args.iter().map(eliminate_connectives).collect();
          if elim_args.len() == 2 {
            let a = &elim_args[0];
            let b = &elim_args[1];
            Expr::FunctionCall {
              name: "Or".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "And".to_string(),
                  args: vec![
                    a.clone(),
                    Expr::FunctionCall {
                      name: "Not".to_string(),
                      args: vec![b.clone()].into(),
                    },
                  ]
                  .into(),
                },
                Expr::FunctionCall {
                  name: "And".to_string(),
                  args: vec![
                    b.clone(),
                    Expr::FunctionCall {
                      name: "Not".to_string(),
                      args: vec![a.clone()].into(),
                    },
                  ]
                  .into(),
                },
              ]
              .into(),
            }
          } else {
            // Reduce: Xor[a, b, c, ...] → Xor[Xor[a, b], c, ...]
            let mut result = elim_args[0].clone();
            for arg in &elim_args[1..] {
              result = eliminate_connectives(&Expr::FunctionCall {
                name: "Xor".to_string(),
                args: vec![result, arg.clone()].into(),
              });
            }
            result
          }
        }
        "Nand" if args.len() >= 2 => {
          // Nand[a, b] → Not[And[a, b]]
          let elim_args: Vec<Expr> =
            args.iter().map(eliminate_connectives).collect();
          Expr::FunctionCall {
            name: "Not".to_string(),
            args: vec![Expr::FunctionCall {
              name: "And".to_string(),
              args: elim_args.into(),
            }]
            .into(),
          }
        }
        "Nor" if args.len() >= 2 => {
          // Nor[a, b] → Not[Or[a, b]]
          let elim_args: Vec<Expr> =
            args.iter().map(eliminate_connectives).collect();
          Expr::FunctionCall {
            name: "Not".to_string(),
            args: vec![Expr::FunctionCall {
              name: "Or".to_string(),
              args: elim_args.into(),
            }]
            .into(),
          }
        }
        "And" | "Or" | "Not" => {
          let new_args: Vec<Expr> =
            args.iter().map(eliminate_connectives).collect();
          Expr::FunctionCall {
            name: name.clone(),
            args: new_args.into(),
          }
        }
        _ => expr.clone(),
      }
    }
    // Convert UnaryOp(Not, x) to FunctionCall("Not", [x]) for uniform processing
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Not,
      operand,
    } => {
      let inner = eliminate_connectives(operand);
      Expr::FunctionCall {
        name: "Not".to_string(),
        args: vec![inner].into(),
      }
    }
    _ => expr.clone(),
  }
}

/// Helper: apply Not to an inner expression and push inward
fn apply_not_inward(inner: &Expr) -> Expr {
  match inner {
    // Not[Not[a]] → a (handles FunctionCall form)
    Expr::FunctionCall {
      name: inner_name,
      args: inner_args,
    } if inner_name == "Not" && inner_args.len() == 1 => {
      push_not_inward(&inner_args[0])
    }
    // Not[Not[a]] → a (handles UnaryOp form)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Not,
      operand,
    } => push_not_inward(operand),
    // Not[And[a, b, ...]] → Or[Not[a], Not[b], ...]
    Expr::FunctionCall {
      name: inner_name,
      args: inner_args,
    } if inner_name == "And" => {
      let new_args: Vec<Expr> =
        inner_args.iter().map(apply_not_inward).collect();
      Expr::FunctionCall {
        name: "Or".to_string(),
        args: new_args.into(),
      }
    }
    // Not[Or[a, b, ...]] → And[Not[a], Not[b], ...]
    Expr::FunctionCall {
      name: inner_name,
      args: inner_args,
    } if inner_name == "Or" => {
      let new_args: Vec<Expr> =
        inner_args.iter().map(apply_not_inward).collect();
      Expr::FunctionCall {
        name: "And".to_string(),
        args: new_args.into(),
      }
    }
    // Not[True] → False, Not[False] → True
    Expr::Identifier(s) if s == "True" => Expr::Identifier("False".to_string()),
    Expr::Identifier(s) if s == "False" => Expr::Identifier("True".to_string()),
    // Not[other] → Not[other] (keep as-is, recurse into inner)
    other => {
      let recurse = push_not_inward(other);
      Expr::FunctionCall {
        name: "Not".to_string(),
        args: vec![recurse].into(),
      }
    }
  }
}

/// Step 2: Push Not inward using De Morgan's laws and double negation elimination
fn push_not_inward(expr: &Expr) -> Expr {
  match expr {
    // Handle Not as FunctionCall
    Expr::FunctionCall { name, args } if name == "Not" && args.len() == 1 => {
      apply_not_inward(&args[0])
    }
    // Handle Not as UnaryOp
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Not,
      operand,
    } => apply_not_inward(operand),
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args.iter().map(push_not_inward).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    _ => expr.clone(),
  }
}

/// Step 3: Distribute And over Or to achieve DNF.
/// The result is an Or of Ands (or simpler expressions).
fn distribute_and_over_or(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if name == "And" => {
      // First recursively convert all sub-expressions
      let sub: Vec<Expr> = args.iter().map(distribute_and_over_or).collect();

      // Flatten nested Ands
      let mut and_groups: Vec<Vec<Expr>> = vec![vec![]];

      for term in &sub {
        match term {
          Expr::FunctionCall {
            name: tn,
            args: targs,
          } if tn == "Or" => {
            // For each existing group, cross-product with Or alternatives
            // (Wolfram order: iterate existing groups first, then alternatives)
            let mut new_groups = Vec::new();
            for existing in &and_groups {
              for alt in targs {
                let alt_literals = match alt {
                  Expr::FunctionCall {
                    name: an,
                    args: aargs,
                  } if an == "And" => aargs.clone(),
                  _ => vec![alt.clone()].into(),
                };
                let mut group = existing.clone();
                group.extend(alt_literals.clone());
                new_groups.push(group);
              }
            }
            and_groups = new_groups;
          }
          Expr::FunctionCall {
            name: an,
            args: aargs,
          } if an == "And" => {
            // Flatten nested And
            for group in &mut and_groups {
              group.extend(aargs.clone());
            }
          }
          _ => {
            for group in &mut and_groups {
              group.push(term.clone());
            }
          }
        }
      }

      // Build result
      let or_terms: Vec<Expr> = and_groups
        .into_iter()
        .map(|group| {
          if group.len() == 1 {
            group.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "And".to_string(),
              args: group.into(),
            }
          }
        })
        .collect();

      if or_terms.len() == 1 {
        or_terms.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Or".to_string(),
          args: or_terms.into(),
        }
      }
    }
    Expr::FunctionCall { name, args } if name == "Or" => {
      // Recursively distribute in each alternative, then flatten Or
      let mut result = Vec::new();
      for arg in args {
        let mut converted = distribute_and_over_or(arg);
        let is_or =
          matches!(&converted, Expr::FunctionCall { name, .. } if name == "Or");
        if is_or {
          if let Expr::FunctionCall {
            args: ref mut oargs,
            ..
          } = converted
          {
            result.extend(oargs.drain(..));
          }
        } else {
          result.push(converted);
        }
      }
      if result.len() == 1 {
        result.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Or".to_string(),
          args: result.into(),
        }
      }
    }
    Expr::FunctionCall { name, args } => {
      // Recurse into other function calls (like Not)
      let new_args: Vec<Expr> =
        args.iter().map(distribute_and_over_or).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    _ => expr.clone(),
  }
}

/// Convert an expression to conjunctive normal form (CNF).
/// Steps: eliminate compound connectives → push Not inward → distribute Or over And → simplify.
fn to_cnf(expr: &Expr) -> Expr {
  let eliminated = eliminate_connectives(expr);
  let negated = push_not_inward(&eliminated);
  let cnf = distribute_or_over_and(&negated);
  simplify_cnf(&cnf)
}

/// Remove tautological clauses from CNF (clauses containing x and Not[x]).
fn simplify_cnf(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if name == "And" => {
      let simplified: Vec<Expr> = args
        .iter()
        .map(simplify_cnf)
        .filter(|clause| !is_tautological_clause(clause))
        .collect();
      if simplified.is_empty() {
        Expr::Identifier("True".to_string())
      } else if simplified.len() == 1 {
        simplified.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "And".to_string(),
          args: simplified.into(),
        }
      }
    }
    _ => expr.clone(),
  }
}

/// Check if an Or clause is a tautology (contains both x and Not[x]).
fn is_tautological_clause(clause: &Expr) -> bool {
  let literals = match clause {
    Expr::FunctionCall { name, args } if name == "Or" => args.clone(),
    _ => return false,
  };

  for lit in &literals {
    // Check if the negation of this literal also appears
    match lit {
      Expr::FunctionCall { name, args } if name == "Not" && args.len() == 1 => {
        // This is Not[x] — check if x appears
        let inner = &args[0];
        for other in &literals {
          if expr_eq(other, inner) {
            return true;
          }
        }
      }
      _ => {
        // This is x — check if Not[x] appears
        for other in &literals {
          if let Expr::FunctionCall { name, args } = other
            && name == "Not"
            && args.len() == 1
            && expr_eq(&args[0], lit)
          {
            return true;
          }
        }
      }
    }
  }
  false
}

/// Simple structural equality check for expressions
fn expr_eq(a: &Expr, b: &Expr) -> bool {
  crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
}

/// Distribute Or over And to achieve CNF (AND of ORs).
/// This is the dual of distribute_and_over_or.
fn distribute_or_over_and(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if name == "Or" => {
      // First recursively convert all sub-expressions
      let sub: Vec<Expr> = args.iter().map(distribute_or_over_and).collect();

      // Cross-product: distribute Or over And
      let mut or_groups: Vec<Vec<Expr>> = vec![vec![]];

      for term in &sub {
        match term {
          Expr::FunctionCall {
            name: tn,
            args: targs,
          } if tn == "And" => {
            let mut new_groups = Vec::new();
            for existing in &or_groups {
              for alt in targs {
                let alt_literals = match alt {
                  Expr::FunctionCall {
                    name: on,
                    args: oargs,
                  } if on == "Or" => oargs.clone(),
                  _ => vec![alt.clone()].into(),
                };
                let mut group = existing.clone();
                group.extend(alt_literals);
                new_groups.push(group);
              }
            }
            or_groups = new_groups;
          }
          Expr::FunctionCall {
            name: on,
            args: oargs,
          } if on == "Or" => {
            for group in &mut or_groups {
              group.extend(oargs.clone());
            }
          }
          _ => {
            for group in &mut or_groups {
              group.push(term.clone());
            }
          }
        }
      }

      // Build result: And of Ors
      let and_terms: Vec<Expr> = or_groups
        .into_iter()
        .map(|group| {
          if group.len() == 1 {
            group.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Or".to_string(),
              args: group.into(),
            }
          }
        })
        .collect();

      if and_terms.len() == 1 {
        and_terms.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "And".to_string(),
          args: and_terms.into(),
        }
      }
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      let mut result = Vec::new();
      for arg in args {
        let mut converted = distribute_or_over_and(arg);
        let is_and = matches!(&converted, Expr::FunctionCall { name, .. } if name == "And");
        if is_and {
          if let Expr::FunctionCall {
            args: ref mut aargs,
            ..
          } = converted
          {
            result.extend(aargs.drain(..));
          }
        } else {
          result.push(converted);
        }
      }
      if result.len() == 1 {
        result.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "And".to_string(),
          args: result.into(),
        }
      }
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> =
        args.iter().map(distribute_or_over_and).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    _ => expr.clone(),
  }
}

// ─── BooleanConvert ────────────────────────────────────────────────

/// BooleanConvert[expr] or BooleanConvert[expr, form]
/// Convert boolean expressions to different normal forms.
/// Supported forms: "DNF", "CNF", or default (eliminate compound connectives).
pub fn boolean_convert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "BooleanConvert expects 1 or 2 arguments".into(),
    ));
  }

  let expr = &args[0];
  // Helper: sort literals within And/Or clauses for canonical output
  // BooleanConvert puts negated variables before positive ones
  fn sort_boolean_expr(expr: &Expr) -> Expr {
    match expr {
      Expr::FunctionCall { name, args } if name == "Or" || name == "And" => {
        let mut sorted_args: Vec<Expr> =
          args.iter().map(sort_boolean_expr).collect();
        sorted_args.sort_by(|a, b| {
          let key = |e: &Expr| -> (u8, String) {
            match e {
              Expr::FunctionCall { name, args }
                if name == "Not" && args.len() == 1 =>
              {
                (0, crate::syntax::expr_to_string(&args[0]))
              }
              Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Not,
                operand,
              } => (0, crate::syntax::expr_to_string(operand)),
              _ => (1, crate::syntax::expr_to_string(e)),
            }
          };
          key(a).cmp(&key(b))
        });
        Expr::FunctionCall {
          name: name.clone(),
          args: sorted_args.into(),
        }
      }
      _ => expr.clone(),
    }
  }

  let form = if args.len() == 2 {
    match &args[1] {
      Expr::String(s) => s.as_str().to_string(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "BooleanConvert".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    "DNF".to_string()
  };

  let result = match form.as_str() {
    "DNF" => {
      let dnf = to_dnf(expr);
      sort_boolean_expr(&normalize_not(&dnf))
    }
    "CNF" => {
      let cnf = to_cnf(expr);
      sort_boolean_expr(&normalize_not(&cnf))
    }
    "OR" => {
      let eliminated = eliminate_connectives(expr);
      let negated = push_not_inward(&eliminated);
      sort_boolean_expr(&normalize_not(&negated))
    }
    "AND" => {
      let cnf = to_cnf(expr);
      sort_boolean_expr(&normalize_not(&cnf))
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BooleanConvert".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  Ok(result)
}

/// Extract free boolean variables (identifiers) from a boolean expression.
fn collect_boolean_variables(expr: &Expr, vars: &mut BTreeSet<String>) {
  match expr {
    Expr::Identifier(name) if name != "True" && name != "False" => {
      vars.insert(name.clone());
    }
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "And"
          | "Or"
          | "Not"
          | "Xor"
          | "Nand"
          | "Nor"
          | "Implies"
          | "Equivalent"
      ) =>
    {
      for arg in args {
        collect_boolean_variables(arg, vars);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_boolean_variables(left, vars);
      collect_boolean_variables(right, vars);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_boolean_variables(operand, vars);
    }
    _ => {}
  }
}

/// TautologyQ[expr] - True if the boolean expression is true for all variable assignments.
pub fn tautology_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "TautologyQ".to_string(),
      args: args.to_vec().into(),
    });
  }

  let expr = &args[0];

  // Collect all boolean variables
  let mut vars = BTreeSet::new();
  collect_boolean_variables(expr, &mut vars);
  let var_list: Vec<String> = vars.into_iter().collect();
  let n = var_list.len();

  // Limit to 20 variables to prevent combinatorial explosion
  if n > 20 {
    return Ok(Expr::FunctionCall {
      name: "TautologyQ".to_string(),
      args: args.to_vec().into(),
    });
  }

  for bits in 0..(1u64 << n) {
    let mut substituted = expr.clone();
    for (i, var_name) in var_list.iter().enumerate() {
      let val = if (bits >> i) & 1 == 1 {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("False".to_string())
      };
      substituted =
        crate::syntax::substitute_variable(&substituted, var_name, &val);
    }
    let result = evaluate_expr_to_expr(&substituted)?;
    if !matches!(&result, Expr::Identifier(s) if s == "True") {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }

  Ok(Expr::Identifier("True".to_string()))
}

/// BooleanMinimize[expr] - Find the minimal sum-of-products form.
pub fn boolean_minimize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "BooleanMinimize".to_string(),
      args: args.to_vec().into(),
    });
  }

  let expr = &args[0];

  // Handle trivial cases
  if matches!(expr, Expr::Identifier(s) if s == "True") {
    return Ok(Expr::Identifier("True".to_string()));
  }
  if matches!(expr, Expr::Identifier(s) if s == "False") {
    return Ok(Expr::Identifier("False".to_string()));
  }

  // Collect all boolean variables
  let mut vars_set = BTreeSet::new();
  collect_boolean_variables(expr, &mut vars_set);
  let var_list: Vec<String> = vars_set.into_iter().collect();
  let n = var_list.len();

  // Limit to 20 variables
  if n > 20 {
    return Ok(Expr::FunctionCall {
      name: "BooleanMinimize".to_string(),
      args: args.to_vec().into(),
    });
  }

  if n == 0 {
    return evaluate_expr_to_expr(expr);
  }

  // Build truth table: collect minterms (variable assignments that give True)
  let mut minterms: Vec<u64> = Vec::new();
  for bits in 0..(1u64 << n) {
    let mut substituted = expr.clone();
    for (i, var_name) in var_list.iter().enumerate() {
      let val = if (bits >> i) & 1 == 1 {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("False".to_string())
      };
      substituted =
        crate::syntax::substitute_variable(&substituted, var_name, &val);
    }
    let result = evaluate_expr_to_expr(&substituted)?;
    if matches!(&result, Expr::Identifier(s) if s == "True") {
      minterms.push(bits);
    }
  }

  if minterms.is_empty() {
    return Ok(Expr::Identifier("False".to_string()));
  }
  if minterms.len() == (1usize << n) {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Quine-McCluskey: find prime implicants
  let prime_implicants = quine_mccluskey(&minterms, n);

  // Greedy set cover
  let cover = greedy_cover(&prime_implicants, &minterms);

  // Convert to Boolean expression
  implicants_to_expr(&cover, &var_list)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Implicant {
  mask: u64,  // which bits matter (1 = matters)
  value: u64, // required value for bits that matter
}

impl Implicant {
  fn covers(&self, minterm: u64) -> bool {
    (minterm & self.mask) == self.value
  }
}

fn quine_mccluskey(minterms: &[u64], n: usize) -> Vec<Implicant> {
  let full_mask = (1u64 << n) - 1;

  let mut current: Vec<Implicant> = minterms
    .iter()
    .map(|&m| Implicant {
      mask: full_mask,
      value: m,
    })
    .collect();

  let mut all_prime: Vec<Implicant> = Vec::new();

  loop {
    let mut next: Vec<Implicant> = Vec::new();
    let mut used = vec![false; current.len()];

    for i in 0..current.len() {
      for j in (i + 1)..current.len() {
        if current[i].mask != current[j].mask {
          continue;
        }
        let diff = current[i].value ^ current[j].value;
        if diff.count_ones() == 1 {
          let new_mask = current[i].mask & !diff;
          let new_value = current[i].value & new_mask;
          let merged = Implicant {
            mask: new_mask,
            value: new_value,
          };
          if !next.contains(&merged) {
            next.push(merged);
          }
          used[i] = true;
          used[j] = true;
        }
      }
    }

    for (i, imp) in current.iter().enumerate() {
      if !used[i] && !all_prime.contains(imp) {
        all_prime.push(imp.clone());
      }
    }

    if next.is_empty() {
      break;
    }
    current = next;
  }

  all_prime
}

fn greedy_cover(primes: &[Implicant], minterms: &[u64]) -> Vec<Implicant> {
  let mut uncovered: Vec<u64> = minterms.to_vec();
  let mut cover: Vec<Implicant> = Vec::new();

  // Essential prime implicants
  loop {
    let mut found_essential = false;
    let mut i = 0;
    while i < uncovered.len() {
      let mt = uncovered[i];
      let covering: Vec<usize> = primes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.covers(mt))
        .map(|(idx, _)| idx)
        .collect();
      if covering.len() == 1 {
        let essential = &primes[covering[0]];
        if !cover.contains(essential) {
          cover.push(essential.clone());
          found_essential = true;
        }
        uncovered.retain(|&m| !essential.covers(m));
        i = 0;
        continue;
      }
      i += 1;
    }
    if !found_essential {
      break;
    }
  }

  // Greedy for remaining
  while !uncovered.is_empty() {
    let best = primes
      .iter()
      .filter(|p| !cover.contains(p))
      .max_by_key(|p| uncovered.iter().filter(|&&m| p.covers(m)).count());
    if let Some(best_prime) = best {
      let bp = best_prime.clone();
      uncovered.retain(|&m| !bp.covers(m));
      cover.push(bp);
    } else {
      break;
    }
  }

  cover
}

fn implicants_to_expr(
  implicants: &[Implicant],
  vars: &[String],
) -> Result<Expr, InterpreterError> {
  if implicants.is_empty() {
    return Ok(Expr::Identifier("False".to_string()));
  }

  let mut terms: Vec<Expr> = Vec::new();
  for imp in implicants {
    let mut literals: Vec<Expr> = Vec::new();
    for (i, var_name) in vars.iter().enumerate() {
      let bit = 1u64 << i;
      if imp.mask & bit != 0 {
        let var = Expr::Identifier(var_name.clone());
        if imp.value & bit != 0 {
          literals.push(var);
        } else {
          literals.push(Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Not,
            operand: Box::new(var),
          });
        }
      }
    }
    let term = if literals.is_empty() {
      Expr::Identifier("True".to_string())
    } else if literals.len() == 1 {
      literals.remove(0)
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: literals.into(),
      }
    };
    terms.push(term);
  }

  if terms.len() == 1 {
    Ok(terms.remove(0))
  } else {
    Ok(Expr::FunctionCall {
      name: "Or".to_string(),
      args: terms.into(),
    })
  }
}

/// BooleanCountingFunction[spec, vars] — build a Boolean expression
/// (DNF) that is True iff the count of True variables in `vars` matches
/// `spec`. Supported `spec` shapes:
///
///   k                   — at most k (counts {0, 1, …, k})
///   {k}                 — exactly k
///   {kmin, kmax}        — between kmin and kmax inclusive
///   {kmin, kmax, step}  — counts {kmin, kmin+step, …} ≤ kmax
///
/// Wolfram emits a specific (sometimes non-minimal) form for each
/// spec shape. We match wolframscript byte-for-byte for the
/// "exactly k" and "at most k" patterns, where the structure is simple
/// and unambiguous. For other shapes the result is a logically
/// equivalent DNF produced by running the sum-of-minterms through
/// `BooleanMinimize` — it may differ from wolframscript's particular
/// term selection but is guaranteed to be correct.
pub fn boolean_counting_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "BooleanCountingFunction".to_string(),
      args: args.to_vec().into(),
    });
  }
  let vars: Vec<String> = match &args[1] {
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for it in items.iter() {
        if let Expr::Identifier(s) = it {
          out.push(s.clone());
        } else {
          return Ok(Expr::FunctionCall {
            name: "BooleanCountingFunction".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
      out
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BooleanCountingFunction".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let n = vars.len();
  if n == 0 {
    return Ok(Expr::FunctionCall {
      name: "BooleanCountingFunction".to_string(),
      args: args.to_vec().into(),
    });
  }

  let count_set: Vec<usize> = match parse_counting_spec(&args[0], n) {
    Some(cs) => cs,
    None => {
      return Ok(Expr::FunctionCall {
        name: "BooleanCountingFunction".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Trivial extremes.
  if count_set.is_empty() {
    return Ok(Expr::Identifier("False".to_string()));
  }
  if count_set.len() == n + 1 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // "Exactly k" — natural sum-of-minterms in lex order over which
  // variables are true.
  if count_set.len() == 1 {
    return Ok(exactly_k_dnf(&vars, count_set[0]));
  }

  // "At most k" — Or over (n-k)-subsets of negated variables.
  // Detected as count_set = {0, 1, …, kmax}.
  if let Some(kmax) = detect_at_most(&count_set)
    && kmax < n
  {
    return Ok(at_most_k_dnf(&vars, kmax));
  }

  // General case: emit sum-of-minterms and minimize.
  let dnf = minterms_to_dnf(&vars, &count_set);
  boolean_minimize_ast(&[dnf])
}

/// Parse the first argument of `BooleanCountingFunction` into the set of
/// true-counts that should produce True. Returns None for unrecognized
/// shapes; counts outside [0, n] are dropped so out-of-range specs
/// degrade to False gracefully.
fn parse_counting_spec(spec: &Expr, n: usize) -> Option<Vec<usize>> {
  let in_range = |k: i128| -> Option<usize> {
    if (0..=n as i128).contains(&k) {
      Some(k as usize)
    } else {
      None
    }
  };
  match spec {
    // `k_max` — at most k_max true.
    Expr::Integer(k) if *k >= 0 => {
      let kmax = (*k as usize).min(n);
      Some((0..=kmax).collect())
    }
    Expr::List(items) => match items.len() {
      // {k} — exactly k.
      1 => {
        if let Expr::Integer(k) = &items[0] {
          in_range(*k).map(|v| vec![v])
        } else {
          None
        }
      }
      // {kmin, kmax} — between kmin and kmax inclusive.
      2 => {
        if let (Expr::Integer(a), Expr::Integer(b)) = (&items[0], &items[1])
          && *a >= 0
          && *b >= *a
        {
          let kmin = (*a as usize).min(n);
          let kmax = (*b as usize).min(n);
          Some((kmin..=kmax).collect())
        } else {
          None
        }
      }
      // {kmin, kmax, step} — counts {kmin, kmin+step, …} ≤ kmax.
      3 => {
        if let (Expr::Integer(a), Expr::Integer(b), Expr::Integer(s)) =
          (&items[0], &items[1], &items[2])
          && *a >= 0
          && *b >= *a
          && *s >= 1
        {
          let kmin = *a as usize;
          let kmax = *b as usize;
          let step = *s as usize;
          let mut out = Vec::new();
          let mut k = kmin;
          while k <= kmax {
            if k <= n {
              out.push(k);
            }
            k += step;
          }
          Some(out)
        } else {
          None
        }
      }
      _ => None,
    },
    _ => None,
  }
}

/// Detect the "at most k" pattern: count_set == {0, 1, …, kmax}.
fn detect_at_most(count_set: &[usize]) -> Option<usize> {
  if count_set.is_empty() || count_set[0] != 0 {
    return None;
  }
  for (i, &v) in count_set.iter().enumerate() {
    if v != i {
      return None;
    }
  }
  Some(*count_set.last().unwrap())
}

/// "Exactly k" DNF: Or over all k-subsets `S` of `vars` of
/// `AND_{v in S} v && AND_{v not in S} !v`. Subsets are emitted in
/// the lex order returned by `subsets_in_lex_order` (matches Wolfram's
/// output).
fn exactly_k_dnf(vars: &[String], k: usize) -> Expr {
  let n = vars.len();
  if k > n {
    return Expr::Identifier("False".to_string());
  }
  let subsets = subsets_in_lex_order(n, k);
  let mut terms: Vec<Expr> = Vec::with_capacity(subsets.len());
  for s in &subsets {
    let mut literals: Vec<Expr> = Vec::with_capacity(n);
    for (i, name) in vars.iter().enumerate() {
      let var = Expr::Identifier(name.clone());
      if s.contains(&i) {
        literals.push(var);
      } else {
        literals.push(Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Not,
          operand: Box::new(var),
        });
      }
    }
    terms.push(if literals.len() == 1 {
      literals.remove(0)
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: literals.into(),
      }
    });
  }
  or_of(terms)
}

/// "At most k_max" DNF: Or over (n - k_max)-subsets `S` of `vars` of
/// `AND_{v in S} !v`. Subsets are in lex order.
fn at_most_k_dnf(vars: &[String], kmax: usize) -> Expr {
  let n = vars.len();
  let neg_count = n.saturating_sub(kmax);
  if neg_count == 0 {
    return Expr::Identifier("True".to_string());
  }
  let subsets = subsets_in_lex_order(n, neg_count);
  let mut terms: Vec<Expr> = Vec::with_capacity(subsets.len());
  for s in &subsets {
    let mut literals: Vec<Expr> = Vec::with_capacity(s.len());
    for &i in s {
      literals.push(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Not,
        operand: Box::new(Expr::Identifier(vars[i].clone())),
      });
    }
    terms.push(if literals.len() == 1 {
      literals.remove(0)
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: literals.into(),
      }
    });
  }
  or_of(terms)
}

/// Sum-of-minterms DNF for the given count set, in lex order over
/// which variables are true.
fn minterms_to_dnf(vars: &[String], count_set: &[usize]) -> Expr {
  let n = vars.len();
  let mut count_set: Vec<usize> = count_set.to_vec();
  count_set.sort_unstable();
  let mut terms: Vec<Expr> = Vec::new();
  for &k in &count_set {
    let subsets = subsets_in_lex_order(n, k);
    for s in &subsets {
      let mut literals: Vec<Expr> = Vec::with_capacity(n);
      for (i, name) in vars.iter().enumerate() {
        let var = Expr::Identifier(name.clone());
        if s.contains(&i) {
          literals.push(var);
        } else {
          literals.push(Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Not,
            operand: Box::new(var),
          });
        }
      }
      terms.push(if literals.len() == 1 {
        literals.remove(0)
      } else {
        Expr::FunctionCall {
          name: "And".to_string(),
          args: literals.into(),
        }
      });
    }
  }
  or_of(terms)
}

/// All k-subsets of {0, 1, …, n-1} in lex order. Each subset is a
/// strictly increasing Vec<usize>.
fn subsets_in_lex_order(n: usize, k: usize) -> Vec<Vec<usize>> {
  let mut out = Vec::new();
  if k > n {
    return out;
  }
  let mut cur: Vec<usize> = (0..k).collect();
  if k == 0 {
    out.push(Vec::new());
    return out;
  }
  loop {
    out.push(cur.clone());
    // Find the rightmost index that can be incremented.
    let mut i = k;
    while i > 0 {
      i -= 1;
      if cur[i] < n - (k - i) {
        cur[i] += 1;
        for j in i + 1..k {
          cur[j] = cur[j - 1] + 1;
        }
        break;
      }
      if i == 0 {
        return out;
      }
    }
    // Termination check: when no index could be incremented we've
    // emitted the last subset (n-k, n-k+1, …, n-1).
    if cur[0] > n - k {
      return out;
    }
  }
}

/// Wrap `terms` in an `Or` head, or unwrap if there's only one.
fn or_of(mut terms: Vec<Expr>) -> Expr {
  if terms.is_empty() {
    return Expr::Identifier("False".to_string());
  }
  if terms.len() == 1 {
    return terms.remove(0);
  }
  Expr::FunctionCall {
    name: "Or".to_string(),
    args: terms.into(),
  }
}

/// VectorLess[{v1, v2, ...}] - componentwise strict less-than comparison
/// Returns True if for each consecutive pair, every component is strictly less.
pub fn vector_less_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  vector_compare_ast(args, "VectorLess", false)
}

/// VectorLessEqual[{v1, v2, ...}] - componentwise less-than-or-equal comparison
/// Returns True if for each consecutive pair, every component is less than or equal.
pub fn vector_less_equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  vector_compare_ast(args, "VectorLessEqual", true)
}

fn vector_compare_ast(
  args: &[Expr],
  name: &str,
  allow_equal: bool,
) -> Result<Expr, InterpreterError> {
  // VectorLess/VectorLessEqual takes exactly 1 argument: a list of vectors/scalars
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    });
  }

  let items = match &args[0] {
    Expr::List(list_args) => list_args,
    _ => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if items.len() < 2 {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    });
  }

  // Check if items are vectors (lists) or scalars
  for i in 0..items.len() - 1 {
    let a = &items[i];
    let b = &items[i + 1];

    match (a, b) {
      // Both are lists (vectors)
      (Expr::List(args_a), Expr::List(args_b)) => {
        // Mismatched lengths → False
        if args_a.len() != args_b.len() {
          return Ok(Expr::Identifier("False".to_string()));
        }
        // Empty vectors → True (vacuously), continue to next pair
        for (ea, eb) in args_a.iter().zip(args_b.iter()) {
          let na = match expr_to_num(ea) {
            Some(n) => n,
            None => {
              return Ok(Expr::FunctionCall {
                name: name.to_string(),
                args: args.to_vec().into(),
              });
            }
          };
          let nb = match expr_to_num(eb) {
            Some(n) => n,
            None => {
              return Ok(Expr::FunctionCall {
                name: name.to_string(),
                args: args.to_vec().into(),
              });
            }
          };
          if allow_equal {
            if na > nb {
              return Ok(Expr::Identifier("False".to_string()));
            }
          } else if na >= nb {
            return Ok(Expr::Identifier("False".to_string()));
          }
        }
      }
      // Both are scalars
      _ => {
        let na = match expr_to_num(a) {
          Some(n) => n,
          None => {
            return Ok(Expr::FunctionCall {
              name: name.to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let nb = match expr_to_num(b) {
          Some(n) => n,
          None => {
            return Ok(Expr::FunctionCall {
              name: name.to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        if allow_equal {
          if na > nb {
            return Ok(Expr::Identifier("False".to_string()));
          }
        } else if na >= nb {
          return Ok(Expr::Identifier("False".to_string()));
        }
      }
    }
  }

  Ok(Expr::Identifier("True".to_string()))
}

// ---------------------------------------------------------------------------
// BooleanVariables / SatisfiableQ / SatisfiabilityCount
// (SatisfiabilityInstances is not implemented: wolframscript's instance
// ordering and free-variable assignments follow its internal BDD structure,
// which is expression-dependent and not reproducible black-box.)

/// Collect the Boolean variables of `expr` as expressions: anything that is
/// not a logical connective, True/False, or a list counts as a variable —
/// including opaque subexpressions like f[b] (matching wolframscript).
fn collect_boolean_variable_exprs(expr: &Expr, out: &mut Vec<Expr>) {
  let push = |e: &Expr, out: &mut Vec<Expr>| {
    if !out
      .iter()
      .any(|v| crate::evaluator::pattern_matching::expr_equal(v, e))
    {
      out.push(e.clone());
    }
  };
  match expr {
    Expr::Identifier(name) if name == "True" || name == "False" => {}
    Expr::Identifier(_) => push(expr, out),
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "And"
          | "Or"
          | "Not"
          | "Xor"
          | "Xnor"
          | "Nand"
          | "Nor"
          | "Implies"
          | "Equivalent"
          | "Majority"
      ) =>
    {
      for arg in args.iter() {
        collect_boolean_variable_exprs(arg, out);
      }
    }
    Expr::List(items) => {
      for item in items.iter() {
        collect_boolean_variable_exprs(item, out);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::And | crate::syntax::BinaryOperator::Or,
      left,
      right,
    } => {
      collect_boolean_variable_exprs(left, out);
      collect_boolean_variable_exprs(right, out);
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Not,
      operand,
    } => {
      collect_boolean_variable_exprs(operand, out);
    }
    _ => push(expr, out),
  }
}

/// Replace every occurrence of the subexpression `var` in `expr` by `val`.
fn substitute_boolean_var(expr: &Expr, var: &Expr, val: &Expr) -> Expr {
  if crate::evaluator::pattern_matching::expr_equal(expr, var) {
    return val.clone();
  }
  match expr {
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_boolean_var(a, var, val))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_boolean_var(left, var, val)),
      right: Box::new(substitute_boolean_var(right, var, val)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_boolean_var(operand, var, val)),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|i| substitute_boolean_var(i, var, val))
        .collect::<Vec<_>>()
        .into(),
    ),
    _ => expr.clone(),
  }
}

/// BooleanVariables[expr] - the Boolean variables, canonically sorted.
pub fn boolean_variables_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "BooleanVariables".to_string(),
      args: args.to_vec().into(),
    });
  }
  let mut vars = Vec::new();
  collect_boolean_variable_exprs(&args[0], &mut vars);
  // compare_exprs follows Wolfram's Order convention: 1 means a sorts first
  vars.sort_by(|a, b| {
    (-crate::functions::list_helpers_ast::compare_exprs(a, b)).cmp(&0)
  });
  Ok(Expr::List(vars.into()))
}

/// Resolve the variable list for the 1- or 2-argument satisfiability forms.
fn satisfiability_vars(args: &[Expr]) -> Option<Vec<Expr>> {
  if args.len() == 2 {
    match &args[1] {
      Expr::List(items) => Some(items.iter().cloned().collect()),
      single => Some(vec![single.clone()]),
    }
  } else {
    let mut vars = Vec::new();
    collect_boolean_variable_exprs(&args[0], &mut vars);
    vars.sort_by(|a, b| {
      (-crate::functions::list_helpers_ast::compare_exprs(a, b)).cmp(&0)
    });
    Some(vars)
  }
}

/// Enumerate assignments (all-True first, descending) and classify each
/// evaluation as True / False / non-Boolean. Returns (true_count, first
/// non-Boolean assignment if any).
fn count_satisfying(
  expr: &Expr,
  vars: &[Expr],
) -> Result<(u64, Option<Vec<bool>>), InterpreterError> {
  let n = vars.len();
  let mut count = 0u64;
  for idx in 0..(1u64 << n) {
    let assignment: Vec<bool> =
      (0..n).map(|j| (idx >> (n - 1 - j)) & 1 == 0).collect();
    let mut substituted = expr.clone();
    for (var, &b) in vars.iter().zip(&assignment) {
      let val = Expr::Identifier(if b { "True" } else { "False" }.to_string());
      substituted = substitute_boolean_var(&substituted, var, &val);
    }
    let result = evaluate_expr_to_expr(&substituted)?;
    match &result {
      Expr::Identifier(s) if s == "True" => count += 1,
      Expr::Identifier(s) if s == "False" => {}
      _ => return Ok((count, Some(assignment))),
    }
  }
  Ok((count, None))
}

/// SatisfiableQ[expr] / SatisfiableQ[expr, vars]
pub fn satisfiable_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "SatisfiableQ".to_string(),
    args: args.to_vec().into(),
  };
  let vars = match satisfiability_vars(args) {
    Some(v) if v.len() <= 24 => v,
    _ => return Ok(unevaluated()),
  };
  // Short-circuiting variant of count_satisfying: stop at the first True.
  let n = vars.len();
  for idx in 0..(1u64 << n) {
    let assignment: Vec<bool> =
      (0..n).map(|j| (idx >> (n - 1 - j)) & 1 == 0).collect();
    let mut substituted = args[0].clone();
    for (var, &b) in vars.iter().zip(&assignment) {
      let val = Expr::Identifier(if b { "True" } else { "False" }.to_string());
      substituted = substitute_boolean_var(&substituted, var, &val);
    }
    let result = evaluate_expr_to_expr(&substituted)?;
    match &result {
      Expr::Identifier(s) if s == "True" => {
        return Ok(Expr::Identifier("True".to_string()));
      }
      Expr::Identifier(s) if s == "False" => {}
      _ => {
        let shown = Expr::List(
          assignment
            .iter()
            .map(|&b| {
              Expr::Identifier(if b { "True" } else { "False" }.to_string())
            })
            .collect::<Vec<_>>()
            .into(),
        );
        crate::emit_message(&format!(
          "SatisfiableQ::boolv: {} is not Boolean valued at {}.",
          crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output),
          crate::syntax::format_expr(&shown, crate::syntax::ExprForm::Output)
        ));
        return Ok(unevaluated());
      }
    }
  }
  Ok(Expr::Identifier("False".to_string()))
}

/// SatisfiabilityCount[expr] / SatisfiabilityCount[expr, vars]
pub fn satisfiability_count_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "SatisfiabilityCount".to_string(),
    args: args.to_vec().into(),
  };
  let vars = match satisfiability_vars(args) {
    Some(v) if v.len() <= 24 => v,
    _ => return Ok(unevaluated()),
  };
  match count_satisfying(&args[0], &vars)? {
    (count, None) => Ok(Expr::Integer(count as i128)),
    // Non-Boolean evaluation: stay unevaluated (wolframscript is silent
    // here, unlike SatisfiableQ's boolv message)
    (_, Some(_)) => Ok(unevaluated()),
  }
}


/// Majority[b1, b2, ...] - True when more than half the arguments are True.
/// Symbolic arguments simplify by cancelling True/False pairs and deciding
/// early when the outcome no longer depends on the unknowns (matching
/// wolframscript: Majority[True, False, x] -> x, Majority[a, b] -> a && b).
pub fn majority_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = args.len();
  let trues = args
    .iter()
    .filter(|a| matches!(a, Expr::Identifier(s) if s == "True"))
    .count();
  let falses = args
    .iter()
    .filter(|a| matches!(a, Expr::Identifier(s) if s == "False"))
    .count();
  let unknowns = n - trues - falses;
  // Decided regardless of the unknowns
  if 2 * trues > n {
    return Ok(Expr::Identifier("True".to_string()));
  }
  if 2 * (trues + unknowns) <= n {
    return Ok(Expr::Identifier("False".to_string()));
  }
  // Cancel one True/False pair and re-evaluate
  if trues >= 1 && falses >= 1 {
    let mut rest: Vec<Expr> = args.to_vec();
    let ti = rest
      .iter()
      .position(|a| matches!(a, Expr::Identifier(s) if s == "True"))
      .unwrap();
    rest.remove(ti);
    let fi = rest
      .iter()
      .position(|a| matches!(a, Expr::Identifier(s) if s == "False"))
      .unwrap();
    rest.remove(fi);
    return majority_ast(&rest);
  }
  // Purely symbolic simplifications
  if n == 1 {
    return Ok(args[0].clone());
  }
  if n == 2 && unknowns == 2 {
    return evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "And".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Majority is Orderless: canonically sort the remaining arguments
  let mut sorted = args.to_vec();
  sorted.sort_by(|a, b| {
    (-crate::functions::list_helpers_ast::compare_exprs(a, b)).cmp(&0)
  });
  Ok(Expr::FunctionCall {
    name: "Majority".to_string(),
    args: sorted.into(),
  })
}