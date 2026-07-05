#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

pub fn n_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "N expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    // N[expr, MachinePrecision] is identical to N[expr] — return a
    // machine-precision Real, not a BigFloat.
    if matches!(&args[1], Expr::Identifier(s) if s == "MachinePrecision") {
      return n_eval(&args[0]);
    }
    // N[expr, precision] — arbitrary-precision evaluation. Precision
    // can be a numeric expression (e.g. N[Pi, Pi] uses ≈3.14159… as
    // the precision marker); we preserve the full f64 value so the
    // emitted BigFloat carries the exact precision tag wolframscript
    // shows.
    let precision_f64 = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as f64,
      other => {
        if let Some(v) = try_eval_to_f64(other)
          && v.floor() as i128 > 0
        {
          v
        } else {
          crate::emit_message(&format!(
            "N::precbd: Requested precision {} is not a machine-sized real number between $MinPrecision and $MaxPrecision.",
            crate::syntax::expr_to_string(other)
          ));
          return Ok(Expr::FunctionCall {
            name: "N".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    };
    return n_eval_arbitrary(&args[0], precision_f64);
  }
  n_eval(&args[0])
}

/// Recursively convert an expression to numeric (Real) form
pub fn n_eval(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::Real(_) => Ok(expr.clone()),
    // N[BigFloat] without an explicit precision collapses to a
    // machine-precision Real, matching wolframscript:
    // `N[1.01234567890123456789]` → `1.0123456789012346`.
    Expr::BigFloat(digits, _) => {
      if let Ok(f) = digits.parse::<f64>() {
        Ok(Expr::Real(f))
      } else {
        Ok(expr.clone())
      }
    }
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items.iter().map(n_eval).collect();
      Ok(Expr::List(results?.into()))
    }
    Expr::BigInteger(n) => Ok(Expr::Real(
      n.to_string().parse::<f64>().unwrap_or(f64::INFINITY),
    )),
    Expr::FunctionCall { name, args } => {
      // First try to evaluate the whole expression to a number
      if let Some(v) = try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      // N[Integrate[expr, {var, a, b}]] — when symbolic Integrate
      // didn't simplify (e.g. Abs[Sin[phi]]), fall back to NIntegrate.
      // Matches wolframscript's behaviour of computing the value
      // numerically rather than returning the unevaluated form.
      if name == "Integrate"
        && args.len() == 2
        && let Expr::List(spec) = &args[1]
        && spec.len() == 3
        && matches!(&spec[0], Expr::Identifier(_))
        && let Ok(r) = crate::functions::calculus_ast::nintegrate_ast(args)
        && matches!(&r, Expr::Real(_) | Expr::Integer(_))
      {
        return Ok(r);
      }
      // Special case for functions that stay symbolic when called
      // directly with a Real but have a numeric value triggered by N[].
      if args.len() == 1
        && let Some(n) = expr_to_i128(&args[0])
      {
        if name == "AiryAiZero"
          && let Some(r) = crate::functions::math_ast::airy_ai_zero_n_eval(n)
        {
          return Ok(r);
        }
        if name == "AiryBiZero"
          && let Some(r) = crate::functions::math_ast::airy_bi_zero_n_eval(n)
        {
          return Ok(r);
        }
      }
      // RootSum[poly &, fn &] — apply fn to each (complex) root of poly
      // and sum, returning a machine-precision Real (or Complex when the
      // imaginary part doesn't cancel).
      if name == "RootSum"
        && args.len() == 2
        && let Some(r) = root_sum_n_eval(&args[0], &args[1])
      {
        return Ok(r);
      }
      // Root[poly &, k] / Root[poly &, k, 0] — the k-th root numerically.
      if name == "Root"
        && (args.len() == 2 || args.len() == 3)
        && let Some(r) = root_n_eval(&args[0], &args[1])
      {
        return Ok(r);
      }
      // Honour NHoldAll / NHoldFirst / NHoldRest attributes (built-in or
      // user-set). When a slot is held, leave the argument literal and
      // skip the recursive N application.
      let attrs: Vec<String> = {
        let builtin: Vec<String> =
          crate::evaluator::get_builtin_attributes(name)
            .into_iter()
            .map(String::from)
            .collect();
        let user = crate::FUNC_ATTRS
          .with(|m| m.borrow().get(name).cloned().unwrap_or_default());
        let mut combined = builtin;
        for a in user {
          if !combined.contains(&a) {
            combined.push(a);
          }
        }
        combined
      };
      let hold_all = attrs.iter().any(|a| a == "NHoldAll");
      let hold_first = attrs.iter().any(|a| a == "NHoldFirst");
      let hold_rest = attrs.iter().any(|a| a == "NHoldRest");
      let new_args: Vec<Expr> = if hold_all {
        args.to_vec()
      } else {
        let mut out = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
          let held = (i == 0 && hold_first) || (i > 0 && hold_rest);
          if held {
            out.push(a.clone());
          } else {
            out.push(n_eval(a)?);
          }
        }
        out
      };
      // Re-evaluate the function with numeric arguments
      let new_expr = Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      };
      match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => Ok(result),
        Err(_) => Ok(new_expr),
      }
    }
    Expr::BinaryOp { op, left, right } => {
      if let Some(v) = try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      let new_expr = Expr::BinaryOp {
        op: *op,
        left: Box::new(n_eval(left)?),
        right: Box::new(n_eval(right)?),
      };
      // Re-evaluate to allow numeric simplification (e.g. complex powers)
      let result = match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => result,
        Err(_) => new_expr,
      };
      // If the result is still a Power with complex operands, force
      // numeric evaluation via z^w = exp(w * log(z))
      if let Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: ref base,
        right: ref exp,
      } = result
        && let (Some((a, b)), Some((c, d))) = (
          try_extract_complex_float(base),
          try_extract_complex_float(exp),
        )
        && (b != 0.0 || d != 0.0)
      {
        let abs_z = (a * a + b * b).sqrt();
        if abs_z > 0.0 {
          let ln_abs = abs_z.ln();
          let arg_z = b.atan2(a);
          let re_exp = c * ln_abs - d * arg_z;
          let im_exp = d * ln_abs + c * arg_z;
          let mag = re_exp.exp();
          let re = mag * im_exp.cos();
          let im = mag * im_exp.sin();
          let re = if re.abs() < 1e-15 { 0.0 } else { re };
          let im = if im.abs() < 1e-15 { 0.0 } else { im };
          if im == 0.0 {
            // Preserve complex form: re + 0.*I (matching Wolfram's convention
            // for complex power results where imaginary part is numerically zero)
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(Expr::Real(re)),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Real(0.0)),
                right: Box::new(Expr::Identifier("I".to_string())),
              }),
            });
          }
          return Ok(build_complex_float_expr(re, im));
        }
      }
      Ok(result)
    }
    // Rule: keep the pattern, apply N to the replacement
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(n_eval(replacement)?),
    }),
    Expr::Identifier(_) | Expr::Constant(_) => {
      if let Some(v) = try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      // Look up user-installed `N[sym, …] = value` rules registered
      // under the symbol's NValues. Wolfram stores these per-symbol
      // (not as a DownValue of `N`), so this lookup is needed before
      // falling back to the generic `N` dispatch. `n_eval` is the
      // machine-precision path, so prefer the
      // `N[sym, {MachinePrecision, MachinePrecision}]` entry.
      if let Expr::Identifier(name) = expr {
        let nval = crate::evaluator::assignment::N_VALUES
          .with(|m| m.borrow().get(name).cloned());
        if let Some(entries) = nval {
          // The canonical Wolfram form is
          // `N[sym, {MachinePrecision, MachinePrecision}]`, but
          // user-supplied `NValues[sym] := {N[sym, MachinePrecision]
          // :> v}` rules also need to fire here. Accept both shapes:
          //   N[sym, {MachinePrecision, MachinePrecision}]
          //   N[sym, MachinePrecision]
          // wolframscript does NOT fire a bare `N[sym]` LHS — that
          // pattern doesn't match the internal `N[sym, prec]` call —
          // so a user-supplied `NValues[sym] := {N[sym] :> v}` is
          // silently ignored at lookup. Mirror that.
          let is_machine_precision_lhs = |lhs_p: &Expr| -> bool {
            let Expr::FunctionCall { name: ln, args: la } = lhs_p else {
              return false;
            };
            if ln != "N" || la.len() != 2 {
              return false;
            }
            match &la[1] {
              Expr::Identifier(s) if s == "MachinePrecision" => true,
              Expr::List(prec) => {
                prec.len() == 2
                  && matches!(&prec[0],
                    Expr::Identifier(s) if s == "MachinePrecision")
              }
              _ => false,
            }
          };
          for (lhs_p, rhs) in &entries {
            if is_machine_precision_lhs(lhs_p) || is_blank_precision_lhs(lhs_p)
            {
              return n_eval(rhs);
            }
          }
        }
      }
      // No built-in numeric value for this symbol — try to invoke
      // `N[expr]` so any user-installed `N[a] = 10.9` style DownValue
      // gets a chance to fire (matching Wolfram's NValues lookup).
      // Skip when no user `N` DownValues exist: the built-in dispatch for
      // `N` is `n_ast`, which would call back into `n_eval` here — an
      // infinite loop that exhausts WASM's smaller stack before the
      // RECURSION_LIMIT guard trips.
      let has_user_n_rules = crate::FUNC_DEFS
        .with(|m| m.borrow().get("N").is_some_and(|v| !v.is_empty()));
      if !has_user_n_rules {
        return Ok(expr.clone());
      }
      let original_str = crate::syntax::expr_to_string(expr);
      let n_call_str = format!("N[{}]", original_str);
      match crate::evaluator::evaluate_function_call_ast("N", &[expr.clone()]) {
        Ok(result) => {
          let result_str = crate::syntax::expr_to_string(&result);
          if result_str == n_call_str {
            Ok(expr.clone())
          } else {
            Ok(result)
          }
        }
        Err(_) => Ok(expr.clone()),
      }
    }
    _ => {
      if let Some(v) = try_eval_to_f64(expr) {
        Ok(Expr::Real(v))
      } else {
        Ok(expr.clone())
      }
    }
  }
}

/// Convert decimal digit precision to the nominal bit-precision.
/// `astro-float` internally rounds up to 64-bit word boundaries.
/// Minimum 128 bits (2 words) to avoid precision issues with small values.
pub fn nominal_bits(precision: usize) -> usize {
  // Compute decimal precision plus ~36-bit guard, padded to the next
  // 64-bit word boundary. Mirrors Wolfram's bit budget so output digit
  // counts line up:
  //   p ≤ 8        →  64 bits  → 19 digits
  //   9 ≤ p ≤ 27   → 128 bits  → 38 digits
  //   28 ≤ p ≤ 47  → 192 bits  → 58 digits
  //   48 ≤ p ≤ 66  → 256 bits  → 77 digits
  // We keep a 128-bit minimum because astro-float's 64-bit operations
  // are unstable for some inputs (e.g. division produces convert_to_radix
  // errors).
  let base_bits =
    (precision as f64 * std::f64::consts::LOG2_10).ceil() as usize + 36;
  let bits = (base_bits + 63) & !63;
  bits.max(128)
}

/// Extract an integer value from an Expr, if it is one.
fn try_as_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      if let Expr::Integer(n) = operand.as_ref() {
        Some(-n)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Compute base^n for integer n, handling negative exponents.
/// Uses BigFloat::powi for the absolute value, then inverts if needed.
fn bigfloat_powi(
  base: &astro_float::BigFloat,
  n: i128,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if n == 0 {
    return BigFloat::from_i32(1, bits);
  }

  let abs_n = n.unsigned_abs() as usize;
  let result = base.powi(abs_n, bits, rm);

  if n < 0 {
    BigFloat::from_i32(1, bits).div(&result, bits, rm)
  } else {
    result
  }
}

/// Arbitrary-precision Gamma via Spouge's approximation.
///
/// `Γ(z+1) = (z+a)^(z+1/2) · e^{−(z+a)} · √(2π) · (c₀ + Σ_{k=1}^{a−1} c_k/(z+k))`
///
/// with `c_0 = 1` and `c_k = (-1)^(k-1)/(k-1)! · (a-k)^(k-1/2) · e^(a-k)`.
/// Choosing `a ≈ 1.6 · digits` gives ~`digits` accurate decimals. We
/// shift the input by 1 (so the formula receives `z = input - 1`) and
/// reflect to positive arguments via the reflection identity when
/// `Re(z) ≤ 0`.
fn gamma_bigfloat(
  z_in: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Choose `a` from the working precision. astro-float's bit count is
  // a little above the requested decimal digits, so use log10(2) ≈ 0.301
  // to convert bits → digits and pad by ~1.6.
  let approx_digits = (bits as f64 * std::f64::consts::LOG10_2).ceil() as i64;
  let a_int: i64 = ((approx_digits as f64) * 1.6).ceil() as i64 + 5;
  let a_int = a_int.max(20);
  let a_bf = BigFloat::from_i64(a_int, bits);

  // Reflection: for Re(z) < 0.5 use Γ(z) = π / (sin(π·z) · Γ(1−z)).
  let half = BigFloat::from_f64(0.5, bits);
  let pi = cc.pi(bits, rm);
  if z_in.cmp(&half).is_some_and(|o| o < 0) {
    let one = BigFloat::from_i32(1, bits);
    let one_minus_z = one.sub(z_in, bits, rm);
    let pi_z = pi.mul(z_in, bits, rm);
    let sin_piz = pi_z.sin(bits, rm, cc);
    let gamma_1mz = gamma_bigfloat(&one_minus_z, bits, rm, cc);
    let denom = sin_piz.mul(&gamma_1mz, bits, rm);
    return pi.div(&denom, bits, rm);
  }

  // Work with z = z_in - 1 so the formula computes Γ(z+1) = Γ(z_in).
  let one = BigFloat::from_i32(1, bits);
  let z = z_in.sub(&one, bits, rm);

  // Outer factor: (z + a)^(z + 1/2) · e^{−(z+a)}.
  // Note: the canonical Spouge form has a `√(2π)` multiplier, but it
  // cancels because c_0 in the convention used here is `1` (rather
  // than `√(2π)`) — i.e. the √(2π) is absorbed into the coefficients
  // implicitly. Empirically `outer_pow · exp_term · sum_with_c0=1`
  // recovers Γ(z+1) without an explicit √(2π) factor.
  let z_plus_a = z.add(&a_bf, bits, rm);
  let z_plus_half = z.add(&half, bits, rm);
  let outer_pow = z_plus_a.pow(&z_plus_half, bits, rm, cc);
  let neg_zpa = z_plus_a.neg();
  let exp_term = neg_zpa.exp(bits, rm, cc);
  let outer = outer_pow.mul(&exp_term, bits, rm);

  // Inner sum: c_0 + Σ_{k=1}^{a-1} c_k / (z + k)
  // Compute c_k = (-1)^(k-1) / (k-1)! · (a-k)^(k-1/2) · e^(a-k).
  let mut sum = BigFloat::from_i32(1, bits); // c_0
  let mut factorial = BigFloat::from_i32(1, bits); // (k-1)! starting at k=1
  for k in 1..a_int {
    if k > 1 {
      factorial = factorial.mul(&BigFloat::from_i64(k - 1, bits), bits, rm);
    }
    let a_minus_k = BigFloat::from_i64(a_int - k, bits);
    let half_offset = BigFloat::from_i64(k - 1, bits).add(&half, bits, rm);
    // (a-k)^(k - 1/2)
    let term_pow = a_minus_k.pow(&half_offset, bits, rm, cc);
    let exp_amk = a_minus_k.exp(bits, rm, cc);
    let mut c_k = term_pow.mul(&exp_amk, bits, rm).div(&factorial, bits, rm);
    if k % 2 == 0 {
      c_k = c_k.neg();
    }
    let z_plus_k = z.add(&BigFloat::from_i64(k, bits), bits, rm);
    let term = c_k.div(&z_plus_k, bits, rm);
    sum = sum.add(&term, bits, rm);
  }

  outer.mul(&sum, bits, rm)
}

/// Convert a BigFloat to a decimal string.
/// If `max_fraction_digits` is Some(n), keep at most `n` digits AFTER the
/// decimal point. The integer part (when |x| ≥ 1) is preserved in full, so
/// e.g. with `n = 58`, Sqrt[2] (exp = 1) yields 1 + 58 = 59 total digits
/// while Sin[1] (exp = 0) yields 0 + 58 = 58 total digits — matching how
/// Wolfram caps display digits relative to the decimal point.
pub fn bigfloat_to_string(
  bf: &astro_float::BigFloat,
  max_fraction_digits: Option<usize>,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<String, InterpreterError> {
  if bf.is_zero() {
    return Ok("0.".to_string());
  }

  let (sign, digits, exponent) = bf
    .convert_to_radix(astro_float::Radix::Dec, rm, cc)
    .map_err(|e| {
      InterpreterError::EvaluationError(format!("N: format error: {}", e))
    })?;

  if digits.is_empty() || digits.iter().all(|&d| d == 0) {
    return Ok("0.".to_string());
  }

  let is_negative = sign == astro_float::Sign::Neg;
  let prefix = if is_negative { "-" } else { "" };

  // Convert digit values to ASCII string
  let digit_str: String = digits.iter().map(|&d| (b'0' + d) as char).collect();

  // Round to (fraction_cap + max(exp, 0)) total digits if requested. This
  // gives Wolfram's display: a fixed number of digits AFTER the dot,
  // independent of the value's magnitude. Rounding (not truncation) the
  // last shown digit matches Wolfram. Carries that ripple past the leading
  // digit increment the decimal exponent.
  let (digit_str, decimal_exp) = if let Some(max_frac) = max_fraction_digits {
    let exp_i64 = exponent as i64;
    // total_keep = max_frac + max(exp, 0). For exp ≤ 0 the number is < 1
    // and all digit_str entries are fractional, so we just keep max_frac.
    // (Leading zeros from a negative exponent come from the format step
    // and don't draw from digit_str.) For exp > 0 the integer part needs
    // `exp` extra digits on top of the fraction cap.
    let total_keep = (max_frac as i64 + exp_i64.max(0)) as usize;
    if digit_str.len() > total_keep && total_keep > 0 {
      let kept = &digit_str[..total_keep];
      let next_digit = digit_str.as_bytes()[total_keep] - b'0';
      let mut bytes: Vec<u8> = kept.bytes().collect();
      let mut carry = if next_digit >= 5 { 1u8 } else { 0u8 };
      // Round-half-up. (Wolfram uses banker's rounding internally, but the
      // BigFloat is already at higher precision than `total_keep` so the
      // next digit is rarely exactly 5 with no further nonzero digits.)
      for b in bytes.iter_mut().rev() {
        if carry == 0 {
          break;
        }
        let v = *b - b'0' + carry;
        if v >= 10 {
          *b = b'0';
          carry = 1;
        } else {
          *b = b'0' + v;
          carry = 0;
        }
      }
      let new_exp = if carry == 1 {
        // All 9s rolled over: prepend a 1, drop the trailing 0, bump exponent.
        bytes.pop();
        bytes.insert(0, b'1');
        exp_i64 + 1
      } else {
        exp_i64
      };
      let s: String = String::from_utf8(bytes).expect("ascii digits");
      (s, new_exp)
    } else {
      (digit_str, exp_i64)
    }
  } else {
    (digit_str, exponent as i64)
  };

  if decimal_exp <= 0 {
    // Number like 0.000xxxx
    let zeros = (-decimal_exp) as usize;
    let trimmed = digit_str.trim_end_matches('0');
    if trimmed.is_empty() {
      Ok(format!("{}0.", prefix))
    } else {
      let frac = format!("{}{}", "0".repeat(zeros), trimmed);
      let frac = frac.trim_end_matches('0');
      if frac.is_empty() {
        Ok(format!("{}0.", prefix))
      } else {
        Ok(format!("{}0.{}", prefix, frac))
      }
    }
  } else {
    let dp = decimal_exp as usize;
    if dp >= digit_str.len() {
      // All digits are in the integer part
      let padded =
        format!("{}{}", &digit_str, "0".repeat(dp - digit_str.len()));
      Ok(format!("{}{}.", prefix, padded))
    } else {
      // Some digits before decimal, some after
      let int_part = &digit_str[..dp];
      let frac_part = digit_str[dp..].trim_end_matches('0');
      if frac_part.is_empty() {
        Ok(format!("{}{}.", prefix, int_part))
      } else {
        Ok(format!("{}{}.{}", prefix, int_part, frac_part))
      }
    }
  }
}

/// SetPrecision[expr, precision] — set every numeric leaf of `expr` to a
/// fixed precision without otherwise evaluating the expression.
///
/// Unlike `N`, SetPrecision walks the expression tree symbolically: numeric
/// leaves are converted to BigFloat (or, with MachinePrecision, demoted to
/// a machine-precision Real), while symbolic heads and identifiers are
/// preserved verbatim.
pub fn set_precision_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "SetPrecision".to_string(),
      args: args.to_vec().into(),
    });
  }
  let expr = &args[0];
  let target = &args[1];

  if matches!(target, Expr::Identifier(s) if s == "MachinePrecision") {
    return set_precision_machine(expr);
  }

  let precision_f64 = match target {
    Expr::Integer(n) if *n > 0 => *n as f64,
    other => {
      if let Some(v) = try_eval_to_f64(other)
        && v > 0.0
      {
        v
      } else {
        return Ok(Expr::FunctionCall {
          name: "SetPrecision".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  };

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  let rm = RoundingMode::ToEven;
  let prec_usize = precision_f64.floor().max(1.0) as usize;
  let bits = nominal_bits(prec_usize) + 64;

  // Truncate the displayed decimal to the digit count wolframscript uses
  // for this precision tier (matches N's existing logic).
  let display_bits = {
    let b = (precision_f64 * std::f64::consts::LOG2_10).ceil() as usize + 36;
    ((b + 63) & !63).max(64)
  };
  let max_fraction_digits =
    ((display_bits as f64 + 1.0) * std::f64::consts::LOG10_2).floor() as usize;

  set_precision_walk(
    expr,
    precision_f64,
    bits,
    Some(max_fraction_digits),
    rm,
    &mut cc,
  )
}

/// Walk `expr` and convert every numeric leaf to a BigFloat at `precision`.
fn set_precision_walk(
  expr: &Expr,
  precision: f64,
  bits: usize,
  max_fraction_digits: Option<usize>,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => {
      leaf_to_bigfloat(expr, precision, bits, max_fraction_digits, rm, cc)
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      leaf_to_bigfloat(expr, precision, bits, max_fraction_digits, rm, cc)
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items.iter() {
        out.push(set_precision_walk(
          item,
          precision,
          bits,
          max_fraction_digits,
          rm,
          cc,
        )?);
      }
      Ok(Expr::List(out.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut out = Vec::with_capacity(args.len());
      for a in args.iter() {
        out.push(set_precision_walk(
          a,
          precision,
          bits,
          max_fraction_digits,
          rm,
          cc,
        )?);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let l =
        set_precision_walk(left, precision, bits, max_fraction_digits, rm, cc)?;
      let r = set_precision_walk(
        right,
        precision,
        bits,
        max_fraction_digits,
        rm,
        cc,
      )?;
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let o = set_precision_walk(
        operand,
        precision,
        bits,
        max_fraction_digits,
        rm,
        cc,
      )?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    // Symbolic real constants (Pi, E, Degree, GoldenRatio, EulerGamma, …)
    // numericize to the requested precision, matching N[c, p]:
    // SetPrecision[Pi, 5] -> 3.1415926535897932385`5.. expr_to_bigfloat
    // errors for a bare symbol (e.g. x), so those pass through unchanged.
    Expr::Constant(_) | Expr::Identifier(_) => {
      leaf_to_bigfloat(expr, precision, bits, max_fraction_digits, rm, cc)
        .or_else(|_| Ok(expr.clone()))
    }
    _ => Ok(expr.clone()),
  }
}

/// Convert a numeric leaf to a BigFloat at the requested precision.
fn leaf_to_bigfloat(
  expr: &Expr,
  precision: f64,
  bits: usize,
  max_fraction_digits: Option<usize>,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let bf = expr_to_bigfloat(expr, bits, rm, cc)?;
  let decimal = bigfloat_to_string(&bf, max_fraction_digits, rm, cc)?;
  Ok(Expr::BigFloat(decimal, precision))
}

/// SetAccuracy[expr, accuracy] — set every numeric leaf of `expr` to a fixed
/// accuracy (absolute uncertainty 10^-accuracy). For a leaf of magnitude |v|
/// this is a precision of `accuracy + Log10[|v|]`; a zero leaf becomes the
/// accuracy-form `0``accuracy`. Mirrors SetPrecision but the precision is
/// computed per leaf from its magnitude.
pub fn set_accuracy_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "SetAccuracy".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated());
  }
  let accuracy = match &args[1] {
    Expr::Integer(n) => *n as f64,
    other => match try_eval_to_f64(other) {
      Some(v) => v,
      None => return Ok(unevaluated()),
    },
  };

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  set_accuracy_walk(&args[0], accuracy, RoundingMode::ToEven, &mut cc)
}

/// Walk `expr`, setting every numeric leaf to `accuracy` (see `set_accuracy_ast`).
fn set_accuracy_walk(
  expr: &Expr,
  accuracy: f64,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => {
      leaf_to_bigfloat_at_accuracy(expr, accuracy, rm, cc)
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      leaf_to_bigfloat_at_accuracy(expr, accuracy, rm, cc)
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items.iter() {
        out.push(set_accuracy_walk(item, accuracy, rm, cc)?);
      }
      Ok(Expr::List(out.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut out = Vec::with_capacity(args.len());
      for a in args.iter() {
        out.push(set_accuracy_walk(a, accuracy, rm, cc)?);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => Ok(Expr::BinaryOp {
      op: *op,
      left: Box::new(set_accuracy_walk(left, accuracy, rm, cc)?),
      right: Box::new(set_accuracy_walk(right, accuracy, rm, cc)?),
    }),
    Expr::UnaryOp { op, operand } => Ok(Expr::UnaryOp {
      op: *op,
      operand: Box::new(set_accuracy_walk(operand, accuracy, rm, cc)?),
    }),
    // Symbolic real constants (Pi, E, …) numericize like SetPrecision; bare
    // symbols pass through unchanged.
    Expr::Constant(_) | Expr::Identifier(_) => {
      leaf_to_bigfloat_at_accuracy(expr, accuracy, rm, cc)
        .or_else(|_| Ok(expr.clone()))
    }
    _ => Ok(expr.clone()),
  }
}

/// Convert a numeric leaf to a BigFloat at the given absolute `accuracy`.
fn leaf_to_bigfloat_at_accuracy(
  expr: &Expr,
  accuracy: f64,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let v = try_eval_to_f64(expr).ok_or_else(|| {
    InterpreterError::EvaluationError("non-numeric leaf".to_string())
  })?;
  if v == 0.0 {
    // Zero has no relative precision; it carries the accuracy directly in the
    // `0``accuracy` form (stored as BigFloat with digits "0").
    return Ok(Expr::BigFloat("0".to_string(), accuracy));
  }
  let precision = accuracy + v.abs().log10();
  let prec_usize = precision.floor().max(1.0) as usize;
  let bits = nominal_bits(prec_usize) + 64;
  let display_bits = {
    let b = (precision * std::f64::consts::LOG2_10).ceil() as usize + 36;
    ((b + 63) & !63).max(64)
  };
  let max_fraction_digits =
    ((display_bits as f64 + 1.0) * std::f64::consts::LOG10_2).floor() as usize;
  leaf_to_bigfloat(expr, precision, bits, Some(max_fraction_digits), rm, cc)
}

/// SetPrecision[..., MachinePrecision] — walk the tree and demote numeric
/// leaves to machine-precision Reals; symbolic parts stay as-is.
fn set_precision_machine(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::BigInteger(n) => Ok(Expr::Real(
      n.to_string().parse::<f64>().unwrap_or(f64::INFINITY),
    )),
    Expr::Real(_) => Ok(expr.clone()),
    Expr::BigFloat(digits, _) => {
      Ok(Expr::Real(digits.parse::<f64>().unwrap_or(f64::NAN)))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Some(p), Some(q)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
      {
        Ok(Expr::Real(p / q))
      } else {
        Ok(expr.clone())
      }
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items.iter() {
        out.push(set_precision_machine(item)?);
      }
      Ok(Expr::List(out.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut out = Vec::with_capacity(args.len());
      for a in args.iter() {
        out.push(set_precision_machine(a)?);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let l = set_precision_machine(left)?;
      let r = set_precision_machine(right)?;
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let o = set_precision_machine(operand)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    _ => Ok(expr.clone()),
  }
}

/// N[expr, precision] — arbitrary-precision numeric evaluation using BigFloat
pub fn n_eval_arbitrary(
  expr: &Expr,
  precision: f64,
) -> Result<Expr, InterpreterError> {
  // Handle List recursively at the Expr level
  if let Expr::List(items) = expr {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|e| n_eval_arbitrary(e, precision))
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  // Floor of the precision tag is what bit/digit budgets are sized
  // from — astro-float can't operate on a fractional precision, but
  // we still want the resulting BigFloat to carry the original f64
  // precision marker.
  let prec_usize = precision.floor().max(1.0) as usize;

  // Machine-precision Reals already live at MachinePrecision and cannot be
  // promoted by N — wolframscript returns the Real unchanged.
  if let Expr::Real(_) = expr {
    return Ok(expr.clone());
  }

  // BigFloat input with already-tracked precision: wolframscript clamps the
  // result precision to the input's. `N[1.012345...123, 50]` keeps the
  // 24-digit-equivalent precision marker the literal carried in, rather
  // than fabricating extra digits we don't have. Re-emit the same digits
  // with the original precision marker.
  if let Expr::BigFloat(digits, prec_f64) = expr {
    let prec_floor = if *prec_f64 > 0.0 { *prec_f64 } else { 0.0 };
    if precision > prec_floor {
      return Ok(Expr::BigFloat(digits.clone(), prec_floor));
    }
  }

  use astro_float::{Consts, RoundingMode};

  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  let rm = RoundingMode::ToEven;

  // Compute at one extra 64-bit word above the nominal display precision
  // so astro-float's `convert_to_radix` always emits at least
  // `max_display_digits` decimal digits — without the extra word, e.g. a
  // 192-bit Sqrt[2] only yields 58 digits and N[Sqrt[2], 40] truncates one
  // digit short of Wolfram's 59-digit display.
  let bits = nominal_bits(prec_usize) + 64;

  // Wolfram displays digit counts that match a 64-bit-aligned bit count
  // computed from precision plus a ~36-bit guard. We always evaluate with
  // at least 128 bits internally for stability, but truncate the displayed
  // decimal string so the digit count tracks Wolfram's:
  //   p ≤ 8        → 20 digits  (64-bit-equivalent)
  //   9 ≤ p ≤ 28   → 39 digits  (128-bit)
  //   29 ≤ p ≤ 47  → 59 digits  (192-bit)
  //   etc.
  let display_bits = {
    let b = (precision * std::f64::consts::LOG2_10).ceil() as usize + 36;
    ((b + 63) & !63).max(64)
  };
  // Match Wolfram's display digit counts. The number of fractional digits
  // shown is fixed per bit budget (e.g. 58 for 192 bits), regardless of
  // the value's magnitude. `bigfloat_to_string` adds the integer-part
  // digits on top when the exponent is positive, so e.g. N[Sqrt[2], 40]
  // displays 1 integer digit + 58 fractional digits = 59 total, while
  // N[Sin[1], 40] displays "0.<58 digits>".
  //   64 → 19, 128 → 38, 192 → 58, 256 → 77, 320 → 96, 384 → 115, ...
  let max_fraction_digits =
    ((display_bits as f64 + 1.0) * std::f64::consts::LOG10_2).floor() as usize;

  // Try full conversion to BigFloat first (fast path for purely numeric expressions)
  match expr_to_bigfloat(expr, bits, rm, &mut cc) {
    Ok(result) => {
      let decimal =
        bigfloat_to_string(&result, Some(max_fraction_digits), rm, &mut cc)?;
      Ok(Expr::BigFloat(decimal, precision))
    }
    Err(_) => {
      // Try complex BigFloat evaluation (handles expressions with I)
      if let Ok((re, im)) = expr_to_complex_bigfloat(expr, bits, rm, &mut cc) {
        // For complex function results, compute per-component precision markers
        if let Expr::FunctionCall { name: _, args } = expr
          && !im.is_zero()
          && args.len() == 1
          && let Ok(input_complex) =
            expr_to_complex_bigfloat(&args[0], bits, rm, &mut cc)
          && let Ok((prec_re_str, prec_im_str)) =
            compute_complex_precision_markers(
              &input_complex.0,
              &input_complex.1,
              &re,
              &im,
              prec_usize,
              rm,
              &mut cc,
            )
        {
          return build_complex_result_with_string_precision(
            re,
            im,
            &prec_re_str,
            &prec_im_str,
            prec_usize,
            rm,
            &mut cc,
          );
        }
        return build_complex_bigfloat_result(re, im, prec_usize, rm, &mut cc);
      }
      // Fall back to partial evaluation: convert numeric sub-expressions
      // to arbitrary precision while leaving symbolic parts as-is
      n_eval_arbitrary_partial(expr, precision, bits, rm, &mut cc)
    }
  }
}

/// Recursively apply arbitrary-precision evaluation to sub-expressions.
/// Numeric parts are converted to BigFloat; symbolic parts are left as-is.
fn n_eval_arbitrary_partial(
  expr: &Expr,
  precision: f64,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  // A machine-precision Real already carries the maximum information
  // an f64 can hold — N cannot recover more digits from it, so leave
  // it alone (matches wolframscript: `N[F[1.2, 2/9], $MachinePrecision]`
  // keeps the `1.2` literal and only expands the rational `2/9`).
  if let Expr::Real(_) = expr {
    return Ok(expr.clone());
  }
  // BigFloat input with a lower precision tag: keep the original tag
  // — N can't manufacture digits the input didn't carry. Matches
  // wolframscript: `N[F[1.2\`3, 2/9], 5]` keeps the `1.2\`3` literal at
  // precision 3 while promoting only the rational to precision 5.
  if let Expr::BigFloat(_, prec_f64) = expr {
    let prec_in = if *prec_f64 > 0.0 { *prec_f64 } else { 0.0 };
    if precision > prec_in {
      return Ok(expr.clone());
    }
  }
  // If the whole expression can be converted to BigFloat, do it
  if let Ok(result) = expr_to_bigfloat(expr, bits, rm, cc) {
    let decimal = bigfloat_to_string(&result, None, rm, cc)?;
    return Ok(Expr::BigFloat(decimal, precision));
  }

  match expr {
    Expr::FunctionCall { name, args } => {
      // Honour NHoldAll / NHoldFirst / NHoldRest so e.g. `N[Out[0], 50]`
      // doesn't recurse into Out's slot and rewrite the index as a
      // BigFloat. (Out has NHoldFirst; the `0` slot must stay literal.)
      let attrs: Vec<String> = {
        let builtin: Vec<String> =
          crate::evaluator::get_builtin_attributes(name)
            .into_iter()
            .map(String::from)
            .collect();
        let user = crate::FUNC_ATTRS
          .with(|m| m.borrow().get(name).cloned().unwrap_or_default());
        let mut combined = builtin;
        for a in user {
          if !combined.contains(&a) {
            combined.push(a);
          }
        }
        combined
      };
      let hold_all = attrs.iter().any(|a| a == "NHoldAll");
      let hold_first = attrs.iter().any(|a| a == "NHoldFirst");
      let hold_rest = attrs.iter().any(|a| a == "NHoldRest");
      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .enumerate()
        .map(|(i, a)| {
          let held = hold_all || (hold_first && i == 0) || (hold_rest && i > 0);
          if held {
            Ok(a.clone())
          } else {
            n_eval_arbitrary_partial(a, precision, bits, rm, cc)
          }
        })
        .collect();
      let new_expr = Expr::FunctionCall {
        name: name.clone(),
        args: new_args?.into(),
      };
      // Try to re-evaluate after converting numeric args
      match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => Ok(result),
        Err(_) => Ok(new_expr),
      }
    }
    Expr::BinaryOp { op, left, right } => {
      let l = n_eval_arbitrary_partial(left, precision, bits, rm, cc)?;
      let r = n_eval_arbitrary_partial(right, precision, bits, rm, cc)?;
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let inner = n_eval_arbitrary_partial(operand, precision, bits, rm, cc)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(inner),
      })
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(n_eval_arbitrary_partial(
        replacement,
        precision,
        bits,
        rm,
        cc,
      )?),
    }),
    // Identifier: check user-installed `N[sym, p] = value` rules. The
    // canonical LHS Wolfram stores under NValues is
    // `N[sym, {p., Infinity}]`; match on numeric `p` equal to (or below)
    // the requested precision. Also fire for "any-precision" LHS shapes
    // like `N[sym, _]` (Blank[]) or `N[sym, x_]` (Pattern[x, Blank[]]),
    // which match every requested precision.
    Expr::Identifier(name) => {
      let nval = crate::evaluator::assignment::N_VALUES
        .with(|m| m.borrow().get(name).cloned());
      if let Some(entries) = nval {
        for (lhs_p, rhs) in &entries {
          if let Some(p) = arbitrary_precision_lhs(lhs_p)
            && (p as usize) == (precision.floor() as usize)
          {
            return n_eval_arbitrary(rhs, precision);
          }
          if is_blank_precision_lhs(lhs_p) {
            return n_eval_arbitrary(rhs, precision);
          }
        }
      }
      Ok(expr.clone())
    }
    // Other non-numeric expressions: leave as-is
    _ => Ok(expr.clone()),
  }
}

/// Does `lhs` have shape `N[sym, _]` (Blank[]) or `N[sym, x_]`
/// (Pattern[name, Blank[]])? Such an LHS pattern matches *any*
/// precision, so the corresponding NValue rule should fire on every
/// `N[sym, p]` lookup regardless of `p`.
fn is_blank_precision_lhs(lhs: &Expr) -> bool {
  let Expr::FunctionCall { name, args } = lhs else {
    return false;
  };
  if name != "N" || args.len() != 2 {
    return false;
  }
  is_blank_pattern_expr(&args[1])
}

/// Recognise any of the Blank-style pattern shapes that match every
/// concrete argument: `Blank[]` / `Blank[h]`, `Pattern[…, Blank[…]]`,
/// or the AST-level `Expr::PatternSimple` / `Expr::PatternTest` etc.
/// Plus a string-level fallback against the FullForm rendering for
/// shapes Woxi displays as `Pattern[, Blank[]]` but stores under a
/// different Expr variant.
fn is_blank_pattern_expr(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name: n2, .. } if n2 == "Blank" => true,
    Expr::FunctionCall { name: n2, args: pa }
      if n2 == "Pattern" && pa.len() == 2 =>
    {
      is_blank_pattern_expr(&pa[1])
    }
    _ => {
      let rendered = crate::syntax::expr_to_string(e);
      rendered == "_"
        || rendered.starts_with("Blank[")
        || rendered.contains("Pattern[")
    }
  }
}

/// If `lhs` has shape `N[sym, {p_real, Infinity}]` (canonical Wolfram
/// form for `N[sym, p] = value` rules), return `p`. Otherwise None.
fn arbitrary_precision_lhs(lhs: &Expr) -> Option<f64> {
  let Expr::FunctionCall { name, args } = lhs else {
    return None;
  };
  if name != "N" || args.len() != 2 {
    return None;
  }
  let Expr::List(prec) = &args[1] else {
    return None;
  };
  if prec.len() != 2 {
    return None;
  }
  let p = match &prec[0] {
    Expr::Real(v) => *v,
    Expr::Integer(n) => *n as f64,
    _ => return None,
  };
  if !matches!(&prec[1], Expr::Identifier(s) if s == "Infinity") {
    return None;
  }
  Some(p)
}

/// Recursively convert an Expr to a BigFloat with the given precision.
pub fn expr_to_bigfloat(
  expr: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<astro_float::BigFloat, InterpreterError> {
  use crate::syntax::BinaryOperator;
  use astro_float::BigFloat;

  match expr {
    Expr::Integer(n) => Ok(BigFloat::from_i128(*n, bits)),
    Expr::BigInteger(n) => {
      // Convert BigInt to BigFloat by parsing its decimal string
      let s = n.to_string();
      Ok(BigFloat::parse(&s, astro_float::Radix::Dec, bits, rm, cc))
    }
    Expr::Real(f) => Ok(BigFloat::from_f64(*f, bits)),
    Expr::BigFloat(digits, _) => Ok(BigFloat::parse(
      digits,
      astro_float::Radix::Dec,
      bits,
      rm,
      cc,
    )),
    Expr::Constant(name) => match name.as_str() {
      "Pi" | "-Pi" => {
        let pi = cc.pi(bits, rm);
        if name == "-Pi" { Ok(pi.neg()) } else { Ok(pi) }
      }
      "E" => Ok(cc.e(bits, rm)),
      "Degree" => {
        let pi = cc.pi(bits, rm);
        let d180 = BigFloat::from_i32(180, bits);
        Ok(pi.div(&d180, bits, rm))
      }
      _ => Err(InterpreterError::EvaluationError(format!(
        "N: cannot evaluate constant {} to arbitrary precision",
        name
      ))),
    },
    Expr::Identifier(name) if name == "GoldenRatio" => {
      // GoldenRatio = (1 + Sqrt[5]) / 2
      let five = BigFloat::from_i32(5, bits);
      let sqrt5 = five.sqrt(bits, rm);
      let one = BigFloat::from_i32(1, bits);
      let numer = one.add(&sqrt5, bits, rm);
      let two = BigFloat::from_i32(2, bits);
      Ok(numer.div(&two, bits, rm))
    }
    Expr::Identifier(name) if name == "GoldenAngle" => {
      // GoldenAngle = Pi * (3 - Sqrt[5]); the subtraction cancels leading
      // bits, so work with guard bits and round at the end
      let wbits = bits + 64;
      let five = BigFloat::from_i32(5, wbits);
      let sqrt5 = five.sqrt(wbits, rm);
      let three = BigFloat::from_i32(3, wbits);
      let factor = three.sub(&sqrt5, wbits, rm);
      let pi = astro_float::Consts::new()
        .map(|mut c| c.pi(wbits, rm))
        .unwrap_or_else(|_| BigFloat::from_i32(0, wbits));
      Ok(pi.mul(&factor, bits, rm))
    }
    Expr::Identifier(name) if name == "EulerGamma" => {
      Ok(compute_euler_gamma(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Catalan" => {
      Ok(compute_catalan(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Glaisher" => {
      Ok(compute_glaisher(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Khinchin" => {
      Ok(compute_khinchin(bits, rm, cc))
    }
    // MachinePrecision = Log10[2^53] = 53 * Log10[2].
    // `N[MachinePrecision, 30]` should yield the arbitrary-precision
    // decimal expansion `15.9545897701910033463281614204`.
    Expr::Identifier(name) if name == "MachinePrecision" => {
      let two = BigFloat::from_i32(2, bits);
      let ten = BigFloat::from_i32(10, bits);
      let log2 = two.ln(bits, rm, cc);
      let log10 = ten.ln(bits, rm, cc);
      let log10_2 = log2.div(&log10, bits, rm);
      let fifty_three = BigFloat::from_i32(53, bits);
      Ok(fifty_three.mul(&log10_2, bits, rm))
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let val = expr_to_bigfloat(operand, bits, rm, cc)?;
      Ok(val.neg())
    }
    Expr::BinaryOp { op, left, right } => {
      // For integer exponents, use powi (repeated squaring) for efficiency
      if matches!(op, BinaryOperator::Power)
        && let Some(n) = try_as_integer(right)
      {
        let base = expr_to_bigfloat(left, bits, rm, cc)?;
        return Ok(bigfloat_powi(&base, n, bits, rm));
      }
      let l = expr_to_bigfloat(left, bits, rm, cc)?;
      let r = expr_to_bigfloat(right, bits, rm, cc)?;
      match op {
        BinaryOperator::Plus => Ok(l.add(&r, bits, rm)),
        BinaryOperator::Minus => Ok(l.sub(&r, bits, rm)),
        BinaryOperator::Times => Ok(l.mul(&r, bits, rm)),
        BinaryOperator::Divide => Ok(l.div(&r, bits, rm)),
        BinaryOperator::Power => Ok(l.pow(&r, bits, rm, cc)),
        _ => Err(InterpreterError::EvaluationError(
          "N: unsupported binary operator for arbitrary precision".into(),
        )),
      }
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Rational" if args.len() == 2 => {
          let n = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let d = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(n.div(&d, bits, rm))
        }
        "Sqrt" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sqrt(bits, rm))
        }
        "Sin" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sin(bits, rm, cc))
        }
        "Cos" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.cos(bits, rm, cc))
        }
        "Tan" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.tan(bits, rm, cc))
        }
        "Exp" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.exp(bits, rm, cc))
        }
        "Log" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.ln(bits, rm, cc))
        }
        "Log" if args.len() == 2 => {
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let val = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(val.log(&base, bits, rm, cc))
        }
        "Abs" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.abs())
        }
        "Sinh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sinh(bits, rm, cc))
        }
        "Cosh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.cosh(bits, rm, cc))
        }
        "Tanh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.tanh(bits, rm, cc))
        }
        "ArcSin" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.asin(bits, rm, cc))
        }
        "ArcCos" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.acos(bits, rm, cc))
        }
        "ArcTan" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.atan(bits, rm, cc))
        }
        "ArcSinh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.asinh(bits, rm, cc))
        }
        "ArcCosh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.acosh(bits, rm, cc))
        }
        "ArcTanh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.atanh(bits, rm, cc))
        }
        "Plus" => {
          // Evaluated Plus[a, b, c, ...] as a function call
          let mut result = BigFloat::from_i32(0, bits);
          for arg in args {
            let v = expr_to_bigfloat(arg, bits, rm, cc)?;
            result = result.add(&v, bits, rm);
          }
          Ok(result)
        }
        "Times" => {
          let mut result = BigFloat::from_i32(1, bits);
          for arg in args {
            let v = expr_to_bigfloat(arg, bits, rm, cc)?;
            result = result.mul(&v, bits, rm);
          }
          Ok(result)
        }
        "Power" if args.len() == 2 => {
          if let Some(n) = try_as_integer(&args[1]) {
            let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
            return Ok(bigfloat_powi(&base, n, bits, rm));
          }
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let exp = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(base.pow(&exp, bits, rm, cc))
        }
        "Gamma" if args.len() == 1 => {
          let z = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(gamma_bigfloat(&z, bits, rm, cc))
        }
        "Erf" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erf(&x, bits, rm, cc))
        }
        "Erfc" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erfc(&x, bits, rm, cc))
        }
        "Erfi" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erfi(&x, bits, rm, cc))
        }
        "ExpIntegralEi" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_exp_integral_ei(&x, bits, rm, cc))
        }
        "ChampernowneNumber" if args.len() <= 1 => {
          let base = match args.first() {
            None => 10,
            Some(Expr::Integer(b)) if *b >= 2 => *b,
            _ => {
              return Err(InterpreterError::EvaluationError(
                "N: invalid ChampernowneNumber base".into(),
              ));
            }
          };
          Ok(compute_champernowne(base, bits, rm, cc))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "N: cannot evaluate {}[...] to arbitrary precision",
          name
        ))),
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate expression to arbitrary precision: {}",
      crate::syntax::expr_to_string(expr)
    ))),
  }
}

/// Convert an expression to a complex (BigFloat, BigFloat) pair with given precision.
/// Returns (real_part, imaginary_part) as BigFloats.
/// Handles expressions involving the imaginary unit I.
fn expr_to_complex_bigfloat(
  expr: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use crate::syntax::BinaryOperator;
  use astro_float::BigFloat;

  // Fast path: if purely real, delegate
  if let Ok(val) = expr_to_bigfloat(expr, bits, rm, cc) {
    return Ok((val, BigFloat::from_i32(0, bits)));
  }

  match expr {
    // I → (0, 1)
    Expr::Identifier(name) if name == "I" => {
      Ok((BigFloat::from_i32(0, bits), BigFloat::from_i32(1, bits)))
    }
    // Unary minus
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (re, im) = expr_to_complex_bigfloat(operand, bits, rm, cc)?;
      Ok((re.neg(), im.neg()))
    }
    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      let (lr, li) = expr_to_complex_bigfloat(left, bits, rm, cc)?;
      let (rr, ri) = expr_to_complex_bigfloat(right, bits, rm, cc)?;
      match op {
        BinaryOperator::Plus => {
          Ok((lr.add(&rr, bits, rm), li.add(&ri, bits, rm)))
        }
        BinaryOperator::Minus => {
          Ok((lr.sub(&rr, bits, rm), li.sub(&ri, bits, rm)))
        }
        BinaryOperator::Times => {
          // (lr + li*i) * (rr + ri*i) = (lr*rr - li*ri) + (lr*ri + li*rr)*i
          let re = lr.mul(&rr, bits, rm).sub(&li.mul(&ri, bits, rm), bits, rm);
          let im = lr.mul(&ri, bits, rm).add(&li.mul(&rr, bits, rm), bits, rm);
          Ok((re, im))
        }
        BinaryOperator::Divide => {
          // (lr + li*i) / (rr + ri*i)
          let denom =
            rr.mul(&rr, bits, rm).add(&ri.mul(&ri, bits, rm), bits, rm);
          let re = lr.mul(&rr, bits, rm).add(&li.mul(&ri, bits, rm), bits, rm);
          let im = li.mul(&rr, bits, rm).sub(&lr.mul(&ri, bits, rm), bits, rm);
          Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
        }
        BinaryOperator::Power => {
          // Only handle real^integer for now
          if li.is_zero()
            && ri.is_zero()
            && let Some(n) = try_as_integer(right)
          {
            return Ok((
              bigfloat_powi(&lr, n, bits, rm),
              BigFloat::from_i32(0, bits),
            ));
          }
          Err(InterpreterError::EvaluationError(
            "N: complex Power not supported yet".into(),
          ))
        }
        _ => Err(InterpreterError::EvaluationError(
          "N: unsupported binary operator for complex arbitrary precision"
            .into(),
        )),
      }
    }
    // Function calls
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Times" => {
        let mut result =
          (BigFloat::from_i32(1, bits), BigFloat::from_i32(0, bits));
        for arg in args {
          let (rr, ri) = expr_to_complex_bigfloat(arg, bits, rm, cc)?;
          let (lr, li) = result;
          let re = lr.mul(&rr, bits, rm).sub(&li.mul(&ri, bits, rm), bits, rm);
          let im = lr.mul(&ri, bits, rm).add(&li.mul(&rr, bits, rm), bits, rm);
          result = (re, im);
        }
        Ok(result)
      }
      "Plus" => {
        let mut result =
          (BigFloat::from_i32(0, bits), BigFloat::from_i32(0, bits));
        for arg in args {
          let (rr, ri) = expr_to_complex_bigfloat(arg, bits, rm, cc)?;
          result = (result.0.add(&rr, bits, rm), result.1.add(&ri, bits, rm));
        }
        Ok(result)
      }
      "Complex" if args.len() == 2 => {
        let re = expr_to_bigfloat(&args[0], bits, rm, cc)?;
        let im = expr_to_bigfloat(&args[1], bits, rm, cc)?;
        Ok((re, im))
      }
      "Sin" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        Ok((sin_a.mul(&cosh_b, bits, rm), cos_a.mul(&sinh_b, bits, rm)))
      }
      "Cos" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        Ok((
          cos_a.mul(&cosh_b, bits, rm),
          sin_a.mul(&sinh_b, bits, rm).neg(),
        ))
      }
      "Tan" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // tan(z) = sin(z)/cos(z)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        // sin(z) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let sr = sin_a.mul(&cosh_b, bits, rm);
        let si = cos_a.mul(&sinh_b, bits, rm);
        // cos(z) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let cr = cos_a.mul(&cosh_b, bits, rm);
        let ci = sin_a.mul(&sinh_b, bits, rm).neg();
        // (sr + si*i) / (cr + ci*i)
        let denom = cr.mul(&cr, bits, rm).add(&ci.mul(&ci, bits, rm), bits, rm);
        let re = sr.mul(&cr, bits, rm).add(&si.mul(&ci, bits, rm), bits, rm);
        let im = si.mul(&cr, bits, rm).sub(&sr.mul(&ci, bits, rm), bits, rm);
        Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
      }
      "Exp" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // exp(a+bi) = exp(a)*(cos(b) + i*sin(b))
        let exp_a = a.exp(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((exp_a.mul(&cos_b, bits, rm), exp_a.mul(&sin_b, bits, rm)))
      }
      "Sinh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((sinh_a.mul(&cos_b, bits, rm), cosh_a.mul(&sin_b, bits, rm)))
      }
      "Cosh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((cosh_a.mul(&cos_b, bits, rm), sinh_a.mul(&sin_b, bits, rm)))
      }
      "Tanh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // tanh(z) = sinh(z)/cosh(z)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        let sr = sinh_a.mul(&cos_b, bits, rm);
        let si = cosh_a.mul(&sin_b, bits, rm);
        let cr = cosh_a.mul(&cos_b, bits, rm);
        let ci = sinh_a.mul(&sin_b, bits, rm);
        let denom = cr.mul(&cr, bits, rm).add(&ci.mul(&ci, bits, rm), bits, rm);
        let re = sr.mul(&cr, bits, rm).add(&si.mul(&ci, bits, rm), bits, rm);
        let im = si.mul(&cr, bits, rm).sub(&sr.mul(&ci, bits, rm), bits, rm);
        Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
      }
      "Log" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // log(a+bi) = ln(|z|) + i*arg(z)
        // |z| = sqrt(a^2 + b^2), arg(z) = atan2(b, a)
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        let abs_val = abs_sq.sqrt(bits, rm);
        let ln_abs = abs_val.ln(bits, rm, cc);
        // atan2(b, a) implemented using atan
        let arg = if a.is_zero() {
          let half_pi =
            cc.pi(bits, rm).div(&BigFloat::from_i32(2, bits), bits, rm);
          if b.is_negative() {
            half_pi.neg()
          } else {
            half_pi
          }
        } else if a.is_negative() {
          let atan_val = b.div(&a, bits, rm).atan(bits, rm, cc);
          if b.is_negative() {
            atan_val.sub(&cc.pi(bits, rm), bits, rm)
          } else {
            atan_val.add(&cc.pi(bits, rm), bits, rm)
          }
        } else {
          b.div(&a, bits, rm).atan(bits, rm, cc)
        };
        Ok((ln_abs, arg))
      }
      "Sqrt" if args.len() == 1 => {
        // sqrt(a+bi) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2)
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        let abs_val = abs_sq.sqrt(bits, rm);
        let two = BigFloat::from_i32(2, bits);
        let re = abs_val.add(&a, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        let mut im =
          abs_val.sub(&a, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        if b.is_negative() {
          im = im.neg();
        }
        Ok((re, im))
      }
      "Rational" if args.len() == 2 => {
        let n = expr_to_bigfloat(&args[0], bits, rm, cc)?;
        let d = expr_to_bigfloat(&args[1], bits, rm, cc)?;
        Ok((n.div(&d, bits, rm), BigFloat::from_i32(0, bits)))
      }
      "Power" if args.len() == 2 => {
        if let Some(n) = try_as_integer(&args[1]) {
          let (re, im) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
          if im.is_zero() {
            return Ok((
              bigfloat_powi(&re, n, bits, rm),
              BigFloat::from_i32(0, bits),
            ));
          }
        }
        Err(InterpreterError::EvaluationError(
          "N: complex Power not fully supported".into(),
        ))
      }
      "Abs" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        Ok((abs_sq.sqrt(bits, rm), BigFloat::from_i32(0, bits)))
      }
      "ExpIntegralEi" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        complex_exp_integral_ei(a, b, bits, rm, cc)
      }
      _ => Err(InterpreterError::EvaluationError(format!(
        "N: cannot evaluate {}[...] to complex arbitrary precision",
        name
      ))),
    },
    _ => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate expression to complex arbitrary precision: {}",
      crate::syntax::expr_to_string(expr)
    ))),
  }
}

/// Build a properly formatted complex result from BigFloat real and imaginary parts.
fn build_complex_bigfloat_result(
  re: astro_float::BigFloat,
  im: astro_float::BigFloat,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let i_expr = Expr::Identifier("I".to_string());
  let max_digits: Option<usize> = None;

  if im.is_zero() {
    let re_str = bigfloat_to_string(&re, None, rm, cc)?;
    return Ok(Expr::BigFloat(re_str, precision as f64));
  }

  let im_negative = im.is_negative();
  let im_abs = if im_negative { im.neg() } else { im.clone() };
  let im_str = bigfloat_to_string(&im_abs, max_digits, rm, cc)?;

  let im_bf = Expr::BigFloat(im_str, precision as f64);

  // Build |im| * I term (always positive coefficient)
  let abs_im_term = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(im_bf),
    right: Box::new(i_expr),
  };

  if re.is_zero() {
    if im_negative {
      // Pure negative imaginary: -|im|*I
      let neg_im_str = bigfloat_to_string(&im, max_digits, rm, cc)?;
      let neg_im_bf = Expr::BigFloat(neg_im_str, precision as f64);
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(neg_im_bf),
        right: Box::new(Expr::Identifier("I".to_string())),
      });
    }
    return Ok(abs_im_term);
  }

  let re_str = bigfloat_to_string(&re, max_digits, rm, cc)?;
  let re_bf = Expr::BigFloat(re_str, precision as f64);

  if im_negative {
    // re - |im|*I
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(re_bf),
      right: Box::new(abs_im_term),
    })
  } else {
    // re + |im|*I
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(re_bf),
      right: Box::new(abs_im_term),
    })
  }
}

/// Build a complex result with per-component string precision markers.
/// Uses Expr::Raw to embed the precision marker directly in the formatted string.
fn build_complex_result_with_string_precision(
  re: astro_float::BigFloat,
  im: astro_float::BigFloat,
  prec_re_str: &str,
  prec_im_str: &str,
  _precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let re_str = bigfloat_to_string(&re, None, rm, cc)?;

  let re_raw = Expr::Raw(format!("{}`{}", re_str, prec_re_str));

  let im_negative = im.is_negative();
  let im_abs_str = if im_negative {
    bigfloat_to_string(&im.neg(), None, rm, cc)?
  } else {
    bigfloat_to_string(&im, None, rm, cc)?
  };
  let im_raw = Expr::Raw(format!("{}`{}", im_abs_str, prec_im_str));

  let i_expr = Expr::Identifier("I".to_string());
  let abs_im_term = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(im_raw),
    right: Box::new(i_expr),
  };

  if im_negative {
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(re_raw),
      right: Box::new(abs_im_term),
    })
  } else {
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(re_raw),
      right: Box::new(abs_im_term),
    })
  }
}

/// Compute per-component precision markers for a complex function evaluation.
///
/// Uses the formula:
///   accuracy = p + log10(|input|) - log10(|output|)
///   precision_component = accuracy + log10(|component|)
///
/// Computes using BigFloat arithmetic for accuracy, then converts to f64.
fn compute_complex_precision_markers(
  in_re: &astro_float::BigFloat,
  in_im: &astro_float::BigFloat,
  out_re: &astro_float::BigFloat,
  out_im: &astro_float::BigFloat,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(String, String), InterpreterError> {
  use astro_float::BigFloat;
  let pbits = 256; // high precision for the log computations

  // |input| = sqrt(in_re^2 + in_im^2)
  let in_abs_sq =
    in_re
      .mul(in_re, pbits, rm)
      .add(&in_im.mul(in_im, pbits, rm), pbits, rm);
  let in_abs = in_abs_sq.sqrt(pbits, rm);

  // |output| = sqrt(out_re^2 + out_im^2)
  let out_abs_sq = out_re.mul(out_re, pbits, rm).add(
    &out_im.mul(out_im, pbits, rm),
    pbits,
    rm,
  );
  let out_abs = out_abs_sq.sqrt(pbits, rm);

  // log10(x) = ln(x) / ln(10)
  let ln10 = BigFloat::from_i32(10, pbits).ln(pbits, rm, cc);
  let log10_in = in_abs.ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_out = out_abs.ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_re = out_re.abs().ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_im = out_im.abs().ln(pbits, rm, cc).div(&ln10, pbits, rm);

  let p_bf = BigFloat::from_i32(precision as i32, pbits);
  // accuracy = p + log10(|input|) - log10(|output|)
  let accuracy = p_bf.add(&log10_in, pbits, rm).sub(&log10_out, pbits, rm);

  // precision_re = accuracy + log10(|re|)
  let prec_re_bf = accuracy.add(&log10_re, pbits, rm);
  // precision_im = accuracy + log10(|im|)
  let prec_im_bf = accuracy.add(&log10_im, pbits, rm);

  // Format precision markers directly from BigFloat with f64-like precision
  // This avoids f64 rounding issues by going from BigFloat → string directly
  let prec_re_s = format_bigfloat_as_precision_marker(&prec_re_bf, rm, cc)?;
  let prec_im_s = format_bigfloat_as_precision_marker(&prec_im_bf, rm, cc)?;

  Ok((prec_re_s, prec_im_s))
}

/// Format a BigFloat as a precision marker string, matching Wolfram's formatting.
/// Wolfram displays precision markers with ~15-17 significant digits (f64-like precision),
/// stripping trailing zeros.
fn format_bigfloat_as_precision_marker(
  bf: &astro_float::BigFloat,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<String, InterpreterError> {
  // Format with ~17 significant digits (matching f64 precision)
  let full_str = bigfloat_to_string(bf, Some(17), rm, cc)?;

  // Parse to f64 and back to get Wolfram-style formatting
  // Wolfram uses a formatting that shows all distinguishing digits
  let val: f64 = full_str.trim_end_matches('.').parse().unwrap_or(0.0);

  if val.fract() == 0.0 {
    Ok(format!("{}.", val as i64))
  } else {
    // Use Rust's Debug format which shows the exact f64 representation
    // with minimum digits to uniquely represent the value
    Ok(format!("{:?}", val))
  }
}

/// Rescale[x, {xmin, xmax}] - rescales x to [0,1]
/// Rescale[x, {xmin, xmax}, {ymin, ymax}] - rescales x to [ymin,ymax]
/// Rescale[list] - rescales list elements to [0,1] based on min/max
/// Recursively collect numeric leaves of a (possibly nested) list for the
/// 1-argument `Rescale`. Sets `ok = false` on any non-numeric, non-list leaf
/// and clears `all_int` when a `Real` leaf is seen.
fn rescale_collect_leaves(
  expr: &Expr,
  vals: &mut Vec<f64>,
  int_vals: &mut Vec<i128>,
  all_int: &mut bool,
  ok: &mut bool,
) {
  match expr {
    Expr::List(items) => {
      for it in items.iter() {
        rescale_collect_leaves(it, vals, int_vals, all_int, ok);
      }
    }
    Expr::Integer(n) => {
      vals.push(*n as f64);
      int_vals.push(*n);
    }
    Expr::Real(f) => {
      vals.push(*f);
      *all_int = false;
    }
    _ => *ok = false,
  }
}

/// Rebuild a (possibly nested) list, rescaling each numeric leaf to [0, 1]
/// using the global `min`/range. Integer data stays exact via `make_rational`;
/// degenerate (zero-range) data maps every leaf to 0, matching wolframscript.
fn rescale_rebuild(
  expr: &Expr,
  all_int: bool,
  min_i: i128,
  range_i: i128,
  min_f: f64,
  range_f: f64,
  degenerate: bool,
) -> Expr {
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|it| {
          rescale_rebuild(
            it, all_int, min_i, range_i, min_f, range_f, degenerate,
          )
        })
        .collect(),
    ),
    // Degenerate (zero-range) data maps every leaf to 0. An inexact list
    // gives the machine real 0., an exact list the Integer 0.
    _ if degenerate => {
      if all_int {
        Expr::Integer(0)
      } else {
        Expr::Real(0.0)
      }
    }
    Expr::Integer(n) if all_int => make_rational(n - min_i, range_i),
    _ => {
      // Reached only for an inexact list (all_int is false), so the result
      // stays a Real even at a whole-number value (Rescale[{1., 2., 3.}] =
      // {0., 0.5, 1.}, not {0, 0.5, 1}). num_to_expr would collapse a whole
      // number to an exact Integer, discarding the inexactness.
      let x = match expr {
        Expr::Integer(n) => *n as f64,
        Expr::Real(f) => *f,
        _ => 0.0,
      };
      Expr::Real((x - min_f) / range_f)
    }
  }
}

pub fn rescale_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Rescale expects 1 to 3 arguments".into(),
    ));
  }

  // Rescale[list] - auto-detect the global min/max across all (possibly
  // nested) elements and rescale each, preserving the list structure.
  if args.len() == 1 {
    if let Expr::List(items) = &args[0] {
      if items.is_empty() {
        return Ok(Expr::List(vec![].into()));
      }
      let mut vals = Vec::new();
      let mut int_vals: Vec<i128> = Vec::new();
      let mut all_int = true;
      let mut ok = true;
      rescale_collect_leaves(
        &args[0],
        &mut vals,
        &mut int_vals,
        &mut all_int,
        &mut ok,
      );
      if !ok || vals.is_empty() {
        return Ok(Expr::FunctionCall {
          name: "Rescale".to_string(),
          args: args.to_vec().into(),
        });
      }
      let min_f = vals.iter().cloned().fold(f64::INFINITY, f64::min);
      let max_f = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
      let degenerate = (max_f - min_f).abs() < f64::EPSILON;
      let (min_i, range_i) = if all_int {
        let mn = *int_vals.iter().min().unwrap();
        let mx = *int_vals.iter().max().unwrap();
        (mn, mx - mn)
      } else {
        (0, 0)
      };
      return Ok(rescale_rebuild(
        &args[0],
        all_int,
        min_i,
        range_i,
        min_f,
        max_f - min_f,
        degenerate,
      ));
    }
    // Single non-list value needs {xmin, xmax}
    return Ok(Expr::FunctionCall {
      name: "Rescale".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Handle list first argument: Rescale[{x1, x2, ...}, range] maps over elements
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        let mut new_args = args.to_vec();
        new_args[0] = item.clone();
        rescale_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  // Rescale[x, {xmin, xmax}] or Rescale[x, {xmin, xmax}, {ymin, ymax}]
  let range = match &args[1] {
    Expr::List(r) if r.len() == 2 => r,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Rescale".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let (ymin, ymax) = if args.len() == 3 {
    match &args[2] {
      Expr::List(r) if r.len() == 2 => (&r[0], &r[1]),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Rescale".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    (&Expr::Integer(0) as &Expr, &Expr::Integer(1) as &Expr)
  };

  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "Rescale".to_string(),
      args: args.to_vec().into(),
    })
  };

  // A literal exact number: Integer, BigInteger, Real or Rational[p, q].
  let is_numeric_literal = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_))
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2)
  };

  // The bounds and targets must be literal numbers. Symbolic bounds/targets
  // ({a, b}) make wolframscript produce an Apart/Expand-normalized form that
  // Woxi's canonicalization does not reproduce, so those stay unevaluated.
  if !is_numeric_literal(&range[0])
    || !is_numeric_literal(&range[1])
    || !is_numeric_literal(ymin)
    || !is_numeric_literal(ymax)
  {
    return unevaluated();
  }
  let xmin_n = try_eval_to_f64(&range[0]).unwrap();
  let xmax_n = try_eval_to_f64(&range[1]).unwrap();
  if (xmax_n - xmin_n).abs() < f64::EPSILON {
    return unevaluated();
  }
  // For a non-literal x (a symbol or constant like Pi), only xmin == 0 is
  // safe: with a nonzero xmin wolframscript distributes the offset
  // (10 + 5 (x - 2) -> 5 x) in a way Woxi's evaluator leaves factored.
  if !is_numeric_literal(&args[0]) && xmin_n.abs() >= f64::EPSILON {
    return unevaluated();
  }

  // Build and evaluate  ymin + (x - xmin) * (ymax - ymin) / (xmax - xmin).
  // Evaluating the symbolic expression keeps exact and symbolic values
  // (Pi/10, 1/3, x/10, …) instead of floatifying, and a Real input still
  // yields a Real.
  let neg = |e: &Expr| Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), e.clone()].into(),
  };
  let sub = |a: &Expr, b: &Expr| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![a.clone(), neg(b)].into(),
  };
  let x_minus_min = sub(&args[0], &range[0]);
  let y_span = sub(ymax, ymin);
  let x_span = sub(&range[1], &range[0]);
  let fraction = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      x_minus_min,
      y_span,
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![x_span, Expr::Integer(-1)].into(),
      },
    ]
    .into(),
  };
  let result = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![ymin.clone(), fraction].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Normalize[v] - normalizes a vector to unit length
/// Normalize[{3, 4}] => {3/5, 4/5}
/// Normalize[{0, 0, 0}] => {0, 0, 0}
/// Norm[v] - Euclidean norm (L2) of a vector
/// Norm[v, p] - Lp norm
/// Recursively check whether an expression tree contains any
/// inexact-Real or BigFloat leaf. Used by Norm (and similar) to
/// decide between exact/symbolic and machine-precision numerical
/// evaluation.
fn contains_inexact_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::List(items) => items.iter().any(contains_inexact_real),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_inexact_real),
    Expr::BinaryOp { left, right, .. } => {
      contains_inexact_real(left) || contains_inexact_real(right)
    }
    Expr::UnaryOp { operand, .. } => contains_inexact_real(operand),
    _ => false,
  }
}

pub fn norm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Norm expects 1 or 2 arguments".into(),
    ));
  }
  // An empty vector has no norm: emit Norm::nvm and stay unevaluated rather
  // than collapsing to 0. (wolframscript parity)
  if matches!(&args[0], Expr::List(items) if items.is_empty()) {
    crate::emit_message(
      "Norm::nvm: The first Norm argument should be a scalar, vector or matrix.",
    );
    return Ok(Expr::FunctionCall {
      name: "Norm".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Determine the norm parameter p
  let p_expr = if args.len() == 2 {
    Some(args[1].clone())
  } else {
    None
  };
  let p_val = match &p_expr {
    Some(e) => try_eval_to_f64(e),
    None => Some(2.0),
  };
  // Check for Infinity norm
  let is_infinity = match &p_expr {
    Some(Expr::Identifier(s)) if s == "Infinity" => true,
    Some(Expr::FunctionCall { name, args })
      if name == "DirectedInfinity" && args.len() == 1 =>
    {
      true
    }
    _ => p_val == Some(f64::INFINITY),
  };

  // Norm[matrix, "Frobenius"] — sqrt of sum of squared absolute values
  // across every entry of the (rectangular) matrix.
  if matches!(&p_expr, Some(Expr::String(s)) if s == "Frobenius")
    && let Expr::List(rows) = &args[0]
    && rows.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let mut flat: Vec<Expr> = Vec::new();
    for row in rows {
      if let Expr::List(cells) = row {
        flat.extend(cells.iter().cloned());
      }
    }
    return norm_ast(&[Expr::List(flat.into())]);
  }

  // Matrix norms: args[0] is a rectangular list of lists. These differ from
  // the element-wise vector norms below:
  //   p = 1        → maximum absolute column sum
  //   p = Infinity → maximum absolute row sum
  //   p = 2        → spectral norm (largest singular value)
  //                  = Sqrt[Max[Eigenvalues[Transpose[A].A]]]
  if let Expr::List(rows) = &args[0]
    && !rows.is_empty()
    && rows.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let mat: Vec<Vec<Expr>> = rows
      .iter()
      .map(|r| match r {
        Expr::List(c) => c.to_vec(),
        _ => unreachable!(),
      })
      .collect();
    // A matrix has scalar entries. If any entry is itself a list the argument
    // is a rank >= 3 tensor, which Norm does not accept: emit Norm::nvm and
    // stay unevaluated (matching wolframscript) rather than threading the
    // matrix-norm formula over the deeper structure.
    if mat
      .iter()
      .any(|row| row.iter().any(|c| matches!(c, Expr::List(_))))
    {
      crate::emit_message(
        "Norm::nvm: The first Norm argument should be a scalar, vector or matrix.",
      );
      return Ok(Expr::FunctionCall {
        name: "Norm".to_string(),
        args: args.to_vec().into(),
      });
    }
    let ncols = mat[0].len();
    if ncols > 0 && mat.iter().all(|r| r.len() == ncols) {
      use crate::evaluator::evaluate_expr_to_expr;
      let abs = |e: &Expr| Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![e.clone()].into(),
      };
      let sum = |terms: Vec<Expr>| Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      };
      let max_of = |terms: Vec<Expr>| Expr::FunctionCall {
        name: "Max".to_string(),
        args: terms.into(),
      };

      if is_infinity {
        // Maximum absolute row sum.
        let row_sums: Vec<Expr> = mat
          .iter()
          .map(|row| sum(row.iter().map(&abs).collect()))
          .collect();
        return evaluate_expr_to_expr(&max_of(row_sums));
      }
      if p_val == Some(1.0) {
        // Maximum absolute column sum.
        let col_sums: Vec<Expr> = (0..ncols)
          .map(|j| sum(mat.iter().map(|row| abs(&row[j])).collect()))
          .collect();
        return evaluate_expr_to_expr(&max_of(col_sums));
      }
      if p_val == Some(2.0) {
        // Spectral norm: Sqrt[Max[Eigenvalues[Transpose[A].A]]].
        let a = args[0].clone();
        let at = Expr::FunctionCall {
          name: "Transpose".to_string(),
          args: vec![a.clone()].into(),
        };
        let ata = Expr::FunctionCall {
          name: "Dot".to_string(),
          args: vec![at, a].into(),
        };
        let eig = Expr::FunctionCall {
          name: "Eigenvalues".to_string(),
          args: vec![ata].into(),
        };
        let sqrt = Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![max_of(vec![eig])].into(),
        };
        return evaluate_expr_to_expr(&sqrt);
      }
      // Other matrix p-norms are not defined; leave unevaluated.
      return Ok(Expr::FunctionCall {
        name: "Norm".to_string(),
        args: args.to_vec().into(),
      });
    }
  }

  match &args[0] {
    Expr::List(items) => {
      let p = p_val.unwrap_or(2.0);

      // "Inexact" mode: any item contains a Real/BigFloat leaf — collapse
      // to a machine-precision numeric result, mirroring Wolfram's
      // behavior (Norm[{1.0, 2, 3}] → 3.741…).
      // Otherwise stay in "exact" mode and build a symbolic expression.
      let inexact = items.iter().any(contains_inexact_real);

      if inexact {
        let mut vals = Vec::with_capacity(items.len());
        for item in items {
          match try_eval_to_f64(item) {
            Some(v) => vals.push(v),
            None => {
              return Ok(Expr::FunctionCall {
                name: "Norm".to_string(),
                args: args.to_vec().into(),
              });
            }
          }
        }
        // An inexact vector gives an inexact norm even at a whole-number value
        // (Norm[{3., 4.}] = 5., not 5), so return Real rather than num_to_expr
        // (which would collapse a whole number to an exact Integer).
        if p == 1.0 {
          let result: f64 = vals.iter().map(|x| x.abs()).sum();
          return Ok(Expr::Real(result));
        }
        if is_infinity {
          let result = vals.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
          return Ok(Expr::Real(result));
        }
        let sum: f64 = vals.iter().map(|x| x.abs().powf(p)).sum();
        return Ok(Expr::Real(sum.powf(1.0 / p)));
      }

      // Exact/symbolic mode.
      use crate::evaluator::evaluate_expr_to_expr;

      // For each item decide whether to wrap in Abs: if the item is a
      // numerically-evaluable (hence known-real) expression — integers,
      // rationals, Pi, Sin[1], 2 Sin[2], … — drop the Abs and build
      // item^p directly so that known scalars combine. For unknown
      // symbols (x, f[x]) keep Abs to preserve correctness over ℂ.
      let is_real_valued =
        |item: &Expr| -> bool { try_eval_to_f64(item).is_some() };

      if is_infinity {
        // Max[Abs[x], Abs[y], ...]
        let abs_items: Vec<Expr> = items
          .iter()
          .map(|item| Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![item.clone()].into(),
          })
          .collect();
        return evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Max".to_string(),
          args: abs_items.into(),
        });
      }

      // Match against `p_val`/`p_expr` (not the defaulted `p`) so a symbolic p
      // falls through to the general p-norm instead of the 2-norm.
      if p_val == Some(1.0) {
        // Sum of Abs[item]
        let terms: Vec<Expr> = items
          .iter()
          .map(|item| Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![item.clone()].into(),
          })
          .collect();
        let sum = if terms.len() == 1 {
          terms.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          }
        };
        return evaluate_expr_to_expr(&sum);
      }

      if p_expr.is_none() || p_val == Some(2.0) {
        // Sqrt[Plus[item^2, ...]] (or Abs[item]^2 for unknown items)
        let sq_items: Vec<Expr> = items
          .iter()
          .map(|item| {
            let base = if is_real_valued(item) {
              item.clone()
            } else {
              Expr::FunctionCall {
                name: "Abs".to_string(),
                args: vec![item.clone()].into(),
              }
            };
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![base, Expr::Integer(2)].into(),
            }
          })
          .collect();
        let sum = if sq_items.len() == 1 {
          sq_items.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: sq_items.into(),
          }
        };
        let sum_eval = evaluate_expr_to_expr(&sum)?;
        return evaluate_expr_to_expr(&make_sqrt(sum_eval));
      }

      // General p-norm (p is a number other than 1/2/Infinity, or symbolic):
      // (Sum Abs[item]^p)^(1/p). Abs of a known real evaluates away, so
      // numeric inputs collapse (Norm[{3, 4}, 4] -> 337^(1/4)).
      let p_term = p_expr.clone().unwrap_or(Expr::Integer(2));
      let power = |base: Expr, exp: Expr| Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![base, exp].into(),
      };
      let terms: Vec<Expr> = items
        .iter()
        .map(|item| {
          power(
            Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![item.clone()].into(),
            },
            p_term.clone(),
          )
        })
        .collect();
      let sum = if terms.len() == 1 {
        terms.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms.into(),
        }
      };
      let result = power(sum, power(p_term, Expr::Integer(-1)));
      evaluate_expr_to_expr(&result)
    }
    // Norm of a scalar. For a number (real or complex) Norm[x] = Abs[x],
    // evaluated symbolically so exact values are preserved (Norm[Pi] -> Pi,
    // Norm[2/3] -> 2/3, Norm[3 + 4 I] -> 5). For a non-numeric scalar (x,
    // a + b I, …) wolframscript leaves Norm unevaluated.
    _ => {
      if crate::functions::predicate_ast::is_numeric_q_pub(&args[0]) {
        crate::evaluator::evaluate_function_call_ast("Abs", &[args[0].clone()])
      } else {
        Ok(Expr::FunctionCall {
          name: "Norm".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
  }
}

pub fn normalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    // Normalize[v, f] — divides v by f[v]
    let norm_val =
      crate::functions::list_helpers_ast::apply_func_ast(&args[1], &args[0])?;
    // v / norm_val
    let result = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(args[0].clone()),
      right: Box::new(norm_val),
    })?;
    return Ok(result);
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Normalize expects 1 or 2 arguments".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![].into()));
      }
      // Compute the Euclidean norm
      let mut vals = Vec::new();
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      for item in items {
        match item {
          Expr::Integer(n) => {
            vals.push(*n as f64);
            int_vals.push(*n);
          }
          Expr::Real(f) => {
            vals.push(*f);
            all_int = false;
          }
          _ => {
            // Symbolic case: return {elem/Sqrt[sum_of_squares], ...}
            // like Mathematica does for Normalize[{a, b}] → {a/Sqrt[a^2+b^2], b/Sqrt[a^2+b^2]}
            let squared_terms: Vec<Expr> = items
              .iter()
              .map(|e| Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Abs".to_string(),
                  args: vec![e.clone()].into(),
                }),
                right: Box::new(Expr::Integer(2)),
              })
              .collect();
            let sum_of_squares = if squared_terms.len() == 1 {
              squared_terms.into_iter().next().unwrap()
            } else {
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: squared_terms.into(),
              }
            };
            // Evaluate the norm so numeric entries collapse the Abs-of-square
            // sum (e.g. Normalize[{1, I}] -> {1/Sqrt[2], I/Sqrt[2]} rather than
            // leaving Sqrt[Abs[1]^2 + Abs[I]^2]); a fully symbolic vector keeps
            // the Abs form unchanged.
            let norm_expr = crate::evaluator::evaluate_expr_to_expr(
              &make_sqrt(sum_of_squares),
            )?;
            let result: Vec<Expr> = items
              .iter()
              .map(|e| {
                crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Divide,
                  left: Box::new(e.clone()),
                  right: Box::new(norm_expr.clone()),
                })
              })
              .collect::<Result<Vec<_>, _>>()?;
            return Ok(Expr::List(result.into()));
          }
        }
      }
      let norm_sq: f64 = vals.iter().map(|x| x * x).sum();
      if norm_sq == 0.0 {
        return Ok(args[0].clone());
      }
      let norm = norm_sq.sqrt();

      if all_int {
        // Try to keep exact: each element / Sqrt[sum_sq]
        let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
        // Check if sum_sq is a perfect square
        let root = (sum_sq as f64).sqrt() as i128;
        if root * root == sum_sq && root > 0 {
          // Exact: each element / root
          let result: Vec<Expr> =
            int_vals.iter().map(|x| make_rational(*x, root)).collect();
          return Ok(Expr::List(result.into()));
        }
        // Return as xi / Sqrt[sum_sq]
        let result: Vec<Expr> = int_vals
          .iter()
          .map(|x| {
            if *x == 0 {
              Expr::Integer(0)
            } else {
              // x / Sqrt[sum_sq] = x * Power[sum_sq, -1/2]
              Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(Expr::Integer(*x)),
                right: Box::new(make_sqrt(Expr::Integer(sum_sq))),
              }
            }
          })
          .collect();
        return Ok(Expr::List(result.into()));
      }

      // Float path
      let result: Vec<Expr> =
        vals.iter().map(|x| num_to_expr(x / norm)).collect();
      Ok(Expr::List(result.into()))
    }
    _ => {
      // Scalar: Normalize[x] = x / Norm[x]
      let norm_val = norm_ast(args)?;
      // If norm is 0, return the original
      let is_zero = match &norm_val {
        Expr::Integer(0) => true,
        Expr::Real(f) if *f == 0.0 => true,
        _ => false,
      };
      if is_zero {
        return Ok(args[0].clone());
      }
      crate::evaluator::evaluate_function_call_ast(
        "Divide",
        &[args[0].clone(), norm_val],
      )
    }
  }
}

/// Unitize[x] - returns 0 for 0, 1 for anything else
/// Unitize[x, dx] - returns 0 when |x| < dx, 1 otherwise
/// Unitize[list] - maps over lists
pub fn unitize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Unitize expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument form: Unitize[x, dx]
  if args.len() == 2 {
    // Thread over lists in the first argument.
    if let Expr::List(items) = &args[0] {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|x| unitize_ast(&[x.clone(), args[1].clone()]))
        .collect();
      return Ok(Expr::List(results?.into()));
    }

    let x_val =
      crate::functions::math_ast::numeric_utils::try_eval_to_f64(&args[0]);
    let tol_val =
      crate::functions::math_ast::numeric_utils::try_eval_to_f64(&args[1]);
    if let (Some(x), Some(tol)) = (x_val, tol_val) {
      if x.abs() < tol {
        return Ok(Expr::Integer(0));
      }
      return Ok(Expr::Integer(1));
    }

    // Non-numeric arguments remain unevaluated.
    return Ok(Expr::FunctionCall {
      name: "Unitize".to_string(),
      args: args.to_vec().into(),
    });
  }

  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(_) => Ok(Expr::Integer(1)),
    Expr::Real(f) if *f == 0.0 => Ok(Expr::Integer(0)),
    Expr::Real(_) => Ok(Expr::Integer(1)),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|x| unitize_ast(&[x.clone()])).collect();
      Ok(Expr::List(results?.into()))
    }
    _ => {
      // Numeric-eval fallback: if the argument collapses to a finite
      // non-zero real, Unitize is 1 (covers Pi, E, EulerGamma,
      // Sqrt[2], 2*Pi, 3/4, Log[2], ...). Zero numeric results are
      // already handled because Pi - Pi etc. simplify symbolically
      // before reaching Unitize.
      if let Some(v) =
        crate::functions::math_ast::numeric_utils::try_eval_to_f64(&args[0])
        && v.is_finite()
        && v != 0.0
      {
        return Ok(Expr::Integer(1));
      }
      Ok(Expr::FunctionCall {
        name: "Unitize".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Precision[x] - the number of significant decimal digits
/// Returns MachinePrecision for machine reals, Infinity for exact numbers
pub fn precision_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Precision expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Constant(_) => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::Real(_) => Ok(Expr::Identifier("MachinePrecision".to_string())),
    Expr::BigFloat(digits, prec) => {
      // A literal zero BigFloat (e.g. `0.`20`, `0.``3`) reports
      // precision 0 in Wolfram — there are no significant digits when
      // the value itself is zero, so precision = -Log10(|0|) is treated
      // as 0 rather than the spec'd precision.
      if digits.parse::<f64>().is_ok_and(|f| f == 0.0) {
        Ok(Expr::Real(0.0))
      } else {
        Ok(Expr::Real(*prec))
      }
    }
    Expr::Identifier(name)
      if name == "Infinity" || name == "ComplexInfinity" =>
    {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      ..
    } => {
      // Exact rationals like 1/2 have infinite precision
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::List(items) => {
      // Precision of a list is the minimum precision of its elements,
      // with one Wolfram-specific quirk: when both MachinePrecision and
      // arbitrary-precision Reals appear in the same list, the whole
      // result is MachinePrecision (machine arithmetic infects the mix).
      let mp: f64 = 15.954589770191003;
      let mut min_prec: Option<f64> = None;
      let mut min_is_machine = false;
      let mut saw_machine = false;
      let mut saw_arb = false;
      for item in items {
        let p = precision_ast(&[item.clone()])?;
        match p {
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
            saw_machine = true;
            match min_prec {
              None => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              Some(v) if mp < v => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              _ => {}
            }
          }
          Expr::Real(f) => {
            saw_arb = true;
            match min_prec {
              None => {
                min_prec = Some(f);
                min_is_machine = false;
              }
              Some(v) if f < v => {
                min_prec = Some(f);
                min_is_machine = false;
              }
              _ => {}
            }
          }
          _ => {}
        }
      }
      if saw_machine && saw_arb {
        return Ok(Expr::Identifier("MachinePrecision".to_string()));
      }
      match min_prec {
        Some(_) if min_is_machine => {
          Ok(Expr::Identifier("MachinePrecision".to_string()))
        }
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(Expr::Identifier("Infinity".to_string())),
      }
    }
    // For symbolic expressions, check if any subexpression has finite precision.
    // If the minimum comes from a machine-real element, return the symbol
    // MachinePrecision (matches wolframscript).
    Expr::FunctionCall { args: fargs, .. } => {
      let mp: f64 = 15.954589770191003;
      let mut min_prec: Option<f64> = None;
      let mut min_is_machine = false;
      for arg in fargs {
        let p = precision_ast(&[arg.clone()])?;
        match p {
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
            match min_prec {
              None => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              Some(v) if mp < v => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              _ => {}
            }
          }
          Expr::Real(f) => match min_prec {
            None => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            Some(v) if f < v => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            _ => {}
          },
          _ => {}
        }
      }
      match min_prec {
        Some(_) if min_is_machine => {
          Ok(Expr::Identifier("MachinePrecision".to_string()))
        }
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(Expr::Identifier("Infinity".to_string())),
      }
    }
    _ => Ok(Expr::Identifier("Infinity".to_string())),
  }
}

/// Accuracy[x] - the number of significant decimal digits to the right of the decimal point
/// Returns Infinity for exact numbers, computes from precision for approximate numbers
pub fn accuracy_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Accuracy expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Constant(_) => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::Real(f) => {
      // Accuracy = MachinePrecision - Log10[Abs[x]]
      // MachinePrecision ≈ 15.9546
      let machine_precision = 15.954589770191003_f64;
      if *f == 0.0 {
        // Accuracy[0.] is the negative log of the smallest distinguishable
        // machine number, i.e. MachinePrecision - Log10[$MinMachineNumber].
        // Wolfram uses 2^-1022 (smallest normalised double) here.
        let min_machine_number = f64::MIN_POSITIVE;
        return Ok(Expr::Real(machine_precision - min_machine_number.log10()));
      }
      let accuracy = machine_precision - f.abs().log10();
      Ok(Expr::Real(accuracy))
    }
    // Arbitrary-precision: Accuracy = Precision - Log10[Abs[x]]. The
    // BigFloat carries the precision in its own field; for x == 0 the
    // precision *is* the accuracy (Wolfram stores `0``a` as a 0 with
    // accuracy a, which we represent as a BigFloat whose precision field
    // already holds a — see the parser).
    Expr::BigFloat(digits, prec) => {
      let val: f64 = digits.parse().unwrap_or(0.0);
      let prec_f = *prec;
      if val == 0.0 {
        return Ok(Expr::Real(prec_f));
      }
      Ok(Expr::Real(prec_f - val.abs().log10()))
    }
    Expr::Identifier(name)
      if name == "Infinity"
        || name == "ComplexInfinity"
        || name == "Indeterminate" =>
    {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      ..
    } => Ok(Expr::Identifier("Infinity".to_string())),
    // Complex number with finite-accuracy parts: apply Wolfram's formula
    // Accuracy[Complex[re, im]] = -Log10[Sqrt[10^(-2*Acc[re]) + 10^(-2*Acc[im])]].
    // Without this, `Accuracy[Complex[3.00``2, 4.00``2]]` would just take
    // min(2, 2) = 2 instead of the correct 1.8494…
    _ if try_complex_accuracy(&args[0]).is_some() => {
      Ok(try_complex_accuracy(&args[0]).unwrap())
    }
    // Compound arithmetic in BinaryOp form: take the minimum accuracy
    // across both operands (mirrors the FunctionCall Plus/Times path
    // below). Otherwise `Plus[BigFloat, Times[BigFloat, I]]` — the
    // canonical form of `Complex[3.00``2, 4.00``2]` — would fall through
    // to Infinity even though both operands have finite accuracy.
    Expr::BinaryOp {
      op:
        crate::syntax::BinaryOperator::Plus
        | crate::syntax::BinaryOperator::Minus
        | crate::syntax::BinaryOperator::Times
        | crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      let mut min_finite: Option<f64> = None;
      for arg in [left.as_ref(), right.as_ref()] {
        let a = accuracy_ast(&[arg.clone()])?;
        match a {
          Expr::Identifier(ref n) if n == "Infinity" => {}
          Expr::Real(v) => {
            min_finite = Some(min_finite.map_or(v, |m| m.min(v)));
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Accuracy".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
      }
      Ok(match min_finite {
        Some(v) => Expr::Real(v),
        None => Expr::Identifier("Infinity".to_string()),
      })
    }
    // Symbolic identifiers (variables) have infinite accuracy
    Expr::Identifier(_) => Ok(Expr::Identifier("Infinity".to_string())),
    // For symbolic expressions and lists, take the minimum of the
    // children's accuracies. Wolfram: Accuracy[F[1.3, Pi, A]] → 15.840…,
    // picking the less-accurate Real over the exact Pi/A. Lists are
    // handled the same way: Accuracy[{1, 1.``5}] → 5.
    Expr::FunctionCall { args: fargs, .. } | Expr::List(fargs) => {
      let mut min_finite: Option<f64> = None;
      for arg in fargs {
        let a = accuracy_ast(&[arg.clone()])?;
        match a {
          Expr::Identifier(ref n) if n == "Infinity" => {}
          Expr::Real(v) => {
            min_finite = Some(min_finite.map_or(v, |m| m.min(v)));
          }
          _ => {
            // Some child still couldn't be reduced to a number — keep the
            // call symbolic rather than guess.
            return Ok(Expr::FunctionCall {
              name: "Accuracy".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
      }
      Ok(match min_finite {
        Some(v) => Expr::Real(v),
        None => Expr::Identifier("Infinity".to_string()),
      })
    }
    _ => Ok(Expr::Identifier("Infinity".to_string())),
  }
}

/// Detect a complex number `Plus[real_part, Times[imag_part, I]]` (in
/// either FunctionCall or BinaryOp form) where both parts have finite
/// accuracy, and return Wolfram's combined accuracy:
///   Acc = -Log10[Sqrt[10^(-2*Acc[re]) + 10^(-2*Acc[im])]].
/// Returns None if the structure isn't a recognised real+imag pair, if
/// either part has infinite accuracy, or if there's no I factor — in
/// those cases the caller falls back to the generic min-of-children path.
fn try_complex_accuracy(expr: &Expr) -> Option<Expr> {
  use crate::syntax::BinaryOperator;
  let plus_args: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      args.to_vec()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![*left.clone(), *right.clone()],
    _ => return None,
  };
  let split = |term: &Expr| -> Option<(bool, Expr)> {
    // Returns (is_imaginary, raw_part). For pure imaginary `I`, raw_part
    // is the implicit coefficient `1`.
    if matches!(term, Expr::Identifier(s) if s == "I") {
      return Some((true, Expr::Integer(1)));
    }
    let factors: Vec<&Expr> = match term {
      Expr::FunctionCall { name, args } if name == "Times" => {
        args.iter().collect()
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => vec![left.as_ref(), right.as_ref()],
      _ => return Some((false, term.clone())),
    };
    let i_count = factors
      .iter()
      .filter(|f| matches!(f, Expr::Identifier(s) if s == "I"))
      .count();
    if i_count == 0 {
      return Some((false, term.clone()));
    }
    if i_count != 1 {
      return None;
    }
    let others: Vec<Expr> = factors
      .iter()
      .filter(|f| !matches!(f, Expr::Identifier(s) if s == "I"))
      .map(|f| (*f).clone())
      .collect();
    let coeff = if others.is_empty() {
      Expr::Integer(1)
    } else if others.len() == 1 {
      others.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: others.into(),
      }
    };
    Some((true, coeff))
  };
  let (a_imag, a_part) = split(&plus_args[0])?;
  let (b_imag, b_part) = split(&plus_args[1])?;
  let (re_part, im_part) = match (a_imag, b_imag) {
    (false, true) => (a_part, b_part),
    (true, false) => (b_part, a_part),
    _ => return None,
  };
  let re_acc = match accuracy_ast(&[re_part]).ok()? {
    Expr::Real(v) => v,
    _ => return None,
  };
  let im_acc = match accuracy_ast(&[im_part]).ok()? {
    Expr::Real(v) => v,
    _ => return None,
  };
  let combined = -((10f64.powf(-2.0 * re_acc) + 10f64.powf(-2.0 * im_acc))
    .sqrt()
    .log10());
  Some(Expr::Real(combined))
}

/// PowerExpand[expr] - expand powers of products and powers
/// Rules: (a^b)^c -> a^(b*c), (a*b)^c -> a^c * b^c
pub fn power_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PowerExpand expects 1 or 2 arguments".into(),
    ));
  }
  // Second argument (Assumptions) is accepted but not used
  Ok(power_expand_recursive(&args[0]))
}

pub fn power_expand_recursive(expr: &Expr) -> Expr {
  // Helper to extract (base, exponent) from any Power representation
  let extract_power = |e: &Expr| -> Option<(Expr, Expr)> {
    match e {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } => Some((*left.clone(), *right.clone())),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((args[0].clone(), args[1].clone()))
      }
      // Sqrt[x] = Power[x, 1/2]
      expr if is_sqrt(expr).is_some() => {
        let sqrt_arg = is_sqrt(expr).unwrap();
        Some((
          sqrt_arg.clone(),
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
          },
        ))
      }
      _ => None,
    }
  };

  // Helper to recursively collect all Times factors (flattening nested Times
  // and converting Divide to Times with Power[..., -1])
  fn collect_times_factors(e: &Expr) -> Vec<Expr> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() >= 2 =>
      {
        args.iter().flat_map(collect_times_factors).collect()
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => {
        let mut factors = collect_times_factors(left);
        factors.extend(collect_times_factors(right));
        factors
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left,
        right,
      } => {
        let mut factors = collect_times_factors(left);
        factors.push(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
        factors
      }
      _ => vec![e.clone()],
    }
  }

  let extract_times = |e: &Expr| -> Option<Vec<Expr>> {
    let factors = collect_times_factors(e);
    if factors.len() >= 2 {
      Some(factors)
    } else {
      None
    }
  };

  match expr {
    _ if extract_power(expr).is_some() => {
      let (raw_base, raw_exp) = extract_power(expr).unwrap();
      let base = power_expand_recursive(&raw_base);
      let exp = power_expand_recursive(&raw_exp);

      // (a^b)^c -> a^(b*c)
      if let Some((inner_base, inner_exp)) = extract_power(&base) {
        let new_exp = match times_ast(&[inner_exp.clone(), exp.clone()]) {
          Ok(r) => r,
          Err(_) => Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(inner_exp),
            right: Box::new(exp),
          },
        };
        return match crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(inner_base),
          right: Box::new(new_exp),
        }) {
          Ok(r) => r,
          Err(_) => expr.clone(),
        };
      }

      // (a*b*...)^c -> a^c * b^c * ...
      // Also applies (a^r)^c -> a^(r*c) for each factor
      if let Some(factors) = extract_times(&base) {
        let expanded: Vec<Expr> = factors
          .iter()
          .map(|factor| {
            // If factor is itself a power, apply (a^r)^c -> a^(r*c)
            if let Some((inner_base, inner_exp)) = extract_power(factor) {
              let new_exp = match times_ast(&[inner_exp.clone(), exp.clone()]) {
                Ok(r) => r,
                Err(_) => Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(inner_exp),
                  right: Box::new(exp.clone()),
                },
              };
              match power_two(&inner_base, &new_exp) {
                Ok(r) => r,
                Err(_) => Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: Box::new(inner_base),
                  right: Box::new(new_exp),
                },
              }
            } else {
              match power_two(factor, &exp) {
                Ok(r) => r,
                Err(_) => Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: Box::new(factor.clone()),
                  right: Box::new(exp.clone()),
                },
              }
            }
          })
          .collect();
        return match times_ast(&expanded) {
          Ok(r) => r,
          Err(_) => Expr::FunctionCall {
            name: "Times".to_string(),
            args: expanded.into(),
          },
        };
      }

      // E^(a*Log[b]) -> b^a  and  E^Log[b] -> b
      if matches!(&base, Expr::Constant(c) if c == "E") {
        // Check if exponent is Log[b]
        if let Expr::FunctionCall { name: ln, args: la } = &exp
          && ln == "Log"
          && la.len() == 1
        {
          return la[0].clone();
        }
        // Check if exponent is a product containing Log[b]
        let exp_factors = collect_times_factors(&exp);
        if exp_factors.len() >= 2 {
          // Find the Log factor
          let log_idx = exp_factors.iter().position(|f| {
            matches!(f, Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1)
          });
          if let Some(idx) = log_idx
            && let Expr::FunctionCall { args: la, .. } = &exp_factors[idx]
          {
            let log_arg = la[0].clone();
            // Remaining factors form the new exponent
            let mut remaining: Vec<Expr> = exp_factors.clone();
            remaining.remove(idx);
            let new_exp = if remaining.len() == 1 {
              remaining.into_iter().next().unwrap()
            } else {
              match times_ast(&remaining) {
                Ok(r) => r,
                Err(_) => {
                  return Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Power,
                    left: Box::new(base),
                    right: Box::new(exp),
                  };
                }
              }
            };
            return match power_two(&log_arg, &new_exp) {
              Ok(r) => r,
              Err(_) => Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(log_arg),
                right: Box::new(new_exp),
              },
            };
          }
        }
      }

      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      }
    }
    // Log expansion rules (assuming positive reals):
    // Log[a*b*...] -> Log[a] + Log[b] + ...
    // Log[a^b] -> b*Log[a]
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      // First, recursively expand the argument
      let expanded_arg = power_expand_recursive(&args[0]);

      // Log of a quotient: Log[a/b] -> Log[a] - Log[b]
      // Convert a/b to Times[a, Power[b, -1]] and fall through to product rule
      let expanded_arg = match &expanded_arg {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left,
          right,
        } => Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            *left.clone(),
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: right.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          ]
          .into(),
        },
        _ => expanded_arg,
      };

      // Log of a product: Log[a*b*...] -> Log[a] + Log[b] + ...
      if let Some(factors) = extract_times(&expanded_arg) {
        let log_terms: Vec<Expr> = factors
          .iter()
          .map(|f| {
            power_expand_recursive(&Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![f.clone()].into(),
            })
          })
          .collect();
        return match plus_ast(&log_terms) {
          Ok(r) => r,
          Err(_) => Expr::FunctionCall {
            name: "Plus".to_string(),
            args: log_terms.into(),
          },
        };
      }

      // Log of a power: Log[a^b] -> b*Log[a]
      if let Some((base, exp)) = extract_power(&expanded_arg) {
        let log_expr = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![base].into(),
        };
        // Evaluate Log[base] to simplify cases like Log[E] -> 1
        let log_base = crate::evaluator::evaluate_expr_to_expr(&log_expr)
          .unwrap_or(log_expr);
        let log_base = power_expand_recursive(&log_base);
        return match times_ast(&[exp, log_base]) {
          Ok(r) => r,
          Err(_) => Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![expanded_arg].into(),
          },
        };
      }

      // Evaluate Log to simplify cases like Log[E] -> 1, Log[1] -> 0
      let log_expr = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![expanded_arg.clone()].into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&log_expr).unwrap_or(log_expr)
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> =
        args.iter().map(power_expand_recursive).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(power_expand_recursive(left)),
      right: Box::new(power_expand_recursive(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(power_expand_recursive(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(power_expand_recursive).collect())
    }
    _ => expr.clone(),
  }
}

// ─── Variables ──────────────────────────────────────────────────────

/// Variables[expr] - list of variables in a polynomial expression
pub fn variables_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Variables expects exactly 1 argument".into(),
    ));
  }
  // Variables only applies to polynomials. A relational or logical expression
  // (equation, inequality, And/Or/...) is not a polynomial, so wolframscript
  // returns {} rather than treating the whole expression as one variable.
  let head = crate::evaluator::evaluate_function_call_ast(
    "Head",
    std::slice::from_ref(&args[0]),
  )?;
  if let Expr::Identifier(h) = &head
    && matches!(
      h.as_str(),
      "Equal"
        | "Unequal"
        | "Less"
        | "Greater"
        | "LessEqual"
        | "GreaterEqual"
        | "Inequality"
        | "And"
        | "Or"
        | "Not"
        | "Nand"
        | "Nor"
        | "Xor"
        | "Implies"
        | "Equivalent"
        | "SameQ"
        | "UnsameQ"
    )
  {
    return Ok(Expr::List(vec![].into()));
  }
  let mut vars = Vec::new();
  collect_variables(&args[0], &mut vars);
  // Deduplicate while preserving order
  let mut seen = std::collections::HashSet::new();
  vars.retain(|v| seen.insert(crate::syntax::expr_to_string(v)));
  // For List input, sort in canonical order (alphabetical);
  // for non-List input, preserve first-appearance order (matching Wolfram).
  if matches!(&args[0], Expr::List(_)) {
    vars.sort_by(|a, b| {
      crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
    });
  }
  Ok(Expr::List(vars.into()))
}

pub fn collect_variables(expr: &Expr, vars: &mut Vec<Expr>) {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _) => {}
    Expr::Identifier(s) => {
      // Skip built-in constants
      if !matches!(
        s.as_str(),
        "True"
          | "False"
          | "Null"
          | "Pi"
          | "E"
          | "I"
          | "Infinity"
          | "ComplexInfinity"
          | "Indeterminate"
      ) {
        vars.push(expr.clone());
      }
    }
    Expr::List(items) => {
      for item in items {
        collect_variables(item, vars);
      }
    }
    Expr::BinaryOp { op, left, right } => {
      match op {
        crate::syntax::BinaryOperator::Plus
        | crate::syntax::BinaryOperator::Minus
        | crate::syntax::BinaryOperator::Times
        | crate::syntax::BinaryOperator::Power
        | crate::syntax::BinaryOperator::Divide => {
          collect_variables(left, vars);
          collect_variables(right, vars);
        }
        _ => {
          // Treat as atomic variable-like term
          vars.push(expr.clone());
        }
      }
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Plus" | "Times" | "Power" | "Rational" => {
          for arg in args {
            collect_variables(arg, vars);
          }
        }
        _ => {
          // Non-polynomial function (like Sin, Cos) — treat as variable
          vars.push(expr.clone());
        }
      }
    }
    Expr::UnaryOp { op: _, operand } => {
      collect_variables(operand, vars);
    }
    _ => {
      vars.push(expr.clone());
    }
  }
}

/// LinearRecurrence[ker, init, n] - generates a linear recurrence sequence of length n.
/// LinearRecurrence[ker, init, {nmin, nmax}] - returns elements nmin through nmax.
/// LinearRecurrence[ker, init, {n}] - returns the list containing only the nth element.
pub fn linear_recurrence_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "LinearRecurrence".to_string(),
      args: args.to_vec().into(),
    });
  }

  let kernel = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearRecurrence".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let init = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearRecurrence".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Parse the third argument: n, {n}, or {nmin, nmax}
  let (total_n, range) = match &args[2] {
    Expr::Integer(n) => (*n as usize, None),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        (n as usize, Some((n as usize, n as usize)))
      } else {
        return Ok(Expr::FunctionCall {
          name: "LinearRecurrence".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      if let (Some(nmin), Some(nmax)) =
        (expr_to_i128(&items[0]), expr_to_i128(&items[1]))
      {
        (nmax as usize, Some((nmin as usize, nmax as usize)))
      } else {
        return Ok(Expr::FunctionCall {
          name: "LinearRecurrence".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearRecurrence".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut seq = init.clone();

  // Extend sequence to total_n elements
  while seq.len() < total_n {
    let mut next = Expr::Integer(0);
    for (i, coeff) in kernel.iter().enumerate() {
      let idx = seq.len() - 1 - i;
      let term = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(seq[idx].clone()),
      };
      next = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(next),
        right: Box::new(term),
      };
    }
    // Evaluate the expression to simplify
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&next)?;
    seq.push(evaluated);
  }

  match range {
    None => Ok(Expr::List(seq[..total_n].to_vec().into())),
    Some((nmin, nmax)) => Ok(Expr::List(seq[nmin - 1..nmax].to_vec().into())),
  }
}

/// EuclideanDistance[u, v] - Euclidean distance between two points
/// WarpingDistance[s1, s2] - dynamic time warping distance between two numeric
/// sequences, with the default Manhattan (|a - b|) local cost. The result is
/// the accumulated cost along the optimal warping path (a Real, matching
/// wolframscript). Returns None when an argument isn't a numeric sequence.
fn warping_distance_value(a: &Expr, b: &Expr) -> Option<f64> {
  let to_vec = |e: &Expr| -> Option<Vec<f64>> {
    match e {
      Expr::List(items) => items
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect(),
      _ => None,
    }
  };
  let a = to_vec(a)?;
  let b = to_vec(b)?;
  let (n, m) = (a.len(), b.len());
  if n == 0 || m == 0 {
    return None;
  }
  let inf = f64::INFINITY;
  let mut dp = vec![vec![inf; m + 1]; n + 1];
  dp[0][0] = 0.0;
  for i in 1..=n {
    for j in 1..=m {
      let cost = (a[i - 1] - b[j - 1]).abs();
      dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
    }
  }
  Some(dp[n][m])
}

pub fn warping_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    if let Some(d) = warping_distance_value(&args[0], &args[1]) {
      return Ok(Expr::Real(d));
    }
    // Identify the offending (non-numeric-vector) argument, matching the
    // wolframscript WarpingDistance::invarg message.
    let is_numeric_vec = |e: &Expr| {
      matches!(e, Expr::List(items) if !items.is_empty()
        && items.iter().all(|x| crate::functions::math_ast::try_eval_to_f64(x).is_some()))
    };
    if let Some(bad) = args.iter().find(|a| !is_numeric_vec(a)) {
      crate::emit_message(&format!(
        "WarpingDistance::invarg: Expecting a real-valued numeric or Boolean \
         vector or matrix instead of {}.",
        crate::syntax::format_expr(bad, crate::syntax::ExprForm::Output)
      ));
    }
  }
  Ok(Expr::FunctionCall {
    name: "WarpingDistance".to_string(),
    args: args.to_vec().into(),
  })
}

pub fn euclidean_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "EuclideanDistance expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(u), Expr::List(v)) => {
      if u.len() != v.len() {
        return Err(InterpreterError::EvaluationError(
          "EuclideanDistance: vectors must have the same length".into(),
        ));
      }
      // Build Sqrt[Sum[Abs[u_i - v_i]^2]]
      let mut sum_args = Vec::new();
      for (ui, vi) in u.iter().zip(v.iter()) {
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[ui.clone(), vi.clone()],
        )?;
        let abs_diff =
          crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
        let sq = crate::evaluator::evaluate_function_call_ast(
          "Power",
          &[abs_diff, Expr::Integer(2)],
        )?;
        sum_args.push(sq);
      }
      let sum =
        crate::evaluator::evaluate_function_call_ast("Plus", &sum_args)?;
      crate::evaluator::evaluate_function_call_ast("Sqrt", &[sum])
    }
    _ => {
      // Scalar distance: Abs[u - v]
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[args[0].clone(), args[1].clone()],
      )?;
      crate::evaluator::evaluate_function_call_ast("Abs", &[diff])
    }
  }
}

/// SquaredEuclideanDistance[u, v] - squared Euclidean distance
pub fn squared_euclidean_distance_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SquaredEuclideanDistance expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(u), Expr::List(v)) => {
      if u.len() != v.len() {
        return Err(InterpreterError::EvaluationError(
          "SquaredEuclideanDistance: vectors must have the same length".into(),
        ));
      }
      let mut sum_args = Vec::new();
      for (ui, vi) in u.iter().zip(v.iter()) {
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[ui.clone(), vi.clone()],
        )?;
        let abs_diff =
          crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
        let sq = crate::evaluator::evaluate_function_call_ast(
          "Power",
          &[abs_diff, Expr::Integer(2)],
        )?;
        sum_args.push(sq);
      }
      crate::evaluator::evaluate_function_call_ast("Plus", &sum_args)
    }
    _ => {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[args[0].clone(), args[1].clone()],
      )?;
      let abs_diff =
        crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
      crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[abs_diff, Expr::Integer(2)],
      )
    }
  }
}

/// Core DFT computation shared by Fourier and InverseFourier.
/// `sign` is +1 for Fourier, -1 for InverseFourier (before applying `b`).
/// FourierParameters {a, b}: F_s = n^((a-1)/2) * sum_{r=0}^{n-1} u_r * exp(2*pi*i*b*(r*s)/n)
pub fn dft_core(
  data: &[(f64, f64)],
  param_a: f64,
  param_b: f64,
  inverse: bool,
) -> Vec<(f64, f64)> {
  let n = data.len();
  if n == 0 {
    return vec![];
  }
  let nf = n as f64;
  // For Fourier: scaling = n^((a-1)/2), exponent sign from b
  // For InverseFourier: scaling = n^((-1-a)/2), exponent sign from -b
  let (scaling, exp_sign) = if inverse {
    (nf.powf((-1.0 - param_a) / 2.0), -param_b)
  } else {
    (nf.powf((param_a - 1.0) / 2.0), param_b)
  };

  // For power-of-2 sizes, use Cooley-Tukey FFT (O(n log n)) instead of
  // the O(n^2) DFT.
  if n >= 2 && n.is_power_of_two() {
    let mut x: Vec<(f64, f64)> = data.to_vec();
    fft_pow2_in_place(&mut x, exp_sign);
    for (re, im) in &mut x {
      *re *= scaling;
      *im *= scaling;
    }
    return x;
  }

  let two_pi_over_n = 2.0 * std::f64::consts::PI / nf;
  let mut result = Vec::with_capacity(n);

  for s in 0..n {
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for r in 0..n {
      let angle = two_pi_over_n * exp_sign * (r as f64) * (s as f64);
      let (sin_a, cos_a) = angle.sin_cos();
      let (ur, ui) = data[r];
      // (ur + ui*i) * (cos_a + sin_a*i) = (ur*cos_a - ui*sin_a) + (ur*sin_a + ui*cos_a)*i
      sum_re += ur * cos_a - ui * sin_a;
      sum_im += ur * sin_a + ui * cos_a;
    }
    result.push((scaling * sum_re, scaling * sum_im));
  }
  result
}

/// In-place Cooley-Tukey radix-2 FFT. `data.len()` must be a power of 2.
/// `exp_sign` is +1 or -1, selecting forward vs inverse twiddle direction.
/// Scaling is left to the caller.
pub(crate) fn fft_pow2_in_place(data: &mut [(f64, f64)], exp_sign: f64) {
  let n = data.len();
  // Bit-reverse permutation.
  let mut j = 0usize;
  for i in 1..n {
    let mut bit = n >> 1;
    while j & bit != 0 {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if i < j {
      data.swap(i, j);
    }
  }
  // Iterative butterflies.
  let mut size = 2;
  while size <= n {
    let half = size / 2;
    let theta = exp_sign * 2.0 * std::f64::consts::PI / size as f64;
    let (sin_t, cos_t) = theta.sin_cos();
    let mut i = 0;
    while i < n {
      // Twiddle for this butterfly group.
      let mut w_re = 1.0_f64;
      let mut w_im = 0.0_f64;
      for k in 0..half {
        let a = data[i + k];
        let b = data[i + k + half];
        let t_re = w_re * b.0 - w_im * b.1;
        let t_im = w_re * b.1 + w_im * b.0;
        data[i + k] = (a.0 + t_re, a.1 + t_im);
        data[i + k + half] = (a.0 - t_re, a.1 - t_im);
        // Advance twiddle.
        let _ = k;
        let new_w_re = w_re * cos_t - w_im * sin_t;
        let new_w_im = w_re * sin_t + w_im * cos_t;
        w_re = new_w_re;
        w_im = new_w_im;
      }
      i += size;
    }
    size <<= 1;
  }
}

/// Round a floating-point number to clean up near-integer/near-half values.
/// This accounts for floating-point errors in DFT trig computations.
pub fn fourier_round(x: f64) -> f64 {
  if x.abs() < 1e-14 {
    return 0.0;
  }
  // Check if x is very close to an integer or half-integer
  let rounded = x.round();
  if (x - rounded).abs() < 1e-14 {
    return rounded;
  }
  // Check half-integer
  let doubled = (x * 2.0).round();
  if (x * 2.0 - doubled).abs() < 1e-13 {
    return doubled / 2.0;
  }
  x
}

/// Build an Expr for a Fourier/InverseFourier result element.
/// If `force_complex` is true, always output as Complex (even if im == 0).
pub fn fourier_result_to_expr(re: f64, im: f64, force_complex: bool) -> Expr {
  let re = fourier_round(re);
  let im = fourier_round(im);

  if force_complex || im != 0.0 {
    // Build via Complex[re, im] which the evaluator will format as a + b*I
    crate::evaluator::evaluate_function_call_ast(
      "Complex",
      &[Expr::Real(re), Expr::Real(im)],
    )
    .unwrap_or_else(|_| build_complex_float_expr(re, im))
  } else {
    Expr::Real(re)
  }
}

/// Parse FourierParameters option from args, returning (a, b).
/// Default is {0, 1}.
pub fn parse_fourier_parameters(
  args: &[Expr],
) -> Result<(f64, f64), InterpreterError> {
  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == "FourierParameters")
    {
      if let Expr::List(params) = replacement.as_ref()
        && params.len() == 2
      {
        let a = try_eval_to_f64(&params[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "FourierParameters: first parameter must be numeric".into(),
          )
        })?;
        let b = try_eval_to_f64(&params[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "FourierParameters: second parameter must be numeric".into(),
          )
        })?;
        return Ok((a, b));
      }
      return Err(InterpreterError::EvaluationError(
        "FourierParameters must be a list of two numbers".into(),
      ));
    }
  }
  Ok((0.0, 1.0)) // default
}

/// Shared implementation for Fourier and InverseFourier
pub fn fourier_impl(
  func_name: &str,
  args: &[Expr],
  inverse: bool,
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects 1 or 2 arguments",
      func_name
    )));
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      crate::emit_message(&format!(
        "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        func_name,
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(Expr::FunctionCall {
        name: func_name.to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract numeric values as complex pairs
  let mut data: Vec<(f64, f64)> = Vec::with_capacity(items.len());
  for item in items {
    if let Some((re, im)) = try_extract_complex_float(item) {
      data.push((re, im));
    } else {
      crate::emit_message(&format!(
        "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        func_name,
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(Expr::FunctionCall {
        name: func_name.to_string(),
        args: args.to_vec().into(),
      });
    }
  }

  let (param_a, param_b) = parse_fourier_parameters(&args[1..])?;
  let result = dft_core(&data, param_a, param_b, inverse);

  // Determine if any element has nonzero imaginary part
  let any_complex = result.iter().any(|(_, im)| {
    let im_r = fourier_round(*im);
    im_r != 0.0
  });

  let exprs: Vec<Expr> = result
    .iter()
    .map(|&(re, im)| fourier_result_to_expr(re, im, any_complex))
    .collect();

  Ok(Expr::List(exprs.into()))
}

/// Fourier[list] or Fourier[list, opts] - Discrete Fourier transform
pub fn fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fourier_impl("Fourier", args, false)
}

/// InverseFourier[list] or InverseFourier[list, opts] - Inverse discrete Fourier transform
pub fn inverse_fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fourier_impl("InverseFourier", args, true)
}

/// Core discrete cosine transform of a real list `u`.
/// `m` selects the DCT method: 1, 2 (default), 3, or 4.
/// Mirrors the normalisation used by Wolfram's `FourierDCT`.
fn dct_core(u: &[f64], m: i64) -> Vec<f64> {
  let n = u.len();
  let nf = n as f64;
  let pi = std::f64::consts::PI;
  let mut out = Vec::with_capacity(n);
  for s in 0..n {
    let sf = s as f64;
    let mut acc = 0.0;
    match m {
      1 => {
        if n == 1 {
          acc = u[0];
        } else {
          for (r, &ur) in u.iter().enumerate() {
            let w = if r == 0 || r == n - 1 { 0.5 } else { 1.0 };
            acc += w * ur * (pi * (r as f64) * sf / (nf - 1.0)).cos();
          }
          acc *= (2.0 / (nf - 1.0)).sqrt();
        }
      }
      3 => {
        for (r, &ur) in u.iter().enumerate() {
          let w = if r == 0 { 0.5 } else { 1.0 };
          acc +=
            w * ur * (pi * (2.0 * sf + 1.0) * (r as f64) / (2.0 * nf)).cos();
        }
        acc *= 2.0 / nf.sqrt();
      }
      4 => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur
            * (pi * (2.0 * (r as f64) + 1.0) * (2.0 * sf + 1.0) / (4.0 * nf))
              .cos();
        }
        acc *= (2.0 / nf).sqrt();
      }
      // default: type 2
      _ => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur * (pi * (2.0 * (r as f64) + 1.0) * sf / (2.0 * nf)).cos();
        }
        acc /= nf.sqrt();
      }
    }
    out.push(acc);
  }
  out
}

/// FourierDCT[list] or FourierDCT[list, m] — discrete cosine transform.
/// `m` (1..4) selects the DCT method; default is 2.
pub fn fourier_dct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FourierDCT expects 1 or 2 arguments".into(),
    ));
  }

  let unevaluated = || {
    crate::emit_message(&format!(
      "FourierDCT::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      crate::syntax::expr_to_string(&args[0])
    ));
    Ok(Expr::FunctionCall {
      name: "FourierDCT".to_string(),
      args: args.to_vec().into(),
    })
  };

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return unevaluated(),
  };

  // Extract real numeric values. Any non-real entry leaves it unevaluated.
  let mut data: Vec<f64> = Vec::with_capacity(items.len());
  for item in items.iter() {
    match try_extract_complex_float(item) {
      Some((re, im)) if im == 0.0 => data.push(re),
      _ => return unevaluated(),
    }
  }

  // Optional method argument (1..4); default 2.
  let m = if args.len() == 2 {
    match try_eval_to_f64(&args[1]) {
      Some(v) if (v - v.round()).abs() < 1e-12 && (1.0..=4.0).contains(&v) => {
        v.round() as i64
      }
      _ => return unevaluated(),
    }
  } else {
    2
  };

  let result = dct_core(&data, m);
  let exprs: Vec<Expr> = result.iter().map(|&x| Expr::Real(x)).collect();
  Ok(Expr::List(exprs.into()))
}

fn dst_core(u: &[f64], m: i64) -> Vec<f64> {
  let n = u.len();
  let nf = n as f64;
  let pi = std::f64::consts::PI;
  let mut out = Vec::with_capacity(n);
  for s in 0..n {
    let sf = s as f64;
    let mut acc = 0.0;
    match m {
      1 => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur * (pi * ((r + 1) as f64) * (sf + 1.0) / (nf + 1.0)).sin();
        }
        acc *= (2.0 / (nf + 1.0)).sqrt();
      }
      3 => {
        // Transpose of type 2; the Nyquist term (r = n-1) is halved
        for (r, &ur) in u.iter().enumerate() {
          let w = if r == n - 1 { 0.5 } else { 1.0 };
          acc += w
            * ur
            * (pi * (2.0 * sf + 1.0) * ((r + 1) as f64) / (2.0 * nf)).sin();
        }
        acc *= 2.0 / nf.sqrt();
      }
      4 => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur
            * (pi * (2.0 * (r as f64) + 1.0) * (2.0 * sf + 1.0) / (4.0 * nf))
              .sin();
        }
        acc *= (2.0 / nf).sqrt();
      }
      // default: type 2
      _ => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur
            * (pi * (2.0 * (r as f64) + 1.0) * (sf + 1.0) / (2.0 * nf)).sin();
        }
        acc /= nf.sqrt();
      }
    }
    out.push(acc);
  }
  out
}

/// FourierDST[list] or FourierDST[list, m] — discrete sine transform.
/// `m` (1..4) selects the DST method; default is 2, mirroring
/// FourierDCT's conventions (including the numericization of exact
/// input and the ::fftl message for non-numeric arguments).
pub fn fourier_dst_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FourierDST expects 1 or 2 arguments".into(),
    ));
  }

  let unevaluated = || {
    crate::emit_message(&format!(
      "FourierDST::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      crate::syntax::expr_to_string(&args[0])
    ));
    Ok(Expr::FunctionCall {
      name: "FourierDST".to_string(),
      args: args.to_vec().into(),
    })
  };

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return unevaluated(),
  };

  let mut data: Vec<f64> = Vec::with_capacity(items.len());
  for item in items.iter() {
    match try_extract_complex_float(item) {
      Some((re, im)) if im == 0.0 => data.push(re),
      _ => return unevaluated(),
    }
  }

  let m = if args.len() == 2 {
    match try_eval_to_f64(&args[1]) {
      Some(v) if (v - v.round()).abs() < 1e-12 && (1.0..=4.0).contains(&v) => {
        v.round() as i64
      }
      _ => return unevaluated(),
    }
  } else {
    2
  };

  let result = dst_core(&data, m);
  let exprs: Vec<Expr> = result.iter().map(|&x| Expr::Real(x)).collect();
  Ok(Expr::List(exprs.into()))
}

/// Wynn epsilon algorithm for series acceleration.
/// Takes a sequence of partial sums and returns an accelerated estimate.
/// Polynomial extrapolation using Neville's algorithm.
/// Given points (x_i, y_i) representing (1/n, S_n), extrapolate to x=0.
pub fn neville_extrapolation(xs: &[f64], ys: &[f64]) -> f64 {
  let n = xs.len();
  if n == 0 {
    return 0.0;
  }
  if n == 1 {
    return ys[0];
  }

  let mut c = ys.to_vec();
  for j in 1..n {
    for i in (j..n).rev() {
      // Neville's algorithm for interpolation at x=0:
      // c[i] = (x_i * c[i-1] - x_{i-j} * c[i]) / (x_i - x_{i-j})
      // Since we evaluate at x=0, this simplifies to:
      let denom = xs[i] - xs[i - j];
      if denom.abs() < 1e-300 {
        continue;
      }
      c[i] = (xs[i] * c[i - 1] - xs[i - j] * c[i]) / denom;
    }
  }
  c[n - 1]
}

/// NSum[expr, {i, min, max}] - Numerical summation
pub fn nsum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(Expr::FunctionCall {
      name: "NSum".to_string(),
      args: args.to_vec().into(),
    });
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let items = match iter_spec {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let var_name = match &items[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if items.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "NSum".to_string(),
      args: args.to_vec().into(),
    });
  }

  let min_val = match try_eval_to_f64(&items[1]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Check for infinite sum
  let is_infinite = matches!(&items[2], Expr::Identifier(s) if s == "Infinity");

  if is_infinite {
    // Numerical infinite sum using polynomial extrapolation in 1/n
    // 1. Compute partial sums at several checkpoint values of n
    // 2. Extrapolate to n → ∞ (i.e., 1/n → 0) using Neville's algorithm
    let checkpoints: Vec<i64> = vec![50, 100, 150, 200, 300, 400, 500];
    let mut running_sum = 0.0_f64;
    let mut checkpoint_idx = 0;
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    let max_n = *checkpoints.last().unwrap();
    for i in min_val..(min_val + max_n) {
      let sub_val = Expr::Integer(i as i128);
      let substituted =
        crate::syntax::substitute_variable(body, &var_name, &sub_val);
      let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
      let term = match try_eval_to_f64(&val) {
        Some(f) => f,
        None => {
          return Ok(Expr::FunctionCall {
            name: "NSum".to_string(),
            args: args.to_vec().into(),
          });
        }
      };

      if !term.is_finite() {
        break;
      }
      running_sum += term;

      let current_n = i - min_val + 1;
      if checkpoint_idx < checkpoints.len()
        && current_n == checkpoints[checkpoint_idx]
      {
        xs.push(1.0 / current_n as f64);
        ys.push(running_sum);
        checkpoint_idx += 1;
      }
    }

    if xs.is_empty() {
      return Ok(Expr::Real(running_sum));
    }

    let result = neville_extrapolation(&xs, &ys);
    return Ok(Expr::Real(result));
  }

  // Finite sum
  let max_val = match try_eval_to_f64(&items[2]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut sum = 0.0_f64;
  for i in min_val..=max_val {
    let sub_val = Expr::Integer(i as i128);
    let substituted =
      crate::syntax::substitute_variable(body, &var_name, &sub_val);
    let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
    let term = match try_eval_to_f64(&val) {
      Some(f) => f,
      None => {
        return Ok(Expr::FunctionCall {
          name: "NSum".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    sum += term;
  }

  Ok(Expr::Real(sum))
}

/// NProduct[f, {i, imin, imax}] - Numerical product
pub fn nproduct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(Expr::FunctionCall {
      name: "NProduct".to_string(),
      args: args.to_vec().into(),
    });
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let items = match iter_spec {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let var_name = match &items[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if items.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "NProduct".to_string(),
      args: args.to_vec().into(),
    });
  }

  let min_val = match try_eval_to_f64(&items[1]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Detect `Infinity` upper bound and switch to Wynn-epsilon acceleration
  // over partial products. The standard `try_eval_to_f64` doesn't recognise
  // the `Infinity` identifier, so check it explicitly first.
  let is_infinity_max =
    matches!(&items[2], Expr::Identifier(s) if s == "Infinity");

  let eval_term = |i: i64| -> Result<Option<f64>, InterpreterError> {
    let sub_val = Expr::Integer(i as i128);
    let substituted =
      crate::syntax::substitute_variable(body, &var_name, &sub_val);
    let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
    Ok(try_eval_to_f64(&val))
  };

  if is_infinity_max {
    // Iterate enough terms to drive the residual small; for products with
    // O(1/n^2) tail (the common case), 100 000 terms gives ~10 digits.
    // After collecting the log-partial-sums, apply Wynn-epsilon to the
    // remaining tail estimates to extract a near-machine-precision limit.
    let n_terms: usize = 20_000;
    let mut log_product = 0.0_f64;
    // Sample log-partial-sums on a geometric schedule so Wynn-epsilon has
    // enough room to extract the asymptotic constant.
    let mut samples: Vec<f64> = Vec::with_capacity(64);
    let mut next_sample = 1usize;
    let sample_factor = 1.3_f64;
    for k in 0..n_terms {
      let i = min_val + k as i64;
      let term = match eval_term(i)? {
        Some(f) => f,
        None => {
          return Ok(Expr::FunctionCall {
            name: "NProduct".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      if term <= 0.0 || !term.is_finite() {
        return Ok(Expr::FunctionCall {
          name: "NProduct".to_string(),
          args: args.to_vec().into(),
        });
      }
      log_product += term.ln();
      if k + 1 >= next_sample {
        samples.push(log_product);
        next_sample =
          ((next_sample as f64) * sample_factor).ceil() as usize + 1;
      }
    }
    if samples.is_empty() {
      samples.push(log_product);
    }

    // Apply Wynn's epsilon algorithm to the log-partial-sums.
    let result_log = {
      let n = samples.len();
      if n >= 3 {
        let mut eps_prev: Vec<f64> = vec![0.0; n + 1];
        let mut eps_curr: Vec<f64> = samples.clone();
        let mut best = *samples.last().unwrap();
        let mut k: usize = 0;
        while eps_curr.len() >= 2 {
          let len_next = eps_curr.len() - 1;
          let mut eps_next: Vec<f64> = vec![0.0; len_next];
          for j in 0..len_next {
            let diff = eps_curr[j + 1] - eps_curr[j];
            if diff.abs() < 1e-300 || !diff.is_finite() {
              eps_next[j] = f64::INFINITY;
            } else {
              eps_next[j] =
                eps_prev.get(j + 1).copied().unwrap_or(0.0) + 1.0 / diff;
            }
          }
          if k.is_multiple_of(2) && k > 0 {
            let candidate = eps_curr[0];
            if candidate.is_finite() {
              best = candidate;
            }
          }
          eps_prev = eps_curr;
          eps_curr = eps_next;
          k += 1;
        }
        best
      } else {
        log_product
      }
    };
    return Ok(Expr::Real(result_log.exp()));
  }

  let max_val = match try_eval_to_f64(&items[2]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut product = 1.0_f64;
  for i in min_val..=max_val {
    let term = match eval_term(i)? {
      Some(f) => f,
      None => {
        return Ok(Expr::FunctionCall {
          name: "NProduct".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    product *= term;
  }

  Ok(Expr::Real(product))
}

/// ManhattanDistance[u, v] - Manhattan (L1) distance between two points
pub fn manhattan_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ManhattanDistance expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(u), Expr::List(v)) => {
      if u.len() != v.len() {
        return Err(InterpreterError::EvaluationError(
          "ManhattanDistance: vectors must have the same length".into(),
        ));
      }
      let mut abs_diffs = Vec::new();
      for (ui, vi) in u.iter().zip(v.iter()) {
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[ui.clone(), vi.clone()],
        )?;
        let abs = crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
        abs_diffs.push(abs);
      }
      crate::evaluator::evaluate_function_call_ast("Plus", &abs_diffs)
    }
    _ => {
      // Scalar distance: Abs[u - v]
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[args[0].clone(), args[1].clone()],
      )?;
      crate::evaluator::evaluate_function_call_ast("Abs", &[diff])
    }
  }
}

/// Compute the error function erf(x) using BigFloat arithmetic.
/// Uses the Taylor series: erf(x) = (2/sqrt(π)) * Σ_{n=0}^{∞} (-1)^n * x^(2n+1) / (n! * (2n+1))
/// Compute erfc(x) for x > 0 using the continued fraction representation.
/// erfc(x) = exp(-x²) / (f * sqrt(π)) where f is computed via modified Lentz's method.
fn bigfloat_erfc_cf(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Use extra guard bits for intermediate computation
  let work_bits = bits + 64;

  let half = BigFloat::from_i32(1, work_bits).div(
    &BigFloat::from_i32(2, work_bits),
    work_bits,
    rm,
  );

  // Modified Lentz's method for the continued fraction
  // erfc(x) = (exp(-x²)/sqrt(π)) * 1/(x + 1/(2x + 2/(x + 3/(2x + ...))))
  // Using: a_n = n * 0.5, b_n = x
  let mut f = x.clone();
  let mut c = x.clone();
  let mut d = BigFloat::from_i32(0, work_bits);

  let max_iterations = work_bits * 2 + 200;
  for n in 1..max_iterations {
    // a_n = n * 0.5
    let a_n = BigFloat::from_i32(n as i32, work_bits).mul(&half, work_bits, rm);

    // d = x + a_n * d
    d = x.add(&a_n.mul(&d, work_bits, rm), work_bits, rm);
    // Guard against zero
    if d.is_zero() {
      d = BigFloat::min_positive_normal(work_bits);
    }

    // c = x + a_n / c
    c = x.add(&a_n.div(&c, work_bits, rm), work_bits, rm);
    if c.is_zero() {
      c = BigFloat::min_positive_normal(work_bits);
    }

    // d = 1/d
    d = BigFloat::from_i32(1, work_bits).div(&d, work_bits, rm);
    let delta = c.mul(&d, work_bits, rm);
    f = f.mul(&delta, work_bits, rm);

    // Check convergence: |delta - 1| is negligible
    let one = BigFloat::from_i32(1, work_bits);
    let diff = delta.sub(&one, work_bits, rm).abs();
    if diff.is_zero() {
      break;
    }
    if let Some(diff_exp) = diff.exponent()
      && diff_exp < -(work_bits as i32)
    {
      break;
    }
  }

  // erfc(x) = exp(-x²) / (f * sqrt(π))
  let x2 = x.mul(x, work_bits, rm);
  let neg_x2 = x2.neg();
  let exp_neg_x2 = neg_x2.exp(work_bits, rm, cc);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let denom = f.mul(&sqrt_pi, work_bits, rm);
  exp_neg_x2.div(&denom, bits, rm)
}

fn bigfloat_erf(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(0, bits);
  }

  // erf is odd: erf(-x) = -erf(x)
  let is_negative = x.is_negative();
  let x_abs = x.abs();

  // For large |x|, use the continued fraction for erfc(x) and compute erf = 1 - erfc.
  // The Taylor series suffers from catastrophic cancellation for large arguments.
  let four = BigFloat::from_i32(4, bits);
  if x_abs.cmp(&four) == Some(1) {
    // |x| > 4: use continued fraction
    let erfc_val = bigfloat_erfc_cf(&x_abs, bits, rm, cc);
    let one = BigFloat::from_i32(1, bits);
    let result = one.sub(&erfc_val, bits, rm);
    return if is_negative { result.neg() } else { result };
  }

  // For small |x| (≤ 4), use the Taylor series with extra guard bits to handle cancellation.
  // With |x| ≤ 4, the peak term is ~exp(x²/2) ≈ exp(8) ≈ 2981, needing ~12 extra bits.
  // We use 64 guard bits for safety.
  let work_bits = bits + 64;

  // Taylor series: term_0 = x, term_n = term_{n-1} * x^2 / n
  // contribution_n = term_n / (2n+1), alternating sign
  // sum = Σ (-1)^n * contribution_n
  let x2 = x_abs.mul(&x_abs, work_bits, rm);
  let mut term = x_abs.clone();
  let mut sum = x_abs.clone();

  let max_iterations = work_bits * 2 + 100;
  for n in 1..max_iterations {
    term = term.mul(&x2, work_bits, rm);
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    term = term.div(&n_bf, work_bits, rm);

    let denom = BigFloat::from_i32((2 * n + 1) as i32, work_bits);
    let contribution = term.div(&denom, work_bits, rm);

    if n % 2 == 1 {
      sum = sum.sub(&contribution, work_bits, rm);
    } else {
      sum = sum.add(&contribution, work_bits, rm);
    }

    if contribution.is_zero() {
      break;
    }
    if let (Some(c_exp), Some(s_exp)) =
      (contribution.exponent(), sum.exponent())
      && s_exp - c_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Multiply by 2/sqrt(π), round to final precision
  let two = BigFloat::from_i32(2, work_bits);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let factor = two.div(&sqrt_pi, work_bits, rm);
  let result = sum.mul(&factor, bits, rm);

  if is_negative { result.neg() } else { result }
}

/// Compute erfc(x) with arbitrary precision.
/// For large x, uses continued fraction directly. For small x, uses 1 - erf(x).
fn bigfloat_erfc(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(1, bits);
  }

  let is_negative = x.is_negative();
  let x_abs = x.abs();

  let four = BigFloat::from_i32(4, bits);
  let result = if x_abs.cmp(&four) == Some(1) {
    // |x| > 4: use continued fraction directly for best precision
    bigfloat_erfc_cf(&x_abs, bits, rm, cc)
  } else {
    // |x| <= 4: compute via 1 - erf(x)
    let erf_val = bigfloat_erf(&x_abs, bits, rm, cc);
    let one = BigFloat::from_i32(1, bits);
    one.sub(&erf_val, bits, rm)
  };

  // erfc(-x) = 2 - erfc(x)
  if is_negative {
    let two = BigFloat::from_i32(2, bits);
    two.sub(&result, bits, rm)
  } else {
    result
  }
}

/// Compute erfi(x) with arbitrary precision.
/// erfi(x) = (2/sqrt(pi)) * sum_{n=0}^{inf} x^(2n+1) / (n! * (2n+1))
/// Unlike erf, the terms do NOT alternate in sign.
fn bigfloat_erfi(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(0, bits);
  }

  // erfi is odd: erfi(-x) = -erfi(x)
  let is_negative = x.is_negative();
  let x_abs = x.abs();

  let work_bits = bits + 64;

  // Taylor series: term_0 = x, term_n = term_{n-1} * x^2 / n
  // contribution_n = term_n / (2n+1), all terms positive (no alternating sign)
  let x2 = x_abs.mul(&x_abs, work_bits, rm);
  let mut term = x_abs.clone();
  let mut sum = x_abs.clone();

  let max_iterations = work_bits * 2 + 100;
  for n in 1..max_iterations {
    term = term.mul(&x2, work_bits, rm);
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    term = term.div(&n_bf, work_bits, rm);

    let denom = BigFloat::from_i32((2 * n + 1) as i32, work_bits);
    let contribution = term.div(&denom, work_bits, rm);

    sum = sum.add(&contribution, work_bits, rm);

    if contribution.is_zero() {
      break;
    }
    if let (Some(c_exp), Some(s_exp)) =
      (contribution.exponent(), sum.exponent())
      && s_exp - c_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Multiply by 2/sqrt(π), round to final precision
  let two = BigFloat::from_i32(2, work_bits);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let factor = two.div(&sqrt_pi, work_bits, rm);
  let result = sum.mul(&factor, bits, rm);

  if is_negative { result.neg() } else { result }
}

/// Compute the exponential integral Ei(x) using BigFloat arithmetic.
/// For real x: Ei(x) = γ + ln|x| + Σ_{n=1}^{∞} x^n / (n * n!)
/// where γ is the Euler-Mascheroni constant.
fn bigfloat_exp_integral_ei(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  let work_bits = bits + 64;

  // γ (Euler-Mascheroni constant)
  let euler_gamma = compute_euler_gamma(work_bits, rm, cc);

  // ln(|x|)
  let ln_x = x.abs().ln(work_bits, rm, cc);

  // Power series: Σ_{n=1}^{∞} x^n / (n * n!)
  let mut sum = BigFloat::from_i32(0, work_bits);
  let mut x_pow = x.clone(); // x^1
  let mut factorial = BigFloat::from_i32(1, work_bits); // 1!

  let max_iterations = bits * 2 + 100;
  for n in 1..max_iterations {
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    if n > 1 {
      x_pow = x_pow.mul(x, work_bits, rm);
      factorial = factorial.mul(&n_bf, work_bits, rm);
    }
    // term = x^n / (n * n!)
    let denom = n_bf.mul(&factorial, work_bits, rm);
    let term = x_pow.div(&denom, work_bits, rm);
    sum = sum.add(&term, work_bits, rm);

    if term.abs().is_zero() {
      break;
    }
    if let (Some(t_exp), Some(s_exp)) =
      (term.abs().exponent(), sum.abs().exponent())
      && s_exp - t_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Ei(x) = γ + ln(|x|) + sum (final result rounded to requested bits)
  euler_gamma.add(&ln_x, work_bits, rm).add(&sum, bits, rm)
}

/// Compute the complex exponential integral Ei(z) using BigFloat arithmetic.
/// For complex z = a + bi: Ei(z) = γ + Log(z) + Σ_{n=1}^{∞} z^n / (n * n!)
/// where γ is the Euler-Mascheroni constant and Log is the complex logarithm.
fn complex_exp_integral_ei(
  a: astro_float::BigFloat,
  b: astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use astro_float::BigFloat;

  // Use many extra guard bits to ensure the rounding to `bits` is correct
  let work_bits = bits + 256;

  // γ (Euler-Mascheroni constant) - purely real
  let euler_gamma = compute_euler_gamma(work_bits, rm, cc);

  // Complex Log(z) = ln|z| + i*arg(z)
  // |z| = sqrt(a^2 + b^2), arg(z) = atan2(b, a)
  let abs_sq =
    a.mul(&a, work_bits, rm)
      .add(&b.mul(&b, work_bits, rm), work_bits, rm);
  let abs_z = abs_sq.sqrt(work_bits, rm);
  let ln_abs_z = abs_z.ln(work_bits, rm, cc);
  // atan2(b, a) via atan and quadrant adjustment
  let arg_z = {
    let zero = BigFloat::from_i32(0, work_bits);
    let pi = cc.pi(work_bits, rm);
    if !a.is_zero() {
      let ratio = b.div(&a, work_bits, rm);
      let atan_val = ratio.atan(work_bits, rm, cc);
      if a.is_positive() {
        atan_val
      } else if b.is_negative() {
        atan_val.sub(&pi, work_bits, rm)
      } else {
        atan_val.add(&pi, work_bits, rm)
      }
    } else if b.is_positive() {
      pi.div(&BigFloat::from_i32(2, work_bits), work_bits, rm)
    } else if b.is_negative() {
      pi.div(&BigFloat::from_i32(2, work_bits), work_bits, rm)
        .neg()
    } else {
      zero
    }
  };

  // Start sum: γ + ln|z| for real part, arg(z) for imaginary part
  let mut sum_re = euler_gamma.add(&ln_abs_z, work_bits, rm);
  let mut sum_im = arg_z;

  // Power series: Σ_{n=1}^{∞} z^n / (n * n!)
  // Track z^n as (pow_re, pow_im), start with z^1 = (a, b)
  let mut pow_re = a.clone();
  let mut pow_im = b.clone();
  let mut factorial = BigFloat::from_i32(1, work_bits); // n!

  let max_iterations = bits * 2 + 100;
  for n in 1..max_iterations {
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    if n > 1 {
      // z^n = z^(n-1) * z
      let new_re = pow_re.mul(&a, work_bits, rm).sub(
        &pow_im.mul(&b, work_bits, rm),
        work_bits,
        rm,
      );
      let new_im = pow_re.mul(&b, work_bits, rm).add(
        &pow_im.mul(&a, work_bits, rm),
        work_bits,
        rm,
      );
      pow_re = new_re;
      pow_im = new_im;
      factorial = factorial.mul(&n_bf, work_bits, rm);
    }
    // term = z^n / (n * n!)
    let denom = n_bf.mul(&factorial, work_bits, rm);
    let term_re = pow_re.div(&denom, work_bits, rm);
    let term_im = pow_im.div(&denom, work_bits, rm);

    sum_re = sum_re.add(&term_re, work_bits, rm);
    sum_im = sum_im.add(&term_im, work_bits, rm);

    // Check convergence
    let term_abs_sq = term_re.mul(&term_re, work_bits, rm).add(
      &term_im.mul(&term_im, work_bits, rm),
      work_bits,
      rm,
    );
    if term_abs_sq.is_zero() {
      break;
    }
    let sum_abs_sq = sum_re.mul(&sum_re, work_bits, rm).add(
      &sum_im.mul(&sum_im, work_bits, rm),
      work_bits,
      rm,
    );
    if let (Some(t_exp), Some(s_exp)) =
      (term_abs_sq.exponent(), sum_abs_sq.exponent())
    {
      // Compare squared magnitudes, so convergence is 2x the bit threshold
      if s_exp - t_exp > (2 * work_bits as i32) {
        break;
      }
    }
  }

  // Perform final addition at target `bits` precision to truncate mantissa
  let zero = BigFloat::from_i32(0, bits);
  let result_re = sum_re.add(&zero, bits, rm);
  let result_im = sum_im.add(&zero, bits, rm);

  Ok((result_re, result_im))
}

/// Compute the Euler-Mascheroni constant γ to the given precision.
fn compute_euler_gamma(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // High-precision string for Euler-Mascheroni constant (105 digits)
  let gamma_str = "0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495";
  BigFloat::parse(gamma_str, astro_float::Radix::Dec, bits, rm, cc)
}

fn compute_catalan(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // High-precision string for Catalan's constant G (OEIS A006752, ~105 digits)
  let catalan_str = "0.91596559417721901505460351493238411077414937428167213426649811962176301977625476947935651292611510624857";
  BigFloat::parse(catalan_str, astro_float::Radix::Dec, bits, rm, cc)
}

fn compute_glaisher(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Glaisher–Kinkelin constant A (OEIS A074962, ~105 digits).
  let glaisher_str = "1.28242712910062263687534256886979172776768892732500119206374002174040630883966455201507550549353290381";
  BigFloat::parse(glaisher_str, astro_float::Radix::Dec, bits, rm, cc)
}

fn compute_khinchin(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Khinchin's constant K_0 (OEIS A002210, ~105 digits).
  let khinchin_str = "2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799";
  BigFloat::parse(khinchin_str, astro_float::Radix::Dec, bits, rm, cc)
}

/// The decimal expansion of the base-b Champernowne constant — the digits
/// of 1, 2, 3, … concatenated after the radix point in base b — truncated
/// to `decimal_digits` decimal places, as a "0.ddd…" string. Computed
/// exactly: enough base-b digits accumulate into a big integer M so that
/// M/b^K determines the requested decimal digits, then long-divided.
pub(crate) fn champernowne_decimal_digits(
  base: i128,
  decimal_digits: usize,
) -> String {
  use num_bigint::BigInt;
  let k_needed = ((decimal_digits as f64 + 2.0) * std::f64::consts::LN_10
    / (base as f64).ln())
  .ceil() as usize
    + 2;
  let big_base = BigInt::from(base);
  let mut m = BigInt::from(0);
  let mut count = 0usize;
  let mut n: i128 = 1;
  'outer: loop {
    let mut digits = Vec::new();
    let mut v = n;
    while v > 0 {
      digits.push(v % base);
      v /= base;
    }
    for &d in digits.iter().rev() {
      m = &m * &big_base + BigInt::from(d);
      count += 1;
      if count >= k_needed {
        break 'outer;
      }
    }
    n += 1;
  }
  let ten_p = BigInt::from(10).pow(decimal_digits as u32);
  let b_k = big_base.pow(count as u32);
  let d = (&m * &ten_p) / &b_k;
  let s = d.to_string();
  format!(
    "0.{}{}",
    "0".repeat(decimal_digits.saturating_sub(s.len())),
    s
  )
}

/// The machine-precision value of ChampernowneNumber[base].
pub(crate) fn champernowne_f64(base: i128) -> f64 {
  champernowne_decimal_digits(base, 20).parse().unwrap_or(0.0)
}

fn compute_champernowne(
  base: i128,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;
  // Generous guard digits: the parsed decimal string must exceed the
  // BigFloat's internal precision comfortably, like the ~105-digit
  // hardcoded constant strings, or the displayed digit tail diverges from
  // wolframscript.
  let decimal_digits =
    (bits as f64 * std::f64::consts::LOG10_2).ceil() as usize + 20;
  let s = champernowne_decimal_digits(base, decimal_digits);
  BigFloat::parse(&s, astro_float::Radix::Dec, bits, rm, cc)
}

/// ListFourierSequenceTransform[{a0, a1, ..., an}, omega] — discrete-time Fourier transform.
///
/// Computes Sum[a_k * E^(-I * omega * k), {k, 0, n-1}].
pub fn list_fourier_sequence_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "ListFourierSequenceTransform".to_string(),
      args: args.to_vec().into(),
    });
  }

  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListFourierSequenceTransform".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if list.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  let omega = &args[1];

  // Build the sum: Sum[a_k * E^(-I * omega * k), {k, 0, n-1}]
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::syntax::BinaryOperator;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, coeff) in list.iter().enumerate() {
    if matches!(coeff, Expr::Integer(0)) {
      continue;
    }

    if k == 0 {
      // E^0 = 1, so just add the coefficient
      terms.push(coeff.clone());
    } else {
      // a_k * E^(-I * omega * k)
      let k_expr = Expr::Integer(k as i128);
      // -I * omega * k
      let exponent = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(-1),
          Expr::Identifier("I".to_string()),
          omega.clone(),
          k_expr,
        ]
        .into(),
      };
      let exp_term = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier("E".to_string())),
        right: Box::new(exponent),
      };
      let term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(exp_term),
      };
      terms.push(term);
    }
  }

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let sum = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  };

  evaluate_expr_to_expr(&sum)
}

/// Generic window function evaluator.
/// All window functions are defined on [-1/2, 1/2] and return 0 outside.
pub fn window_function_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    });
  }

  // For exact rational arguments, try exact evaluation
  if let Expr::FunctionCall {
    name: fname,
    args: fargs,
  } = &args[0]
    && fname == "Rational"
    && fargs.len() == 2
    && let (Some(n), Some(d)) =
      (try_eval_to_f64(&fargs[0]), try_eval_to_f64(&fargs[1]))
  {
    let x = n / d;
    if x.abs() > 0.5 {
      return Ok(Expr::Integer(0));
    }
  }

  let x = match try_eval_to_f64(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if x.abs() > 0.5 {
    return Ok(if matches!(&args[0], Expr::Real(_)) {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }

  let pi = std::f64::consts::PI;
  let val = match name {
    "HammingWindow" => 25.0 / 46.0 + 21.0 / 46.0 * (2.0 * pi * x).cos(),
    "HannWindow" => (1.0 + (2.0 * pi * x).cos()) / 2.0,
    "BlackmanWindow" => {
      0.42 + 0.5 * (2.0 * pi * x).cos() + 0.08 * (4.0 * pi * x).cos()
    }
    "DirichletWindow" => 1.0,
    "BartlettWindow" => 1.0 - 2.0 * x.abs(),
    "WelchWindow" => 1.0 - 4.0 * x * x,
    "CosineWindow" => (pi * x).cos(),
    "ConnesWindow" => {
      let t = 1.0 - 4.0 * x * x;
      t * t
    }
    "LanczosWindow" => {
      let arg = 2.0 * x;
      if arg.abs() < 1e-15 {
        1.0
      } else {
        (pi * arg).sin() / (pi * arg)
      }
    }
    "ExactBlackmanWindow" => {
      3946.0 / 18608.0
        + 9274.0 / 18608.0 * (2.0 * pi * x).cos()
        + 5388.0 / 18608.0 * (4.0 * pi * x).cos()
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // For exact arguments, try to return exact results
  if matches!(&args[0], Expr::Integer(_))
    || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational")
  {
    // Check for common exact values
    let rounded = fourier_round(val);
    if (rounded - rounded.round()).abs() < 1e-14 && rounded.abs() < 1e18 {
      return Ok(Expr::Integer(rounded.round() as i128));
    }
    // Try to express as a simple fraction
    if let Some((n, d)) = approximate_rational(rounded) {
      return Ok(make_rational(n, d));
    }
  }

  Ok(Expr::Real(val))
}

/// TukeyWindow[x] / TukeyWindow[x, alpha] — Tukey (tapered-cosine) window.
/// Default alpha = 2/3. The window is 0 outside [-1/2, 1/2], a flat 1 for
/// |x| <= (1-alpha)/2, and a raised-cosine taper
/// (1 + Cos[Pi (2|x| - 1 + alpha)/alpha]) / 2 in between. Exact arguments are
/// evaluated symbolically (so Cos simplifies to wolframscript's radical forms,
/// e.g. TukeyWindow[1/4] -> (1 + 1/Sqrt[2])/2); Real arguments numericize.
pub fn tukey_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "TukeyWindow".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let alpha = args.get(1).cloned().unwrap_or_else(|| make_rational(2, 3));

  let (xf, af) = match (try_eval_to_f64(x), try_eval_to_f64(&alpha)) {
    (Some(xf), Some(af)) => (xf, af),
    _ => return Ok(unevaluated()),
  };

  fn contains_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(_, _) => true,
      Expr::BinaryOp { left, right, .. } => {
        contains_real(left) || contains_real(right)
      }
      Expr::UnaryOp { operand, .. } => contains_real(operand),
      Expr::FunctionCall { args, .. } | Expr::List(args) => {
        args.iter().any(contains_real)
      }
      _ => false,
    }
  }
  let inexact = contains_real(x) || contains_real(&alpha);

  let ax = xf.abs();
  // Outside the window.
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  // Flat top.
  if ax <= (1.0 - af) / 2.0 {
    return Ok(if inexact {
      Expr::Real(1.0)
    } else {
      Expr::Integer(1)
    });
  }
  // Raised-cosine taper.
  if inexact {
    let theta = (2.0 * ax - 1.0 + af) / af;
    return Ok(Expr::Real(
      (1.0 + (std::f64::consts::PI * theta).cos()) / 2.0,
    ));
  }
  // Exact: build (1 + Cos[Pi (2 Abs[x] - 1 + alpha)/alpha]) / 2 and evaluate
  // symbolically so Cos simplifies (matching wolframscript's radical forms).
  use crate::syntax::BinaryOperator;
  let abs_x = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![x.clone()].into(),
  };
  let theta_num = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(2), abs_x].into(),
      },
      Expr::Integer(-1),
      alpha.clone(),
    ]
    .into(),
  };
  let theta = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(theta_num),
    right: Box::new(alpha),
  };
  let cos = Expr::FunctionCall {
    name: "Cos".to_string(),
    args: vec![Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Identifier("Pi".to_string())),
      right: Box::new(theta),
    }]
    .into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![Expr::Integer(1), cos].into(),
    }),
    right: Box::new(Expr::Integer(2)),
  })
}

/// True if `e` contains an inexact (machine) number, so a window function
/// numericizes; exact arguments stay symbolic.
fn window_arg_inexact(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      window_arg_inexact(left) || window_arg_inexact(right)
    }
    Expr::UnaryOp { operand, .. } => window_arg_inexact(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(window_arg_inexact)
    }
    _ => false,
  }
}

/// ParzenWindow[x] — de la Vallée Poussin window: 0 outside [-1/2, 1/2],
/// 1 - 24 x^2 + 48 |x|^3 for |x| <= 1/4, and 2 (1 - 2 |x|)^3 for 1/4 < |x| <= 1/2.
/// Exact arguments give the exact polynomial value (e.g. ParzenWindow[1/3] ->
/// 2/27); Real arguments numericize.
pub fn parzen_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator;
  let unevaluated = || Expr::FunctionCall {
    name: "ParzenWindow".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let xf = match try_eval_to_f64(x) {
    Some(v) => v,
    None => return Ok(unevaluated()),
  };
  let inexact = window_arg_inexact(x);
  let ax = xf.abs();
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if inexact {
    let v = if ax <= 0.25 {
      1.0 - 24.0 * xf * xf + 48.0 * ax * ax * ax
    } else {
      let t = 1.0 - 2.0 * ax;
      2.0 * t * t * t
    };
    return Ok(Expr::Real(v));
  }
  // Exact polynomial.
  let abs_x = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![x.clone()].into(),
  };
  let pow = |base: Expr, n: i128| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(base),
    right: Box::new(Expr::Integer(n)),
  };
  let times = |c: i128, e: Expr| Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(c), e].into(),
  };
  let expr = if ax <= 0.25 {
    // 1 - 24 x^2 + 48 Abs[x]^3
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Integer(1),
        times(-24, pow(x.clone(), 2)),
        times(48, pow(abs_x, 3)),
      ]
      .into(),
    }
  } else {
    // 2 (1 - 2 Abs[x])^3
    times(
      2,
      pow(
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(1), times(-2, abs_x)].into(),
        },
        3,
      ),
    )
  };
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// GaussianWindow[x] / GaussianWindow[x, sigma] — Gaussian window
/// E^(-x^2/(2 sigma^2)) on [-1/2, 1/2] (0 outside), default sigma = 3/10.
/// Exact arguments give E^(rational) (e.g. GaussianWindow[1/4] -> E^(-25/72));
/// Real arguments numericize.
pub fn gaussian_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator;
  let unevaluated = || Expr::FunctionCall {
    name: "GaussianWindow".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let sigma = args.get(1).cloned().unwrap_or_else(|| make_rational(3, 10));
  let (xf, sf) = match (try_eval_to_f64(x), try_eval_to_f64(&sigma)) {
    (Some(xf), Some(sf)) => (xf, sf),
    _ => return Ok(unevaluated()),
  };
  let inexact = window_arg_inexact(x) || window_arg_inexact(&sigma);
  let ax = xf.abs();
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if inexact {
    return Ok(Expr::Real((-xf * xf / (2.0 * sf * sf)).exp()));
  }
  // Exact: Exp[-x^2 / (2 sigma^2)].
  let pow = |base: Expr, n: i128| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(base),
    right: Box::new(Expr::Integer(n)),
  };
  let numer = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), pow(x.clone(), 2)].into(),
  };
  let denom = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(2), pow(sigma, 2)].into(),
  };
  let arg = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(numer),
    right: Box::new(denom),
  };
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Exp".to_string(),
    args: vec![arg].into(),
  })
}

/// BohmanWindow[x] — Bohman window: 0 outside [-1/2, 1/2], otherwise
/// (1 - 2|x|) Cos[2 Pi |x|] + Sin[2 Pi |x|] / Pi. Exact arguments evaluate the
/// symbolic form (e.g. BohmanWindow[1/4] -> 1/Pi); Real arguments numericize.
pub fn bohman_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator;
  let unevaluated = || Expr::FunctionCall {
    name: "BohmanWindow".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let xf = match try_eval_to_f64(x) {
    Some(v) => v,
    None => return Ok(unevaluated()),
  };
  let inexact = window_arg_inexact(x);
  let ax = xf.abs();
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if inexact {
    let arg = 2.0 * std::f64::consts::PI * ax;
    let v = (1.0 - 2.0 * ax) * arg.cos() + arg.sin() / std::f64::consts::PI;
    return Ok(Expr::Real(v));
  }
  // Exact: (1 - 2 Abs[x]) Cos[2 Pi Abs[x]] + Sin[2 Pi Abs[x]] / Pi.
  let abs_x = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![x.clone()].into(),
  };
  let two_pi_absx = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::Integer(2),
      Expr::Identifier("Pi".to_string()),
      abs_x.clone(),
    ]
    .into(),
  };
  let one_minus_2ax = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::Integer(1),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-2), abs_x].into(),
      },
    ]
    .into(),
  };
  let cos = Expr::FunctionCall {
    name: "Cos".to_string(),
    args: vec![two_pi_absx.clone()].into(),
  };
  let sin = Expr::FunctionCall {
    name: "Sin".to_string(),
    args: vec![two_pi_absx].into(),
  };
  let term1 = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![one_minus_2ax, cos].into(),
  };
  let term2 = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(sin),
    right: Box::new(Expr::Identifier("Pi".to_string())),
  };
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![term1, term2].into(),
  })
}

/// Try to express a float as a simple rational p/q with small denominator.
fn approximate_rational(val: f64) -> Option<(i128, i128)> {
  if val == 0.0 {
    return Some((0, 1));
  }
  // Try denominators up to 10000
  for d in 1..=10000i128 {
    let n = (val * d as f64).round() as i128;
    let approx = n as f64 / d as f64;
    if (approx - val).abs() < 1e-14 {
      let g = gcd(n.abs(), d);
      return Some((n / g, d / g));
    }
  }
  None
}

/// BandpassFilter[data, {omega1, omega2}] or BandpassFilter[data, {omega1, omega2}, n]
/// Applies a bandpass FIR filter using a windowed-sinc kernel with exact Hamming window.
/// Default order n = length of data. Default SampleRate = 1.
pub fn bandpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "BandpassFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Extract {omega1, omega2}
  let (omega1, omega2) = match &args[1] {
    Expr::List(freqs) if freqs.len() == 2 => {
      let o1 = match try_eval_to_f64(&freqs[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandpassFilter".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      let o2 = match try_eval_to_f64(&freqs[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandpassFilter".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      (o1, o2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandpassFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let w1 = omega1 / sample_rate;
  let w2 = omega2 / sample_rate;

  // Image input: separable 2D filter (row-then-column 1D passes).
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = bandpass_kernel(row_order, w1, w2);
    let col_kernel = bandpass_kernel(col_order, w1, w2);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  // Extract data list
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandpassFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let order = explicit_order.unwrap_or(items.len());
  let kernel = bandpass_kernel(order, w1, w2);

  // Try numeric path first
  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() == items.len() {
    // All numeric — fast path
    let result = convolve_edge_padded(&data, &kernel);
    return Ok(Expr::List(result.into_iter().map(Expr::Real).collect()));
  }

  // Symbolic path: convolve symbolically
  let result = convolve_edge_padded_symbolic(items, &kernel);
  Ok(Expr::List(result.into()))
}

/// Compute the bandpass FIR kernel of length n using windowed-sinc with exact Hamming window.
fn bandpass_kernel(n: usize, omega1: f64, omega2: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;

    // Exact Hamming window: alpha = 25/46
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };

    // Bandpass sinc kernel
    let sinc_bp = if t.abs() < 1e-15 {
      (omega2 - omega1) / std::f64::consts::PI
    } else {
      ((omega2 * t).sin() - (omega1 * t).sin()) / (std::f64::consts::PI * t)
    };

    kernel.push(w * sinc_bp);
  }
  kernel
}

/// Convolve data with kernel using edge-padding (repeat boundary values).
/// Returns output of same length as data.
pub fn convolve_edge_padded(data: &[f64], kernel: &[f64]) -> Vec<f64> {
  let n = kernel.len();
  let len = data.len();
  let left_pad = n / 2;

  let mut result = Vec::with_capacity(len);
  for m in 0..len {
    let mut sum = 0.0;
    for j in 0..n {
      let idx = m as isize + j as isize - left_pad as isize;
      let val = if idx < 0 {
        data[0]
      } else if idx >= len as isize {
        data[len - 1]
      } else {
        data[idx as usize]
      };
      sum += kernel[j] * val;
    }
    result.push(fourier_round(sum));
  }
  result
}

/// Symbolic convolution: for each output position, compute a symbolic sum of kernel[j] * data[idx].
/// Uses edge-padding (repeat boundary values) like the numeric version.
fn convolve_edge_padded_symbolic(data: &[Expr], kernel: &[f64]) -> Vec<Expr> {
  let n = kernel.len();
  let len = data.len();
  let left_pad = n / 2;

  let mut result = Vec::with_capacity(len);
  for m in 0..len {
    // Group kernel coefficients by data index
    let mut coeff_by_idx: Vec<f64> = vec![0.0; len];
    for j in 0..n {
      let idx = m as isize + j as isize - left_pad as isize;
      let data_idx = if idx < 0 {
        0
      } else if idx >= len as isize {
        len - 1
      } else {
        idx as usize
      };
      coeff_by_idx[data_idx] += kernel[j];
    }
    // Build symbolic sum
    let mut terms: Vec<Expr> = Vec::new();
    for (i, &coeff) in coeff_by_idx.iter().enumerate() {
      if coeff == 0.0 {
        continue;
      }
      terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Real(coeff), data[i].clone()].into(),
      });
    }
    let expr = if terms.is_empty() {
      Expr::Integer(0)
    } else if terms.len() == 1 {
      terms.pop().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      }
    };
    // Evaluate the symbolic expression to simplify
    match crate::evaluator::evaluate_expr_to_expr(&expr) {
      Ok(e) => result.push(e),
      Err(_) => result.push(expr),
    }
  }
  result
}

/// LowpassFilter[data, omega_c] or LowpassFilter[data, omega_c, n]
/// Applies a lowpass FIR filter using a windowed-sinc kernel with exact Hamming window.
pub fn lowpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "LowpassFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  let omega_c = match try_eval_to_f64(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "LowpassFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let wc = omega_c / sample_rate;

  // Image input: filter each channel row-by-row, then column-by-column
  // with the same 1D windowed-sinc kernel. Matches wolframscript's
  // separable 2D filter (`LowpassFilter[Image[…], ωc]`).
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = lowpass_kernel(row_order, wc);
    let col_kernel = lowpass_kernel(col_order, wc);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      // Row-wise pass: into a (h × w) scratch buffer.
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      // Column-wise pass.
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LowpassFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let order = explicit_order.unwrap_or(items.len());

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(Expr::FunctionCall {
      name: "LowpassFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  let kernel = lowpass_kernel(order, wc);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the lowpass FIR kernel of length n, normalized to sum to 1.
fn lowpass_kernel(n: usize, omega_c: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    let sinc_lp = if t.abs() < 1e-15 {
      omega_c / std::f64::consts::PI
    } else {
      (omega_c * t).sin() / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_lp);
  }
  // Normalize to sum to 1 (unity DC gain)
  let sum: f64 = kernel.iter().sum();
  if sum.abs() > 1e-15 {
    for v in &mut kernel {
      *v /= sum;
    }
  }
  kernel
}

/// HighpassFilter[data, omega_c] or HighpassFilter[data, omega_c, n]
/// Applies a highpass FIR filter using a windowed-sinc kernel with exact Hamming window.
pub fn highpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "HighpassFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  let omega_c = match try_eval_to_f64(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "HighpassFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let wc = omega_c / sample_rate;

  // Image input: separable 2D filter (row-then-column 1D passes).
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = highpass_kernel(row_order, wc);
    let col_kernel = highpass_kernel(col_order, wc);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HighpassFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let order = explicit_order.unwrap_or(items.len());

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(Expr::FunctionCall {
      name: "HighpassFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  let kernel = highpass_kernel(order, wc);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the highpass FIR kernel of length n.
/// Highpass = delta - lowpass (spectral inversion).
fn highpass_kernel(n: usize, omega_c: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    // Ideal highpass: delta(t) - sinc_lp(t) for the windowed version
    let sinc_hp = if t.abs() < 1e-15 {
      1.0 - omega_c / std::f64::consts::PI
    } else {
      -(omega_c * t).sin() / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_hp);
  }
  kernel
}

/// BandstopFilter[data, {omega1, omega2}] or BandstopFilter[data, {omega1, omega2}, n]
/// Applies a bandstop (notch) FIR filter.
pub fn bandstop_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "BandstopFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  let (omega1, omega2) = match &args[1] {
    Expr::List(freqs) if freqs.len() == 2 => {
      let o1 = match try_eval_to_f64(&freqs[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandstopFilter".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      let o2 = match try_eval_to_f64(&freqs[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandstopFilter".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      (o1, o2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandstopFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let w1 = omega1 / sample_rate;
  let w2 = omega2 / sample_rate;

  // Image input: separable 2D filter (row-then-column 1D passes).
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = bandstop_kernel(row_order, w1, w2);
    let col_kernel = bandstop_kernel(col_order, w1, w2);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandstopFilter".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let order = explicit_order.unwrap_or(items.len());

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(Expr::FunctionCall {
      name: "BandstopFilter".to_string(),
      args: args.to_vec().into(),
    });
  }

  let kernel = bandstop_kernel(order, w1, w2);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the bandstop FIR kernel of length n.
/// Bandstop = delta - bandpass (spectral inversion).
fn bandstop_kernel(n: usize, omega1: f64, omega2: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    let sinc_bs = if t.abs() < 1e-15 {
      1.0 - (omega2 - omega1) / std::f64::consts::PI
    } else {
      -((omega2 * t).sin() - (omega1 * t).sin()) / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_bs);
  }
  kernel
}

// ─── Numeric RootSum ────────────────────────────────────────────────

/// `N[RootSum[poly &, fn &]]`: find every (complex) root of `poly`, apply
/// `fn` to each, and sum. Returns `Some(Real)` (or a Plus[Real, Times[Real,
/// I]] for residual imaginary parts) on success and `None` for shapes that
/// don't fit — letting the caller fall back to the generic recursive `N`.
fn root_sum_n_eval(poly_arg: &Expr, fn_arg: &Expr) -> Option<Expr> {
  use crate::functions::polynomial_ast::extract_poly_coeffs;
  // poly_arg should be a Function whose body is a polynomial in #1.
  let poly_body = match poly_arg {
    Expr::Function { body } => body.as_ref(),
    _ => return None,
  };
  // Substitute Slot(1) → __rs_x__ to get a polynomial in a named variable
  // (extract_poly_coeffs needs an identifier).
  let var = "__rs_x__";
  let poly_in_var = crate::syntax::substitute_variable(
    poly_body,
    "#1",
    &Expr::Identifier(var.to_string()),
  );
  // Slot(1) inside Function bodies is stored differently from Identifier
  // "#1"; substitute that variant too.
  let poly_in_var = substitute_slot_with_identifier(&poly_in_var, 1, var);
  let expanded =
    crate::functions::polynomial_ast::expand_and_combine(&poly_in_var);
  let coeffs_i = extract_poly_coeffs(&expanded, var)?;
  if coeffs_i.len() < 2 {
    return None;
  }
  let coeffs_f: Vec<f64> = coeffs_i.iter().map(|&c| c as f64).collect();
  let roots = aberth_complex_roots(&coeffs_f);
  if roots.is_empty() {
    return None;
  }
  // Apply fn_arg (a Function or a builtin like Sin) to each complex root,
  // sum the results numerically.
  let mut sum_re = 0.0f64;
  let mut sum_im = 0.0f64;
  for (zr, zi) in &roots {
    let val = apply_fn_to_complex(fn_arg, *zr, *zi)?;
    sum_re += val.0;
    sum_im += val.1;
  }
  if sum_im.abs() < 1e-10 * sum_re.abs().max(1.0) {
    Some(Expr::Real(sum_re))
  } else {
    // Build Real + Real*I as Plus[..., Times[..., I]] so subsequent
    // Chop can drop the imaginary tail.
    Some(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(sum_re),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Real(sum_im), Expr::Identifier("I".to_string())]
            .into(),
        },
      ]
      .into(),
    })
  }
}

/// `N[Root[poly &, k]]`: the k-th root of the polynomial `poly`, ordered as
/// wolframscript does — real roots first in increasing order, then the
/// non-real roots by increasing real part and then increasing imaginary part.
/// Returns the root as a `Real` (or `Plus[Real, Times[Real, I]]` for complex
/// roots), or `None` when the shape doesn't fit.
fn root_n_eval(poly_arg: &Expr, k_arg: &Expr) -> Option<Expr> {
  use crate::functions::polynomial_ast::extract_poly_coeffs;
  let k = expr_to_i128(k_arg)?;
  if k < 1 {
    return None;
  }
  let poly_body = match poly_arg {
    Expr::Function { body } => body.as_ref(),
    _ => return None,
  };
  let var = "__root_x__";
  let poly_in_var = crate::syntax::substitute_variable(
    poly_body,
    "#1",
    &Expr::Identifier(var.to_string()),
  );
  let poly_in_var = substitute_slot_with_identifier(&poly_in_var, 1, var);
  let expanded =
    crate::functions::polynomial_ast::expand_and_combine(&poly_in_var);
  let coeffs_i = extract_poly_coeffs(&expanded, var)?;
  if coeffs_i.len() < 2 {
    return None;
  }
  let coeffs_f: Vec<f64> = coeffs_i.iter().map(|&c| c as f64).collect();
  let mut roots = aberth_complex_roots(&coeffs_f);
  if roots.is_empty() || (k as usize) > roots.len() {
    return None;
  }
  let is_real = |re: f64, im: f64| im.abs() < 1e-8 * (1.0 + re.abs());
  roots.sort_by(|a, b| {
    use std::cmp::Ordering;
    let ar = is_real(a.0, a.1);
    let br = is_real(b.0, b.1);
    match (ar, br) {
      (true, false) => Ordering::Less,
      (false, true) => Ordering::Greater,
      _ => a
        .0
        .partial_cmp(&b.0)
        .unwrap_or(Ordering::Equal)
        .then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)),
    }
  });
  let (re, im) = roots[(k - 1) as usize];
  if is_real(re, im) {
    Some(Expr::Real(re))
  } else {
    Some(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(re),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Real(im), Expr::Identifier("I".to_string())].into(),
        },
      ]
      .into(),
    })
  }
}

/// Replace every `Slot(k)` in `expr` with `Identifier(name)`. A companion
/// to `substitute_variable` so `Function`-bodied polynomials in `#1` can
/// be passed through the named-variable polynomial helpers.
fn substitute_slot_with_identifier(expr: &Expr, k: usize, name: &str) -> Expr {
  match expr {
    Expr::Slot(n) if *n == k => Expr::Identifier(name.to_string()),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_slot_with_identifier(e, k, name))
        .collect(),
    ),
    Expr::FunctionCall { name: fname, args } => Expr::FunctionCall {
      name: fname.clone(),
      args: args
        .iter()
        .map(|e| substitute_slot_with_identifier(e, k, name))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_slot_with_identifier(left, k, name)),
      right: Box::new(substitute_slot_with_identifier(right, k, name)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_slot_with_identifier(operand, k, name)),
    },
    _ => expr.clone(),
  }
}

/// Aberth's method for finding all complex roots of a real-coefficient
/// polynomial `p(x) = c_0 + c_1 x + … + c_n x^n`. Returns a vector of
/// `(re, im)` pairs of length `n` (degree of polynomial).
fn aberth_complex_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
  let n = coeffs.len() - 1;
  if n == 0 {
    return vec![];
  }
  if coeffs[n].abs() < 1e-300 {
    return vec![];
  }
  // Cauchy bound on |roots|: 1 + max_{i<n} |c_i / c_n|.
  let lead = coeffs[n].abs();
  let r = 1.0
    + coeffs[..n]
      .iter()
      .map(|c| c.abs() / lead)
      .fold(0.0f64, f64::max);
  let pi = std::f64::consts::PI;
  // Initial guesses on the Cauchy circle, slightly off-axis.
  let mut zs: Vec<(f64, f64)> = (0..n)
    .map(|k| {
      let theta = 2.0 * pi * (k as f64 + 0.5) / (n as f64) + 0.7;
      (r * theta.cos(), r * theta.sin())
    })
    .collect();
  for _ in 0..200 {
    let mut max_change: f64 = 0.0;
    let mut new_zs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for k in 0..n {
      let (zr, zi) = zs[k];
      let (pr, pi_) = poly_eval_complex(coeffs, zr, zi);
      let (dr, di) = poly_deriv_eval_complex(coeffs, zr, zi);
      let denom = dr * dr + di * di;
      if denom < 1e-300 {
        new_zs.push((zr, zi));
        continue;
      }
      // q = p(z) / p'(z) = (pr + i*pi) * (dr - i*di) / |p'|^2
      let qr = (pr * dr + pi_ * di) / denom;
      let qi = (pi_ * dr - pr * di) / denom;
      // sum_{j != k} 1 / (z_k - z_j)
      let mut sr = 0.0f64;
      let mut si = 0.0f64;
      for j in 0..n {
        if j == k {
          continue;
        }
        let dxr = zr - zs[j].0;
        let dxi = zi - zs[j].1;
        let d = dxr * dxr + dxi * dxi;
        if d < 1e-300 {
          continue;
        }
        sr += dxr / d;
        si += -dxi / d;
      }
      // Aberth: z_new = z - q / (1 - q*sum)
      let qsr = qr * sr - qi * si;
      let qsi = qr * si + qi * sr;
      let denr = 1.0 - qsr;
      let deni = -qsi;
      let dnorm = denr * denr + deni * deni;
      if dnorm < 1e-300 {
        new_zs.push((zr, zi));
        continue;
      }
      // q / (1 - q*sum)
      let dxr = (qr * denr + qi * deni) / dnorm;
      let dxi = (qi * denr - qr * deni) / dnorm;
      let new_zr = zr - dxr;
      let new_zi = zi - dxi;
      let change = (dxr * dxr + dxi * dxi).sqrt();
      if change > max_change {
        max_change = change;
      }
      new_zs.push((new_zr, new_zi));
    }
    zs = new_zs;
    if max_change < 1e-13 {
      break;
    }
  }
  zs
}

/// Horner-style polynomial evaluation at a complex point `z = zr + i*zi`.
fn poly_eval_complex(coeffs: &[f64], zr: f64, zi: f64) -> (f64, f64) {
  let mut rr = 0.0f64;
  let mut ri = 0.0f64;
  for &c in coeffs.iter().rev() {
    // r = r * z + c
    let nr = rr * zr - ri * zi + c;
    let ni = rr * zi + ri * zr;
    rr = nr;
    ri = ni;
  }
  (rr, ri)
}

/// Evaluate `p'(z)` at a complex point `z`. `p'(x) = sum_{k>=1} k * c_k *
/// x^(k-1)`.
fn poly_deriv_eval_complex(coeffs: &[f64], zr: f64, zi: f64) -> (f64, f64) {
  if coeffs.len() < 2 {
    return (0.0, 0.0);
  }
  let dcoeffs: Vec<f64> =
    (1..coeffs.len()).map(|k| (k as f64) * coeffs[k]).collect();
  poly_eval_complex(&dcoeffs, zr, zi)
}

/// Apply a simple unary numeric function (or a `Function` body) to a
/// complex argument. Supports `Sin`, `Cos`, `Tan`, `Exp`, `Log`, `Identity`,
/// and arbitrary `Function`-bodied expressions in #1 by substituting the
/// complex value back through the symbolic evaluator.
fn apply_fn_to_complex(fnexpr: &Expr, zr: f64, zi: f64) -> Option<(f64, f64)> {
  match fnexpr {
    Expr::Identifier(s) => match s.as_str() {
      "Sin" => Some(complex_sin(zr, zi)),
      "Cos" => Some(complex_cos(zr, zi)),
      "Identity" => Some((zr, zi)),
      _ => None,
    },
    Expr::Function { body } => {
      // Substitute Slot(1) → Complex[zr, zi] and evaluate symbolically.
      let value = if zi.abs() < 1e-300 {
        Expr::Real(zr)
      } else {
        Expr::FunctionCall {
          name: "Complex".to_string(),
          args: vec![Expr::Real(zr), Expr::Real(zi)].into(),
        }
      };
      let substituted = crate::syntax::substitute_slots(body, &[value]);
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
      let n_form = n_eval(&evaluated).ok()?;
      complex_from_expr(&n_form)
    }
    _ => None,
  }
}

/// Convert an evaluated numeric expression into a `(re, im)` pair when
/// possible. Recognises plain Reals, Complex[a, b], and Plus[real, im*I]
/// shapes.
fn complex_from_expr(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::Real(v) => Some((*v, 0.0)),
    Expr::Integer(n) => Some((*n as f64, 0.0)),
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      let r = match &args[0] {
        Expr::Real(v) => *v,
        Expr::Integer(n) => *n as f64,
        _ => return None,
      };
      let i = match &args[1] {
        Expr::Real(v) => *v,
        Expr::Integer(n) => *n as f64,
        _ => return None,
      };
      Some((r, i))
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut re = 0.0;
      let mut im = 0.0;
      for a in args {
        if let Some((r, i)) = complex_from_expr(a) {
          re += r;
          im += i;
        } else if let Some(coef) = imaginary_coeff(a) {
          im += coef;
        } else {
          return None;
        }
      }
      Some((re, im))
    }
    Expr::Identifier(s) if s == "I" => Some((0.0, 1.0)),
    // Times of (possibly complex) factors: multiply through using
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i, recursively. Enables shapes
    // like `Times[Plus[real, im*I], Complex[…, …]]` produced by `N[]`
    // on a rational of complex inputs to collapse to a single complex.
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let mut re = 1.0f64;
      let mut im = 0.0f64;
      for a in args {
        let (ar, ai) = complex_from_expr(a)?;
        let (nr, ni) = (re * ar - im * ai, re * ai + im * ar);
        re = nr;
        im = ni;
      }
      Some((re, im))
    }
    _ => imaginary_coeff(e).map(|im| (0.0, im)),
  }
}

/// Recognise `Times[real, I]` (in either nesting) and return the real
/// coefficient.
fn imaginary_coeff(e: &Expr) -> Option<f64> {
  if let Expr::FunctionCall { name, args } = e
    && name == "Times"
  {
    let mut coef = 1.0f64;
    let mut has_i = false;
    for a in args {
      match a {
        Expr::Identifier(s) if s == "I" => has_i = true,
        Expr::Real(v) => coef *= *v,
        Expr::Integer(n) => coef *= *n as f64,
        _ => return None,
      }
    }
    if has_i {
      return Some(coef);
    }
  }
  None
}

fn complex_sin(zr: f64, zi: f64) -> (f64, f64) {
  // Sin(a + bi) = Sin(a) Cosh(b) + i Cos(a) Sinh(b)
  (zr.sin() * zi.cosh(), zr.cos() * zi.sinh())
}

fn complex_cos(zr: f64, zi: f64) -> (f64, f64) {
  // Cos(a + bi) = Cos(a) Cosh(b) - i Sin(a) Sinh(b)
  (zr.cos() * zi.cosh(), -zr.sin() * zi.sinh())
}

/// The cosine-sum windows: w(x) = Σ aₖ Cos[2πkx] on [-1/2, 1/2], zero
/// outside, with the exact rational coefficients wolframscript uses (all
/// with plus signs — negative lobes come from coefficient magnitudes, e.g.
/// FlatTopWindow[1/4] = -54736843/1000000000). Exact arguments build the
/// sum symbolically so the trig tables produce exact rationals and radical
/// forms; Real arguments evaluate in floating point. Symbolic arguments
/// stay unevaluated, like wolframscript.
pub fn cosine_sum_window_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  // (numerators for a0, a1, …; common denominator)
  let (numerators, denom): (&[i128], i128) = match name {
    "BlackmanHarrisWindow" => (&[35875, 48829, 14128, 1168], 100000),
    "NuttallWindow" => (&[88942, 121849, 36058, 3151], 250000),
    "BlackmanNuttallWindow" => (&[3635819, 4891775, 1365995, 106411], 10000000),
    "FlatTopWindow" => (
      &[215578947, 416631580, 277263158, 83578947, 6947368],
      1000000000,
    ),
    // wolframscript's KaiserBesselWindow is this rational 4-term
    // cosine-sum approximation (fitted exactly from its outputs).
    "KaiserBesselWindow" => (&[402, 498, 99, 1], 1000),
    _ => return Ok(unevaluated()),
  };

  let Some(x) = try_eval_to_f64(&args[0]) else {
    return Ok(unevaluated());
  };
  let is_real = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..));
  if x.abs() > 0.5 {
    return Ok(if is_real {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if is_real {
    // Integer-numerator terms summed in coefficient order, divided once at
    // the end. This matches wolframscript's machine values at most tested
    // points; the remaining last-ULP drift at some arguments is the known
    // summation-order/libm divergence class (no single ordering reproduces
    // wolframscript everywhere — reverse and value-sorted orders each fix
    // some points while breaking others).
    let pi = std::f64::consts::PI;
    let val: f64 = numerators
      .iter()
      .enumerate()
      .map(|(k, &n)| n as f64 * (2.0 * pi * k as f64 * x).cos())
      .sum::<f64>()
      / denom as f64;
    return Ok(Expr::Real(val));
  }

  // Exact argument: build Σ (nₖ/d) Cos[2πk x] and let the evaluator's exact
  // trig tables fold it (rational at the common grid points, radicals
  // elsewhere).
  let rational = |n: i128, d: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
  };
  let terms: Vec<Expr> = numerators
    .iter()
    .enumerate()
    .map(|(k, &n)| {
      if k == 0 {
        rational(n, denom)
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            rational(n, denom),
            Expr::FunctionCall {
              name: "Cos".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Integer(2 * k as i128),
                  Expr::Identifier("Pi".to_string()),
                  args[0].clone(),
                ]
                .into(),
              }]
              .into(),
            },
          ]
          .into(),
        }
      }
    })
    .collect();
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })
}

/// ListZTransform[{a0, a1, …}, z] — the finite Z transform
/// Sum[a_k z^(-k-n)] with the optional third argument n shifting the
/// starting index (default 0). An empty list gives {}; non-list first
/// arguments emit ::arg1 and echo, like wolframscript.
pub fn list_z_transform_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ListZTransform".to_string(),
      args: args.to_vec().into(),
    })
  };
  let Expr::List(items) = &args[0] else {
    crate::emit_message(&format!(
      "ListZTransform::arg1: Expected a numeric array instead of {}.",
      crate::syntax::expr_to_output(&args[0])
    ));
    return unevaluated();
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  let shift: i128 = match args.get(2) {
    None => 0,
    Some(Expr::Integer(n)) => *n,
    Some(_) => return unevaluated(),
  };
  let terms: Vec<Expr> = items
    .iter()
    .enumerate()
    .map(|(k, a)| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        a.clone(),
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![args[1].clone(), Expr::Integer(-(k as i128) - shift)]
            .into(),
        },
      ]
      .into(),
    })
    .collect();
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// DiscreteHadamardTransform[list] — the sequency-ordered Walsh–Hadamard
/// transform: the input is zero-padded to the next power of two N, each
/// output is a ±-signed sum scaled by 1/Sqrt[N], and the rows come in
/// sequency (Walsh) order — natural row bitreverse(grayEncode(s)) for
/// output slot s, matching wolframscript. Exact input stays exact (odd
/// log2 sizes give radical forms like 15/(2*Sqrt[2])); non-numeric data
/// emits ::data.
pub fn discrete_hadamard_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "DiscreteHadamardTransform".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  let data_err = || {
    crate::emit_message(&format!(
      "DiscreteHadamardTransform::data: {} is not a numerical array.",
      crate::syntax::expr_to_output(&args[0])
    ));
    Ok(Expr::FunctionCall {
      name: "DiscreteHadamardTransform".to_string(),
      args: args.to_vec().into(),
    })
  };
  let Expr::List(items) = &args[0] else {
    return data_err();
  };
  if items.is_empty()
    || !items
      .iter()
      .all(|e| try_eval_to_f64(e).is_some_and(|v| v.is_finite()))
  {
    return data_err();
  }
  let is_real = items
    .iter()
    .any(|e| matches!(e, Expr::Real(_) | Expr::BigFloat(..)));
  let mut bits = 0u32;
  while (1usize << bits) < items.len() {
    bits += 1;
  }
  let n = 1usize << bits;
  if n > 1 << 16 {
    return unevaluated();
  }
  // Natural-order row for sequency slot s: bitreverse(gray(s)).
  let natural_row = |s: usize| -> usize {
    let g = s ^ (s >> 1);
    let mut r = 0usize;
    for b in 0..bits {
      if g & (1 << b) != 0 {
        r |= 1 << (bits - 1 - b);
      }
    }
    r
  };
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let mut out: Vec<Expr> = Vec::with_capacity(n);
  for s in 0..n {
    let r = natural_row(s);
    if is_real {
      let mut sum = 0.0f64;
      for (j, e) in items.iter().enumerate() {
        let v = try_eval_to_f64(e).unwrap_or(0.0);
        if (r & j).count_ones() % 2 == 0 {
          sum += v;
        } else {
          sum -= v;
        }
      }
      out.push(Expr::Real(sum / (n as f64).sqrt()));
    } else {
      let terms: Vec<Expr> = items
        .iter()
        .enumerate()
        .map(|(j, e)| {
          if (r & j).count_ones() % 2 == 0 {
            e.clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), e.clone()].into(),
            }
          }
        })
        .collect();
      let scaled = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          },
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::Integer(n as i128),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
              },
            ]
            .into(),
          },
        ]
        .into(),
      };
      out.push(eval(&scaled)?);
    }
  }
  Ok(Expr::List(out.into()))
}

/// CauchyWindow[x] / CauchyWindow[x, α] — 1/(1 + (2αx)²) on [-1/2, 1/2]
/// (default α = 3), zero outside; PoissonWindow[x] / [x, α] — E^(-2α|x|);
/// BartlettHannWindow[x] — 31/50 - (12/25)|x| + (19/50)Cos[2πx]. Exact
/// arguments evaluate symbolically (Poisson keeps wolframscript's
/// unfolded E^(-3/2) form); Reals numericize; symbolic arguments stay
/// unevaluated.
pub fn parametric_window_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    })
  };
  let max_args = if name == "BartlettHannWindow" { 1 } else { 2 };
  if args.is_empty() || args.len() > max_args {
    return unevaluated();
  }
  let Some(x) = try_eval_to_f64(&args[0]) else {
    return unevaluated();
  };
  let is_real = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..))
    || matches!(args.get(1), Some(Expr::Real(_) | Expr::BigFloat(..)));
  if !x.is_finite() {
    return unevaluated();
  }
  if x.abs() > 0.5 {
    return Ok(if is_real {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  // The α parameter (Cauchy/Poisson only; default 3).
  let alpha_expr = args.get(1).cloned().unwrap_or(Expr::Integer(3));
  let Some(alpha) = try_eval_to_f64(&alpha_expr) else {
    return unevaluated();
  };
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let fc = |n: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: n.to_string(),
    args: a.into(),
  };
  let rational = |p: i128, q: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(p), Expr::Integer(q)].into(),
  };
  if is_real {
    let pi = std::f64::consts::PI;
    let v = match name {
      "CauchyWindow" => 1.0 / (1.0 + (2.0 * alpha * x).powi(2)),
      "PoissonWindow" => (-2.0 * alpha * x.abs()).exp(),
      _ => {
        31.0 / 50.0 - 12.0 / 25.0 * x.abs() + 19.0 / 50.0 * (2.0 * pi * x).cos()
      }
    };
    return Ok(Expr::Real(v));
  }
  let expr = match name {
    // 1/(1 + (2 α x)²)
    "CauchyWindow" => fc(
      "Power",
      vec![
        fc(
          "Plus",
          vec![
            Expr::Integer(1),
            fc(
              "Power",
              vec![
                fc(
                  "Times",
                  vec![Expr::Integer(2), alpha_expr, args[0].clone()],
                ),
                Expr::Integer(2),
              ],
            ),
          ],
        ),
        Expr::Integer(-1),
      ],
    ),
    // E^(-2 α |x|)
    "PoissonWindow" => fc(
      "Power",
      vec![
        Expr::Identifier("E".to_string()),
        fc(
          "Times",
          vec![
            Expr::Integer(-2),
            alpha_expr,
            fc("Abs", vec![args[0].clone()]),
          ],
        ),
      ],
    ),
    // 31/50 - (12/25)|x| + (19/50)Cos[2 π x]
    _ => fc(
      "Plus",
      vec![
        rational(31, 50),
        fc(
          "Times",
          vec![rational(-12, 25), fc("Abs", vec![args[0].clone()])],
        ),
        fc(
          "Times",
          vec![
            rational(19, 50),
            fc(
              "Cos",
              vec![fc(
                "Times",
                vec![
                  Expr::Integer(2),
                  Expr::Identifier("Pi".to_string()),
                  args[0].clone(),
                ],
              )],
            ),
          ],
        ),
      ],
    ),
  };
  let _ = alpha;
  eval(&expr)
}
